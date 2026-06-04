# Temporal (`src/temporal/`)

> **Maturity:** mature · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose

The temporal subsystem advances a PDE solution in time. It provides explicit ODE integrators — forward Euler (`integrators::euler`) and classic 4-stage Runge–Kutta (`integrators::rk4`) — that step a solution forward by repeatedly calling a `system`'s right-hand-side and boundary updates. Integrators operate on **registry slots** (`field_ref` handles into a `sim_registry`) rather than owning field memory, so they never allocate; `simulation_cycle` owns the buffers. A `step_controller` tracks simulation time and step count, supplies the (fixed) CFL factors, and enforces a minimum-dt floor. `slot_ops.hpp` supplies the BLAS-like elementwise kernels (zero / axpy / accumulate) the integrators are built from.

## Where it lives

| File | Role |
| --- | --- |
| `integrator.hpp` / `integrator.cpp` | Public face: type-erased `std::variant<empty, rk4, euler>` wrapper `ccs::integrator` with a fixed 6-arg `operator()`, `std::visit` dispatch that forwards the right scratch-slot arity to each concrete integrator, and the `from_lua` factory (parses `simulation.integrator.type`). |
| `rk4.hpp` / `rk4.cpp` | Classic RK4: Butcher tableau `rki`/`rkf`, per-stage `submit_rhs_graph` + `update_boundary`, accumulate into the RK slot, final combine. The reference implementation for the slot/graph convention. |
| `euler.hpp` / `euler.cpp` | Forward Euler; documents the `deep_copy(output←u0)`-before-submit convention that keeps the pre-built RHS graph valid. |
| `empty_integrator.hpp` | `struct integrators::empty {}` — no-op integrator used for eigenvalue / zero-step runs; the default when no integrator is configured. |
| `slot_ops.hpp` | Header-only Kokkos kernels (`slot_zero`, `slot_assign_lc` = axpy, `slot_accumulate`) the integrators build on. Scalar-only (asserts on vectors); fences after each call. |
| `step_controller.hpp` / `step_controller.cpp` | Time/step bookkeeping over `bounded<int>`/`bounded<real>`, fixed CFL getters, `min_dt` floor via `check_timestep_size`, implicit conversions to `real`/`int`/`bool`, and `from_lua`. |
| `rk4_v2.t.cpp` / `euler_v2.t.cpp` | Single-step heat integration vs. a manufactured solution; also the canonical example of wiring registry slots + system + integrator by hand (outside `simulation_cycle`). |
| `step_controller.t.cpp` | Unit test for construction, `from_lua` parsing, `min_dt` floor, and `advance`/`bool` semantics (the only test here with no Kokkos runtime dependency). |
| `src/simulation/simulation_cycle.cpp` | (Not in this dir, but defines the contract.) Production caller: allocates the 4 slots, builds the RHS graph once, and drives `integrate(...)` + `controller.advance(...)` in `run()`. |

## Public API / entry points

All symbols live in `ccs` (or `ccs::integrators`). `real = double`.

### The integrator wrapper — `integrator.hpp`

```cpp
class integrator {
    std::variant<integrators::empty, integrators::rk4, integrators::euler> v;
public:
    integrator() = default;                       // == integrators::empty (first alternative)
    template <typename T> integrator(T&& t);      // construct from a concrete integrator

    void operator()(system& sys, sim_registry& reg,
                    field_ref u0, field_ref output,
                    field_ref scratch1, field_ref scratch2,
                    const step_controller& ctrl, real dt);

    static std::optional<integrator> from_lua(const sol::table&, const logs& = {});
};
```

- The **fixed 6-field signature** is the stable contract callers obey: `u0` (current solution), `output` (working slot, becomes the new solution), and `scratch1`, `scratch2`. Callers always pass 4 refs even though euler ignores one of them (see *Gotchas*).
- `from_lua` reads `simulation.integrator.type`: `"rk4"` → `rk4`, `"euler"` → `euler`, missing key → warns and returns `empty`, anything else → logs an error and returns `std::nullopt`.

### Concrete integrators — note the differing arities

```cpp
// rk4.hpp — needs TWO scratch slots
void integrators::rk4::operator()(system& sys, sim_registry& reg,
                                  field_ref u0, field_ref output,
                                  field_ref rk_rhs_ref, field_ref system_rhs_ref,
                                  const step_controller& ctrl, real dt);

// euler.hpp — needs ONE scratch slot
void integrators::euler::operator()(system& sys, sim_registry& reg,
                                    field_ref u0, field_ref output,
                                    field_ref system_rhs_ref,
                                    const step_controller& ctrl, real dt);

// empty_integrator.hpp
struct integrators::empty {};   // no operator(); the wrapper treats it as a no-op
```

`integrator::operator()` (in `integrator.cpp`) reconciles the arities: it forwards both scratch slots to `rk4`, only `scratch2` to `euler`, and does nothing for `empty`.

### Step controller — `step_controller.hpp`

```cpp
class step_controller {
    bounded<int> step; bounded<real> time;
    real h_cfl; real p_cfl; real min_dt;
public:
    step_controller() = default;
    step_controller(bounded<int> step, bounded<real> time,
                    real h_cfl, real p_cfl, real min_dt);

    std::optional<real> check_timestep_size(real dt) const; // dt >= min_dt ? dt : nullopt

    operator real() const;   // == simulation_time()
    operator int()  const;   // == simulation_step()
    operator bool() const;   // time && step  (i.e. still within max_time AND max_step)

    real simulation_time() const;
    int  simulation_step() const;
    void advance(real dt);            // time += dt; step += 1

    real parabolic_cfl()  const;      // p_cfl
    real hyperbolic_cfl() const;      // h_cfl

    static std::optional<step_controller> from_lua(const sol::table&, const logs& = {});
};
```

### Slot kernels — `slot_ops.hpp` (header-only, `inline`)

```cpp
void slot_zero(sim_registry& reg, field_ref ref);                                  // ref[i] = 0
void slot_assign_lc(sim_registry& reg, field_ref dst,                              // dst = src + coeff*rhs  (axpy)
                    field_ref src, real coeff, field_ref rhs);
void slot_accumulate(sim_registry& reg, field_ref dst, real coeff, field_ref src); // dst += coeff*src
```

Each iterates a slot's scalar buffers (`scalar_handle{s * layout_type::scalar_stride}` × `.all()` buffers), dispatches a `Kokkos::parallel_for` over the flat buffer, then calls `Kokkos::fence()`.

## How it works

### The slot/graph contract (read this first)

`simulation_cycle::run()` pre-allocates **exactly four registry slots** and reuses them every step:

| Slot | `field_ref` | Meaning |
| --- | --- | --- |
| 0 | `u0_ref`  | current solution (input) |
| 1 | `u1_ref`  | working / output slot (becomes next solution) |
| 2 | `rk_ref`  | RK accumulator (used by rk4; passed but unused by euler) |
| 3 | `srhs_ref`| system RHS output |

Before the loop, the RHS graph is **built once** (`sys.build_rhs_graph(reg, u1_ref, reg, srhs_ref)`, `simulation_cycle.cpp:75`) and bound to the *data pointers* of slots 1 and 3. Those pointers must stay stable for the lifetime of the captured graph. Consequences that propagate into the integrators:

- Integrators **`deep_copy_slot(output ← u0)`** at the start, rather than swapping `u0`/`output`. A swap would change which buffer the captured graph reads/writes and silently corrupt it. (`rk4.cpp:25`, `euler.cpp:21`; the rationale is spelled out in `simulation_cycle.cpp:122–125`.)
- After integration, `simulation_cycle` copies the result back with `deep_copy_slot(u0 ← u1)` (again, not a swap) for the next iteration.

### Forward Euler (`euler.cpp`)

```
time = ctrl                          // operator real()
deep_copy_slot(output ← u0)          // graph reads current data from the output slot
slot_zero(system_rhs)
submit_rhs_graph(output → system_rhs, time)
slot_assign_lc(output = u0 + dt*system_rhs)
update_boundary(output, time + dt)
```

### Classic RK4 (`rk4.cpp`)

Stage coefficients `rki = {0, ½, ½, 1}` and weights `rkf = {1/6, 1/3, 1/3, 1/6}`.

```
slot_zero(rk_rhs); slot_zero(system_rhs)
time = ctrl
deep_copy_slot(output ← u0)
for i in 0..3:                                 // each stage is a Kokkos profiling region
    if i > 0:
        slot_assign_lc(output = u0 + dt*rki[i]*system_rhs)   // form stage state
        update_boundary(output, time + dt*rki[i])
    submit_rhs_graph(output → system_rhs, time + dt*rki[i])  // evaluate RHS at stage
    slot_accumulate(rk_rhs += dt*rkf[i]*system_rhs)          // weighted sum
slot_assign_lc(output = u0 + 1.0*rk_rhs)       // final combine
update_boundary(output, time + dt)
```

Note stage 0 reuses the freshly copied `output` (= `u0`) directly, so there is no `slot_assign_lc` before the first `submit_rhs_graph`. Each stage's RHS evaluation, accumulation, and the whole stage are wrapped in `Kokkos::Profiling::ScopedRegion`s (`rk4::stage_i`, `rk4::rhs`, `rk4::accumulate`).

### The empty / zero-step path

`integrators::empty` is the first variant alternative, so a default-constructed `integrator` is also empty. It is selected when `simulation.integrator` is absent from the config (with a warn). It pairs with the **zero-step run**: `step_controller::from_lua` forces `max_step = 0` when neither `max_step` nor `max_time` is configured (the eigenvalue-analysis case). With `max_step = 0` the controller is falsy, so `simulation_cycle`'s `while (controller && ...)` loop never even calls the integrator — the no-op is a consistent companion to the zero-step controller and the eigenvalues system, not an executed code path in practice. See `eigenvalues.lua`.

### Dispatch flow

```
simulation_cycle::from_lua → integrator::from_lua → integrator{rk4|euler|empty}
simulation_cycle::run loop  → integrator::operator()(... 6 refs ...) → std::visit
                                   ├─ rk4   : integ(u0, output, scratch1, scratch2, ...)
                                   ├─ euler : integ(u0, output, scratch2, ...)        // scratch1 dropped
                                   └─ empty : (nothing)
```

## How to extend

**Add a new explicit integrator** (e.g. SSP-RK3, low-storage RK):

1. Create `integrators::<name>` in `src/temporal/<name>.hpp` + `.cpp`, copying the `rk4` pattern. The `operator()` should take `(system& sys, sim_registry& reg, field_ref u0, field_ref output, /* scratch refs */, const step_controller& ctrl, real dt)`. Build the update purely from `slot_zero` / `slot_assign_lc` / `slot_accumulate`. **Start with `reg.deep_copy_slot(output.slot, u0.slot)`** and call `sys.submit_rhs_graph` + `sys.update_boundary` once per stage — never swap slots.
2. Add the type to the `std::variant` in `integrator.hpp` (`integrator::v`).
3. Add an `else if constexpr` branch in `integrator::operator()` (`integrator.cpp`) mapping the alternative to your `operator()` (forwarding the scratch refs it needs from the fixed 4), and add a string case to `integrator::from_lua`.
4. Register sources/test in `src/temporal/CMakeLists.txt`: add the `.cpp` to the `shoccs-integrate` library, and add a `t-<name>_v2`-style test executable (link `Catch2::Catch2 shoccs-integrate Kokkos::kokkos`, label `"temporal"`). Copy `rk4_v2.t.cpp` as the test template.

**Scratch-slot budget:** the wrapper's fixed signature exposes only `scratch1`/`scratch2`, and `simulation_cycle` pre-allocates exactly 4 slots (u0, u1, rk, srhs). A new integrator that needs **more** than two scratch slots also requires changing `simulation_cycle`'s allocation block (`simulation_cycle.cpp:44–56`) and the wrapper signature.

**A genuinely adaptive controller** (error-controlled step rejection/retry) is *not* a drop-in: `step_controller` is fixed-CFL today (see below), so you would extend `step_controller` and the `simulation_cycle::run` loop, not just add an integrator.

## Gotchas & invariants

- **Slot kernels are scalar-only.** `slot_zero`/`slot_assign_lc`/`slot_accumulate` all `assert(n_vectors == 0 && "slot_ops: vector support not yet implemented")`. Production never trips this only because the lone vector-capable system (`inviscid_vortex`) is a stub whose `size()` returns `{}` (0 vectors). Adding a real vector PDE (e.g. Euler) will hit these asserts. Note `assert` is compiled out under `NDEBUG`, so a release build would silently produce wrong results instead of failing — but this is moot until a vector system exists.
- **Deep-copy, never swap.** The RHS graph is built once and bound to fixed slot data pointers (`simulation_cycle.cpp:75`); integrators must `deep_copy_slot(output ← u0)` to keep those pointers stable. Swapping invalidates the captured graph.
- **euler's pre-copy is load-bearing.** `euler.cpp` copies `u0 → output` *before* `submit_rhs_graph` specifically so the pre-built graph (bound to the output slot) reads the current solution — a non-obvious coupling between the integrator and `simulation_cycle`'s graph-binding choice.
- **Arity mismatch is hidden inside the wrapper.** `integrator::operator()` has a fixed 6-ref signature but `euler` only uses one scratch; `integrator.cpp` passes `scratch2` to euler and both scratches to rk4. Callers must always pass 4 refs regardless.
- **`step_controller` implicit conversions.** It converts implicitly to `real` (time), `int` (step), and `bool` (in-bounds). The integrators use `const real time = ctrl;` which relies on `operator real()`. This conversion soup is easy to misuse — passing a `step_controller` where an `int` step index is expected silently yields the step count.
- **`from_lua` zero-step trap.** If a `step_controller` config sets neither `max_step` nor `max_time`, `from_lua` forces `max_step = 0` (a zero-step run for eigenvalue analysis). Easy to hit accidentally if a config omits both.
- **Many fences per step.** Every `slot_ops` kernel ends with `Kokkos::fence()`, so RK4 fences several times per step. This is a known serial/perf cost, not a bug — relevant before optimizing.
- **Build status.** `t-rk4_v2`/`t-euler_v2` both pass (build fixed 2026-06-04 — was the Kokkos 5.1 `create_graph` API break, since resolved by migrating all call sites to the templated 1-arg form).

## Maturity & known gaps

**Verdict: mature.** Evidence: the subsystem is the live driver of the time-stepping loop — `simulation_cycle::run()` (`simulation_cycle.cpp:92`) invokes `integrate(...)` inside `run()`, and `from_lua` wires both the integrator and the step controller from config. `step_controller` is consumed across `io/` and every concrete system (`heat`, `scalar_wave`, `hyperbolic_eigenvalues`, `empty`, `inviscid_vortex`). The integrators implement the complete algorithms (full 4-stage Butcher tableau with per-stage boundary updates; euler matches the same slot convention). The range-v3 → Kokkos migration here is **finished**, and the files saw active maintenance through the graph-API and profiling phases (2026-03-27). Three registered tests exist and all pass (`t-step_controller`; the two integrator tests exercise a real one-step heat integration to 1e-13).

Per-item status (from verified audit flags):

- **`_v2` test naming** — *cosmetic tech-debt, not partial.* The v1 field-based `rk4.t.cpp`/`euler.t.cpp` and the old field-based `operator()` overloads were deleted/migrated in commit `03923f6` ("Phase 9.7a"); only the misleading `_v2` suffix on the surviving test files/targets remains. The migration is complete. Low-priority normalization: rename `rk4_v2.t.cpp → rk4.t.cpp`, `t-euler_v2 → t-euler`. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`step_controller` is fixed-CFL, not adaptive** — *mature for what it is; doc mismatch only.* There is no error estimator, PI controller, or dt growth/shrink. `timestep_size` (in the systems) returns a constant CFL-scaled value (`parabolic_cfl()·h²/(4ν)` for heat, `hyperbolic_cfl()·h` for wave) recomputed each step from current state; `check_timestep_size` only enforces the `min_dt` floor (returns `nullopt` → `simulation_cycle` aborts). `CLAUDE.md`'s Temporal entry calls it "adaptive time stepping," which overstates it. The class is complete and load-bearing — not a maturity downgrade, just a docs nit. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`integrators::empty`** — *intentional, used, but undertested.* A complete-by-design no-op tag; it survived a dedicated dead-code-removal pass (Phase 18). Reachable in production via `eigenvalues.lua` (no integrator key → `empty` + `max_step = 0`). Gap: zero direct test coverage and invisible unless you read `from_lua`. Not dead, not experimental. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`slot_ops.hpp` vector branch** — *partial (scaffolding).* The scalar path is mature/tested/production; the vector path is just an `assert` and is unreachable today (every system returns `nvectors == 0`). It blocks the advertised Euler-equations capability in the sense that finishing Euler requires implementing this branch. Consider upgrading the `assert` to a hard throw so a future vector system fails loudly even under `NDEBUG`. See [Cleanup Plan](../CLEANUP_PLAN.md).

There are **no dead (zero-caller) items** in this subsystem.

## Tests

| Test | Label | Covers |
| --- | --- | --- |
| `t-step_controller` (`step_controller.t.cpp`, via `add_unit_test`) | `temporal` | Default-ctor invariants; `from_lua` parsing (`max_step`, `max_time`, `min_dt`, `cfl.hyperbolic`/`cfl.parabolic`); `check_timestep_size` `min_dt` floor (both below- and above-floor); `advance`/`bool` semantics across multiple steps. No Kokkos runtime dependency. |
| `t-rk4_v2` (`rk4_v2.t.cpp`) | `temporal` | Full registry-based **single-step** integration of the `heat` system against a polynomial manufactured solution; asserts fluid-point error `WithinAbs(0, 1e-13)`. Custom `main` with `Kokkos::ScopeGuard`. |
| `t-euler_v2` (`euler_v2.t.cpp`) | `temporal` | Same as `t-rk4_v2` but for forward Euler (near-duplicate boilerplate, differing only by integrator type/arity). |

Run with `ctest --test-dir build -L temporal`.

**Not covered:** multi-step temporal order-of-accuracy / convergence (only one step is taken); the `integrators::empty` no-op path; the `integrator` variant wrapper and `from_lua` dispatch (rk4 vs euler vs empty vs invalid type); `scalar_wave` or any hyperbolic system through an integrator; multi-scalar systems (tests allocate per-scalar but heat is single-scalar); the `slot_ops` vector branch (asserted-out); and `step_controller`'s `min_dt`-floor → `simulation_cycle` early-return. No tests are disabled.

## Related docs

- [Fields reference](fields.md) — `sim_registry`, `field_ref`, `scalar_handle`, `allocate_scalar`, `deep_copy_slot`, `view`/`data`/`size` (the storage the integrators and `slot_ops` act on).
- [Systems reference](systems.md) — the `system` variant whose `submit_rhs_graph`/`build_rhs_graph`/`update_boundary`/`timestep_size`/`initialize`/`stats` the integrators drive each step.
- [Operators reference](operators.md) — the graph nodes a system's RHS is built from (what `submit_rhs_graph` actually evaluates).
- [Core types reference](core-types.md) — `real`/`integer`/`real3` and the Kokkos type aliases used here.
- [Utilities reference](utils.md) — `bounded<T>`, the counter type backing `step_controller`.
- [Simulation reference](simulation.md) — `simulation_cycle::run` (the 4-slot + build-graph-once loop) and `simulation_cycle::from_lua` calling `integrator::from_lua`.
- Legacy: `plans/06-temporal.md` (temporal migration notes), `plans/09-field-lifecycle.md` (Phase 9.7a removed the v1 field-based integrators/tests), `plans/17-kokkos5-upgrade-and-optimization.md` and `plans/18-cleanup-and-dedup.md` (graph-API + profiling/dedup work on these files). These are **pre-/mid-Kokkos rationale archives** — useful for history, not current API.
