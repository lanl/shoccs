# Systems (`src/systems/`)

> **Maturity:** partial (per-system: heat mature · scalar_wave mature · hyperbolic_eigenvalues mature-for-purpose · empty placeholder · inviscid_vortex dead stub) · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose

A *system* is one PDE problem: it bundles a mesh + boundary conditions + a discrete spatial operator (laplacian/gradient) and exposes the operations the time-stepping loop needs — spatial discretization (`rhs`), boundary enforcement (`update_boundary`), initial condition (`initialize`), diagnostics (`stats`), CFL timestep (`timestep_size`), and IO (`write`). A `std::variant`-based wrapper class `ccs::system` type-erases the concrete systems so the temporal integrators and `simulation_cycle` drive a single object regardless of which PDE is selected. Systems are the bridge between the registry-based field API (`sim_registry` + `field_ref`) and the operator/stencil machinery: they hold the operator, manage scratch/BC buffers, and emit per-step error statistics.

## Where it lives

| File | Role |
| --- | --- |
| `src/systems/system.hpp` | Public entry point: the `system` class wrapping `std::variant<empty, scalar_wave, inviscid_vortex, heat, hyperbolic_eigenvalues>` and all dispatch method declarations. |
| `src/systems/system.cpp` | `std::visit` dispatch for every method; `from_lua` factory mapping `simulation.system.type` strings to concrete systems; `if constexpr (requires{...})` gating of the graph path. |
| `src/systems/empty_system.hpp` / `.cpp` | Canonical API-contract template and the variant's default-constructible first alternative. The file to copy when adding a new system. |
| `src/systems/heat.hpp` / `.cpp` | Most complete / reference system: `dT/dt = k·lap(T)` with MMS source, Dirichlet+Neumann grid/object BCs, eager `rhs()` plus full `Kokkos::Graph` path. |
| `src/systems/heat.t.cpp` | Deepest test suite in the subsystem (7 `TEST_CASE`s, ~61 assertions): convergence, 2D, eval/stats correctness, graph-vs-eager equivalence. |
| `src/systems/scalar_wave.hpp` / `.cpp` | Second mature system: expanding spherical wave, RHS = `dot(grad_G, grad u)`; eager + graph paths. |
| `src/systems/scalar_wave.t.cpp` | Boundary correctness + gradient/dot values + graph-vs-eager equivalence. |
| `src/systems/hyperbolic_eigenvalues.hpp` / `.cpp` | Diagnostic system (no time integration): `stats()` computes the spectral radius of the gradient operator via `eigenvalue_visitor`. |
| `src/systems/hyperbolic_eigenvalues.t.cpp` | Single `TEST_CASE` asserting the max eigenvalue is ~0 for the configured stencil. |
| `src/systems/inviscid_vortex.hpp` / `.cpp` | Euler isentropic-vortex **stub**: every interface method is empty; only an unused analytic-solution namespace remains. Non-functional. |
| `src/systems/detail/scalar_system_utils.hpp` | Shared scalar-system helpers (`eval_at_locations`, `compute_scalar_stats`, `initialize_scalar_field`, `write_scalar_error`) used by both heat and scalar_wave. |
| `src/types.hpp` | Defines `system_stats { std::vector<real> stats; real wall_time_s; }`, consumed by `valid()` / `summary()` / `log()`. |
| `src/fields/field_registry.hpp` | Defines `system_size { nscalars, nvectors, d_size, rx_size, ry_size, rz_size }` returned by each system's `size()` to drive registry allocation, plus `extract_scalar_view`/`extract_scalar_span`. |

## Public API / entry points

### `ccs::system` (the wrapper — what the loop and integrators use)

`src/systems/system.hpp`:

```cpp
class system {
    std::variant<systems::empty,            // [0] default-constructed alternative
                 systems::scalar_wave,
                 systems::inviscid_vortex,
                 systems::heat,
                 systems::hyperbolic_eigenvalues> v;
public:
    system() = default;                                       // -> systems::empty
    template <typename T> system(T&& t);                      // wrap a concrete system

    static std::optional<system> from_lua(const sol::table&, const logs& = {});

    // status / diagnostics
    bool   valid(const system_stats&) const;                  // loop kill-switch
    real3  summary(const system_stats&) const;                // {Linf, min, max}
    void   log(const system_stats&, const step_controller&);
    system_size size() const;                                 // registry allocation token

    // RHS — two paths (see "How it works")
    void rhs(const sim_registry& creg, field_ref input,
             sim_registry& reg, field_ref output, real time);
    void build_rhs_graph(const sim_registry& creg, field_ref input,
                         sim_registry& reg, field_ref output);   // capture once
    void submit_rhs_graph(const sim_registry& creg, field_ref input,
                          sim_registry& reg, field_ref output, real time);

    // lifecycle
    void update_boundary(sim_registry& reg, field_ref ref, real time);
    void initialize(sim_registry& reg, field_ref ref, const step_controller&);
    system_stats stats(const sim_registry& reg, field_ref u0,
                       field_ref u1, const step_controller&) const;
    std::optional<real> timestep_size(const sim_registry& reg, field_ref u,
                                      const step_controller&) const;   // CFL-checked
    bool write(field_io& io, const sim_registry& reg, field_ref ref,
               const step_controller& c, real dt);
};
```

`from_lua` reads `simulation.system.type` and dispatches:

| `system.type` string | Concrete system | Notes |
| --- | --- | --- |
| `"heat"` | `systems::heat` | reads `system.diffusivity` (default 1.0) |
| `"scalar wave"` | `systems::scalar_wave` | note the **space**, not underscore; reads `system.center`/`system.radius` or first sphere shape; `system.max_error` (default 100) |
| `"eigenvalues"` | `systems::hyperbolic_eigenvalues` | diagnostic only |
| `"inviscid vortex"` | `systems::inviscid_vortex` | constructs the stub; **never runs** (see gaps) |
| anything else / missing | returns `std::nullopt` and logs an error | |

There is **no** Lua string that maps to `systems::empty`; it is only ever the default-constructed alternative.

### The concrete-system interface contract

Every concrete system must provide the method set demonstrated in `empty_system.hpp`. Method semantics:

- `bool valid(const system_stats&) const` — the loop's kill switch. heat/scalar_wave gate on `std::isfinite(stats[0]) && |stats[0]| <= limit`; `hyperbolic_eigenvalues` returns `true`; `empty` and `inviscid_vortex` return `false`.
- `system_size size() const` — `{nscalars, nvectors, d_size, rx_size, ry_size, rz_size}`. heat/scalar_wave return `{1, 0, m.size(), |Rx|, |Ry|, |Rz|}`; eigenvalues returns `{0, 0, ...}` (no field allocated).
- `void rhs(creg, input, reg, output, time)` — eager spatial discretization, writes into `output`'s buffers.
- `void update_boundary(reg, ref, time)` — writes boundary values into `ref`'s field buffers.
- `void initialize(reg, ref, step_controller)` — sets the initial condition.
- `system_stats stats(reg, u0, u1, step_controller) const` — computes the positional `stats[]` vector (layout below). heat/scalar_wave only use `u1`.
- `real timestep_size(reg, ref, step_controller) const` — predicted dt *before* the `system` wrapper applies `step_controller::check_timestep_size`.
- `bool write(io, reg, ref, step_controller, dt)` — emit fields to IO.
- `real3 summary(...)`, `void log(...)` — reporting.

Optional graph methods (opt-in, free functions on the concrete type, **not** in the variant signature): `void fill_source(real)`, `void build_rhs_graph(scalar_view u, scalar_span du)`, `void submit_rhs_graph()`. heat and scalar_wave implement all three.

### `system_stats::stats[]` positional layout

Defined only in `detail::compute_scalar_stats` (`scalar_system_utils.hpp`). For scalar systems the vector is, in order:

```
[0]  Linf error (max over D + R components)   <- used by valid()/summary()
[1]  u_min
[2]  u_max
[3]  err_d            [4]  err_d_idx
[5]  err_rx           [6]  idx_rx
[7]  err_ry           [8]  idx_ry
[9]  err_rz           [10] idx_rz
```

`hyperbolic_eigenvalues::stats` produces a one-element vector `{ -h·min(eigenvalue) }`.

### Shared helpers — `ccs::systems::detail` (`scalar_system_utils.hpp`)

```cpp
void eval_at_locations(const mesh& m, auto&& func, scalar_span out, bool parallel = true);
system_stats compute_scalar_stats(const mesh& m, const bcs::Object&, scalar_view u, scalar_view sol);
void initialize_scalar_field(const mesh& m, scalar_span u, scalar_span sol);
bool write_scalar_error(const mesh& m, const bcs::Object&, const bcs::Grid&,
                        scalar_view u, scalar_view sol, scalar_span error,
                        field_io&, std::span<const std::string> io_names,
                        const step_controller&, real dt);
```

`eval_at_locations` evaluates `func(real3 loc)` at every D location (cartesian product of `m.x()`/`y()`/`z()`) and every R cut-point. The `parallel` flag selects a `Kokkos::parallel_for` path vs a serial loop — see the MMS gotcha below.

## How it works

### Field model for scalar systems

Each scalar field lives in four buffers handled via `scalar_handle{0}`: the dense Cartesian field `D` plus three cut-cell boundary-point buffers `Rx`, `Ry`, `Rz` (one per ray-cast direction). Systems read fields with `extract_scalar_view(reg, ref, sh)` (const `scalar_view`) and write with `extract_scalar_span(reg, ref, sh)` (mutable `scalar_span`); raw pointers come from `reg.data(ref, sh.D()/Rx()/...)`. Selection descriptors carve out subsets for BC application:

- `m.fluid_desc()` — interior fluid D indices (where the PDE applies / error is measured).
- `m.dirichlet_object_desc(dir, object_bcs)` / `m.non_dirichlet_object_desc(dir, object_bcs)` — object cut-points by BC type, per direction.
- `for_each_grid_bc_desc<bcs::Dirichlet>(grid_bcs, m.extents(), fn)` — iterate grid-face plane descriptors of a given BC type.

### Heat RHS (reference pattern)

`heat::rhs` computes `du = k·lap(u, neumann) + (dS/dt − k·lap S)` where `S` is the manufactured solution (MMS):
1. `u_rhs = lap(u, neumann_view)` (the Neumann buffer carries gradient BC values set in `update_boundary`).
2. Scale all four buffers by `diffusivity` (`times_assign_scalar`).
3. If an MMS is present: `fill_source(time)` evaluates the source into member buffers, then `plus_assign_selected` scatters it onto fluid-D and non-Dirichlet object indices.
4. Zero the RHS at Dirichlet faces/objects (those values are owned by `update_boundary`, not the RHS).

### Two RHS paths (eager vs Kokkos::Graph)

| | eager | graph |
| --- | --- | --- |
| entry | `rhs(creg, input, reg, output, time)` | `build_rhs_graph(...)` once, then `submit_rhs_graph(...)` per step |
| body | computes everything inline each call | replays a pre-instantiated `Kokkos::Experimental::Graph` of `then_parallel_for` nodes; `fill_source(time)` runs before submit |
| time-dependence | `time` flows through directly | only the source buffers are time-dependent (refilled by `fill_source`); BC/operator structure is captured once |

`system::submit_rhs_graph` is `if constexpr (requires{ s.submit_rhs_graph(); })`-gated: a concrete system **without** the graph methods silently falls back to eager `rhs()`. `build_rhs_graph` is similarly gated on `requires{ s.build_rhs_graph(scalar_view, scalar_span); }`. The graph captures **raw data pointers** (`du.D.data()`, member buffers), so the captured slots must keep stable addresses for the graph's lifetime.

### Loop integration (`simulation_cycle::run`, `src/simulation/simulation_cycle.cpp`)

1. Allocate **4 slots per field** from `sys.size()`: `u0` (slot 0), `u1` (slot 1), `rk` (slot 2), `srhs` (slot 3). Zero-field systems (eigenvalues) leave refs at their default `{slot,0,0}` and slot-ops no-op.
2. `initialize(reg, u0)` → `deep_copy_slot(u1, u0)` → `update_boundary(reg, u0)`.
3. `stats(...)` → `log` → initial `write`.
4. `build_rhs_graph(reg, u1, reg, srhs)` once (heat/scalar_wave capture here; others no-op).
5. Loop `while (controller && sys.valid(stats))`: `timestep_size` → `integrate(sys, reg, u0, u1, rk, srhs, controller, dt)` (the integrator calls `submit_rhs_graph`/`rhs`) → `controller.advance(dt)` → `stats` → `write` → `log` → **`reg.deep_copy_slot(u0, u1)`** (a copy, *not* a slot swap — this is what keeps the graph's captured pointers valid).

### hyperbolic_eigenvalues (diagnostic)

`rhs`/`initialize`/`update_boundary` are intentionally **empty**; `timestep_size` returns a dummy `1.0`; `size()` allocates no field. The real work is in `stats()`: it builds an `eigenvalue_visitor` over the gradient operator (passing per-cut-point Dirichlet predicates from `object_bcs`), calls `grad.visit(v)`, and returns `-h·min(eigenvalues_real())` — the (scaled) spectral radius of the discrete gradient. `valid()` returns `true` so a Lua run constructs cleanly; it is meant to be queried, not time-stepped.

## How to extend

To add a new PDE system `systems::foo`:

1. **Copy `empty_system.hpp` as the contract.** Implement the full method set: `valid`, `log`, `summary`, `size`, `rhs`, `update_boundary`, `timestep_size`, `stats`, `initialize`, `write`, plus a static `from_lua`. For a scalar PDE, copy `heat.{hpp,cpp}` instead — it is the most complete pattern (operator member, MMS, grid+object Dirichlet/Neumann, buffer management).
2. **Reuse `detail/scalar_system_utils.hpp`** for `eval_at_locations` / `compute_scalar_stats` / `initialize_scalar_field` / `write_scalar_error` so your `stats`/`initialize`/`write` match the established `stats[]` layout and error conventions.
3. **(Optional) graph path:** add `fill_source(real)` (if time-dependent), `build_rhs_graph(scalar_view, scalar_span)`, and `submit_rhs_graph()`. Build the graph from **member (stable-pointer) scratch buffers** and the output `du` pointers. Your graph result must match eager `rhs()` exactly — mirror the `"... - graph matches eager"` test in `heat.t.cpp` / `scalar_wave.t.cpp`.
4. **Register in the variant:** add `systems::foo` to the `std::variant` in `system.hpp` and include its header.
5. **Wire `from_lua`:** add a `type == "foo"` branch in `system::from_lua` (`system.cpp`) that calls `systems::foo::from_lua`.
6. **Build + test:** add `foo.cpp` to `add_library(shoccs-system ...)` in `src/systems/CMakeLists.txt`, and add an `add_executable(t-foo foo.t.cpp)` / `add_test` block with `set_tests_properties(t-foo PROPERTIES LABELS "systems")`. Tests need a custom `main()` with `Kokkos::ScopeGuard` (link `Catch2::Catch2`, not `WithMain`).
7. **Multi-component systems** (e.g. a real Euler solver) need `size()` to return `nscalars > 1` or `nvectors > 0`; the loop allocates 4 slots for each component automatically.

To add a new end-to-end regression for an existing system, mirror the heat case in `src/simulation/simulation_cycle.t.cpp` (currently heat-only).

## Gotchas & invariants

- **`valid()` is the loop kill-switch.** `simulation_cycle` runs `while (controller && sys.valid(stats))`. `inviscid_vortex::valid()` and `empty::valid()` return `false`, so selecting `type="inviscid vortex"` builds a `system` that **never time-steps** — a silent no-op, not an error.
- **`"scalar wave"` has a space**, not an underscore, in the Lua `system.type` string. The class is `scalar_wave` but the config key is `scalar wave`.
- **Two RHS paths must stay in sync.** A graph-capable system must reproduce its eager `rhs()` exactly; the `"graph matches eager"` tests guard this.
- **Graph captures raw pointers; the loop uses `deep_copy_slot`, not slot-swap**, to keep `u0`/`u1`/`srhs` data addresses stable across steps. A new graph-capturing system must respect this — capture **member** scratch buffers, never temporaries.
- **Graph methods are silently optional.** Forgetting `build_rhs_graph`/`submit_rhs_graph` is *not* a compile error (the `if constexpr (requires{...})` dispatch just falls back to eager). Watch for unexpectedly slow systems that you *thought* were graph-accelerated.
- **MMS thread-safety.** `eval_at_locations` takes `parallel = m_sol.is_thread_safe()`. Lua-based manufactured solutions are **not** thread-safe and must use the serial path; passing `parallel=true` with a Lua MMS is a data race. Compiled (functor) MMS use the parallel path.
- **`stats[]` is an untyped positional `std::vector<real>`.** Index 0 is the Linf consumed by `valid()`/`summary()`; the full layout exists only in `detail::compute_scalar_stats`. Easy to desync a producer and a consumer — change one, change both.
- **`timestep_size` is checked twice.** The concrete system returns a *predicted* dt; `system::timestep_size` then runs it through `step_controller::check_timestep_size`, which may return `std::nullopt` (loop aborts as "timestep too small").
- **Flattened git history.** All `src/systems` files share one import commit date, so per-file recency is not a maturity signal here — judge by code completeness and tests instead.

## Maturity & known gaps

Overall verdict **partial** because maturity is per-system. Evidence is from the audit (`/tmp/audit/subsystems/systems.json`) cross-checked against source.

- **heat — mature.** Full eager + graph RHS, MMS source/BC handling (Dirichlet + Neumann, grid + object), 7 `TEST_CASE`s / ~61 assertions, and the only system exercised end-to-end (`simulation_cycle.t.cpp`, `euler_v2.t.cpp`, `rk4_v2.t.cpp` all use `type="heat"`).
- **scalar_wave — mature.** Complete eager + graph RHS, real numeric tests (boundary correctness, gradient/dot values, graph-vs-eager). **Gap:** never exercised through the full `integrate→advance→stats→write` loop in an automated test (only heat is). A scalar_wave-specific loop regression (e.g. graph pointer-stability, `hyperbolic_cfl` over many steps) could go uncaught. Recommended follow-up: add a `type="scalar wave"`/rk4 case to `simulation_cycle.t.cpp`. Real Lua configs that drive it exist (`lua-configs/brady_livescu_4_3*.lua`). See [Cleanup Plan](../CLEANUP_PLAN.md).
- **hyperbolic_eigenvalues — mature for its narrow purpose.** It is a diagnostic, not an integrator; empty `rhs`/`initialize`/`update_boundary` are *by design*, and its real output (`stats()` spectral radius) is unit-tested. Do not mistake the empty stubs for incompleteness.
- **empty — experimental / intentional placeholder.** Self-documented in `empty_system.hpp` as the API template and the variant's default-constructible alternative. Compiled in and reachable as the default-constructed `system`, but **not Lua-selectable** and has no dedicated test. Keep. (The plans intended a `static_assert(SystemV2<systems::empty>)` concept check that is absent from current source — a low-risk hardening.)
- **inviscid_vortex — dead stub (Euler is NOT implemented).** Every one of its 9 interface methods is empty/trivial: `valid()` returns `false` (so the loop body can never execute even if constructed), `timestep_size()` is hardcoded `1.0`, `size()`/`stats()`/`summary()` return `{}`, `write()` returns `false`. It is wired into the variant and `from_lua` (`type="inviscid vortex"`) but **no shipped Lua config selects it** and there is **no test**. The former `#if 0` Euler algorithm core was already deleted (plan 15.8a). Audit recommendation: **document-as-experimental** (it is the documented placeholder for future Euler/isentropic-vortex verification work); the next step if abandoned is to delete the file + variant arm + `from_lua` branch + CMake entry. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **inviscid_vortex `solution::{rho,rhoU,rhoV,rhoE,P}` structs and the `vars` enum — confirmed dead, safe to delete.** Zero external callers: the `vars` enum is never referenced anywhere; the analytic-solution structs reference only each other and no `inviscid_vortex` method touches them (`initialize()` is empty). These are vestigial fragments of the abandoned Euler implementation (the formulas are the standard textbook isentropic vortex). Audit recommendation: **delete lines 13–116 of `inviscid_vortex.cpp`** (the `vars` enum + the `solution` namespace); leave the enclosing class alone as a separate decision. See [Cleanup Plan](../CLEANUP_PLAN.md).

Doc-staleness note for newcomers: `CLAUDE.md` (Project Overview) lists "Euler equations" as a solved PDE — this is **aspirational**, not real. The only Euler-flavored code is the `inviscid_vortex` stub above.

## Tests

Three binaries, all labeled `systems` (`ctest --test-dir build -L systems`):

- **t-heat** (`heat.t.cpp`) — most thorough. `TEST_CASE`s: `heat - E2`, `heat - E2 - floating`, `2D heat - E2 - floating`, `heat - eval_at_locations correctness`, `heat - stats reduction correctness`, `heat - eval_at_locations parallel path (gaussian MMS)`, `heat - graph matches eager`. Covers convergence, 2D, the `detail/` helpers, and graph-vs-eager equivalence.
- **t-scalar_wave** (`scalar_wave.t.cpp`) — `scalar_wave - update_boundary` (boundary + gradient/dot values) and `scalar_wave - graph matches eager`; also asserts `stats[0] == 0` at `t=0`.
- **t-hyperbolic_eigenvalues** (`hyperbolic_eigenvalues.t.cpp`) — single `TEST_CASE` asserting the max eigenvalue ~= 0.

**Not directly tested:** `inviscid_vortex` (no `.t.cpp` exists; consistent with it being a stub), `empty` (covered only implicitly as the default variant), and `detail/scalar_system_utils.hpp` (exercised transitively via heat/scalar_wave — the count-0 / empty-R defensive branches are only indirectly covered). End-to-end loop coverage (`simulation_cycle.t.cpp`, temporal `euler_v2.t.cpp`/`rk4_v2.t.cpp`) uses **`type="heat"` only**; scalar_wave and eigenvalues are only exercised in their own unit tests. No disabled/commented-out tests in `CMakeLists.txt`.

Build/test status: all three targets **pass** (build fixed 2026-06-04). The earlier audit recorded them as `FAIL(link)` from a repo-wide Kokkos version mismatch (`libkokkoscore.so.5.0` vs installed 5.1.1); that was the Kokkos 5.1 `create_graph` API break, resolved by migrating call sites to `create_graph<execution_space>(closure)`, and a reconfigure + rebuild confirms them green (`ctest --test-dir build` = 47/48).

## Related docs

- [Fields reference](fields.md) — `sim_registry`, `field_ref`, `scalar_view`/`scalar_span`, `expr`, `selection_desc` (the storage and algebra systems sit on).
- [Operators reference](operators.md) — `laplacian`, `gradient`, `eigenvalue_visitor`, and the `add_graph_nodes` graph hooks the systems call.
- [Mesh reference](mesh.md) — `fluid_desc` / `dirichlet_object_desc` / `non_dirichlet_object_desc` selection descriptors and cut-cell `Rx/Ry/Rz` geometry.
- [Stencils reference](stencils.md) — `stencil::from_lua` and the scheme coefficients operators are built from.
- [Temporal reference](temporal.md) — `euler`/`rk4` integrators and `step_controller` that drive `system` per step.
- [Simulation reference](simulation.md) — `simulation_cycle::run` (the loop described above) and the builder that calls `system::from_lua`.
- Legacy: `plans/05-systems.md` (pre/intra-Kokkos systems migration notes), `plans/09-field-lifecycle.md`, `plans/15-code-smells.md` (records the deletion of the `inviscid_vortex` Euler core in 15.8a). These are **pre-/mid-Kokkos rationale archives** — useful for history, not current API.
