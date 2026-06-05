# Method of Manufactured Solutions (`src/mms/`)

> **Maturity:** mature · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose

The `mms` subsystem implements the Method of Manufactured Solutions (MMS) for verification and convergence testing. You pick an exact ("manufactured") solution `u*(t, x)`, substitute it into the PDE to derive the forcing term it must satisfy, run the solver with that forcing, and measure the computed solution against `u*` to confirm the discretization achieves its design order of accuracy. The subsystem provides a type-erased `manufactured_solution` value type that wraps any object supplying the exact solution and its analytic derivatives (time derivative, gradient, divergence, laplacian) at any `(time, location)`. It is the verification backbone for the `heat` system.

## Where it lives

| File | Role |
| --- | --- |
| `src/mms/manufactured_solutions.hpp` | Core header: the `ManufacturedSolution` concept, the type-erased `manufactured_solution` value type (clone-based deep copy over `any_sol`/`any_sol_impl`), the evaluation API, `is_thread_safe()`, and the `from_lua` declaration. Every consumer includes this. |
| `src/mms/mms.cpp` | Implements `manufactured_solution::from_lua` (parses `simulation.manufactured_solution`, dispatches on `type = "gaussian"`/`"lua"`) and the `build_ms_gauss(dims, ...)` dimension dispatcher. |
| `src/mms/gauss.hpp` | Shared `struct gauss` data (center/variance/amplitude/frequency vectors + span ctor) and declarations of `build_ms_gauss1d/2d/3d`. |
| `src/mms/gauss1d.cpp` | 1D Gaussian backend: closed-form value/ddt/gradient/laplacian (divergence = 0). The template the 2D/3D variants copy. |
| `src/mms/gauss2d.cpp` | 2D Gaussian backend (closed-form, hand-expanded laplacian). |
| `src/mms/gauss3d.cpp` | 3D Gaussian backend (closed-form, large hand-expanded laplacian). |
| `src/mms/lua_mms.hpp` | Lua-backed backend: holds a `sol::table` plus `std::function` callbacks; `thread_safe = false`; copy/move re-bind the callbacks from the table by key. |
| `src/mms/lua_mms.cpp` | `lua_mms` ctor (validates `call`/`ddt`/`grad`/`div`/`lap` are functions, logs errors) and `from_lua`. |
| `src/mms/mms.t.cpp` | Unit test (`t-mms`): gauss1d/2d/3d + lua cases with hardcoded reference numerics. The only direct test of this subsystem. |
| `src/mms/CMakeLists.txt` | Builds `shoccs-mms` (links `fields`, `sol2`, `lua`, `spdlog`) and registers `add_unit_test(mms "mms" shoccs-mms)`. |
| `src/systems/heat.cpp` | Sole production consumer; the reference for integrating MMS into a system (source term, IC, Dirichlet/Neumann BCs, error stats). |

## Public API / entry points

All symbols live in namespace `ccs`.

### The concept

```cpp
template <typename M>
concept ManufacturedSolution = requires(const M& ms, real time, const real3& loc, int dim) {
    { ms(time, loc) }            -> std::same_as<real>;
    { ms.ddt(time, loc) }        -> std::same_as<real>;
    { ms.gradient(time, loc) }   -> std::same_as<real3>;
    { ms.divergence(time, loc) } -> std::same_as<real>;
    { ms.laplacian(time, loc) }  -> std::same_as<real>;
};
```

Any backend must supply all five methods. `divergence` is required by the concept even though every shipped backend leaves it at 0.0 (see Gotchas).

### The value type

`class manufactured_solution` is a copyable, type-erased holder. It owns a heap `any_sol*` and deep-copies via `clone()`.

```cpp
manufactured_solution();                                    // null ("no MMS"); operator bool == false
manufactured_solution(const manufactured_solution&);        // clone() deep copy
manufactured_solution(manufactured_solution&&);             // steals pointer

template <ManufacturedSolution T>                            // converting ctor from any concept-satisfying type
manufactured_solution(T&& other);                           // e.g. return {gauss1d{...}};

explicit operator bool() const;                             // true if a backend is held
bool is_thread_safe() const;                                // false when null OR backend opts out

// Primary factory. Reads tbl["manufactured_solution"], dispatches on type.
static std::optional<manufactured_solution>
from_lua(const sol::table& tbl, int dims = 3, const logs& logger = {});
```

Point-evaluation methods (the ones consumers actually call) take `(time, loc)`:

```cpp
real  operator()(real time, const real3& loc) const;   // exact solution u*(t, x)
real  ddt(real time, const real3& loc) const;          // ∂u*/∂t
real3 gradient(real time, const real3& loc) const;     // ∇u*
real  divergence(real time, const real3& loc) const;   // ∇·u* (0 for all shipped backends)
real  laplacian(real time, const real3& loc) const;    // ∇²u*
```

Each also has a generic-tuple `loc` overload (constrained on `!std::same_as<real3, ...>`) that destructures `std::get<0..2>(loc)` into a `real3` — handy when feeding tuple-like coordinate handles.

> **Note:** the methods after the converting constructor (`operator=`, the evaluation methods, `from_lua`) are over-indented in the header. That is a formatting artifact; they are all members of `manufactured_solution`, not nested in a sub-scope.

### Gaussian factories (`gauss.hpp`)

```cpp
struct gauss {                                          // shared data; the three backends inherit it
    std::vector<real3> center, variance;
    std::vector<real>  amplitude, frequency;
    gauss() = default;
    gauss(std::span<const real3> center, std::span<const real3> variance,
          std::span<const real> amplitude, std::span<const real> frequency);
};

manufactured_solution build_ms_gauss1d(std::span<const real3> center, std::span<const real3> variance,
                                       std::span<const real> amplitude, std::span<const real> frequency);
manufactured_solution build_ms_gauss2d(/* same */);
manufactured_solution build_ms_gauss3d(/* same */);
```

`build_ms_gauss(int dims, ...)` in `mms.cpp` dispatches to the `1d/2d/3d` factory by `dims` and returns an **empty** `manufactured_solution` for `dims` outside `{1,2,3}`.

The Gaussian solution is a sum of time-modulated Gaussians: each entry `i` contributes
`amplitude[i] * cos(time * frequency[i]) * exp(-0.5 * Σ_d ((x_d - center[i][d]) / variance[i][d])²)`,
over only the active dimensions (1, 2, or 3). These backends are pure math and thread-safe.

### Lua backend (`lua_mms.hpp`)

```cpp
struct lua_mms {
    static constexpr bool thread_safe = false;          // NOT thread-safe
    sol::table tbl;
    std::function<real(real, const real3&)>                 call_, ddt_, divergence_, laplacian_;
    std::function<std::tuple<real,real,real>(real, const real3&)> gradient_;

    lua_mms(const sol::table& tbl);                     // validates call/ddt/grad/div/lap are functions
    // copy/move ctors + assignment RE-BIND the std::functions by re-reading keys from the table
    static std::optional<manufactured_solution> from_lua(const sol::table& tbl);
};
```

## How it works

1. **Configuration.** A Lua config supplies a `simulation.manufactured_solution` table. `manufactured_solution::from_lua(tbl, dims, logger)`:
   - returns `std::nullopt` (logged at `info`) if no `manufactured_solution` table is present;
   - lowercases `type` and dispatches:
     - `"gaussian"` → loops over array entries `ms[1], ms[2], ...`, reads `center`/`variance` (1-indexed Lua arrays, missing components default to 0.0) plus scalar `amplitude`/`frequency`, then calls `build_ms_gauss(dims, ...)` — but only if at least one entry was parsed (`center.size() > 0`);
     - `"lua"` → reads `call`/`ddt`/`grad`/`div`/`lap` as `sol::optional<std::function<...>>`. **All five** must be present (the gate at `mms.cpp:82` is `call && ddt && grad && div && lap`) or it logs an error and returns `nullopt`. On success it builds a `lua_mms`;
     - anything else → logs an error listing the valid types and returns `nullopt`.

2. **Type erasure.** Whichever concrete backend is built, the converting constructor wraps it in a `manufactured_solution::any_sol_impl<M>` behind the abstract `any_sol`. Virtual `operator()/ddt/gradient/divergence/laplacian` forward to the backend; `clone()` enables value-semantic deep copy; `is_thread_safe()` reports `M::thread_safe` if the backend declares it (defaulting to `true` otherwise).

3. **Consumption (in `heat`).** `heat::from_lua` builds the MMS (`heat.cpp:91`) and stores it as `manufactured_solution m_sol`, falling back to a default-constructed (null) object when `from_lua` returns nullopt:
   ```cpp
   auto ms_opt = manufactured_solution::from_lua(tbl, mesh_opt->dims(), logger);
   auto t = ms_opt ? MOVE(*ms_opt) : manufactured_solution{};
   ```
   Then, at runtime, `heat` evaluates the analytic pieces over mesh locations via `detail::eval_at_locations(mesh, lambda, scalar_span, parallel)`:
   - **Source term** (`fill_source`, `heat.cpp:114`): `S = ddt(t,x) − diffusivity · laplacian(t,x)` — this is exactly the forcing that makes `u*` an exact solution of the heat equation `∂u/∂t = k ∇²u + S`.
   - **Initial condition / exact field** (`heat.cpp:306`, `:376`): `operator()(time, loc)`.
   - **Boundary conditions** (`heat.cpp:337`): Dirichlet from `operator()`, Neumann from `gradient(time, loc)[dir]`.
   - **Error stats / output** (`heat.cpp:386` onward): L∞ error of the computed field vs `operator()`.
   - Each call passes `m_sol.is_thread_safe()` as the `parallel` flag, so Gaussian MMS runs in parallel and Lua MMS forces a serial path.

## How to extend

### Add a new closed-form (math) manufactured solution

1. Write a `struct` satisfying `ManufacturedSolution` — the five methods with exact signatures `operator()(real, const real3&) -> real`, `ddt(...) -> real`, `gradient(...) -> real3`, `divergence(...) -> real`, `laplacian(...) -> real`. Add `static constexpr bool thread_safe = true;` (or omit; it defaults to true) so consumers pick the parallel path. The cleanest pattern is to copy `src/mms/gauss1d.cpp`: derive from `struct gauss` and pull in its data + span ctor with `using gauss::gauss;`.
2. Provide a factory returning a `manufactured_solution` by relying on the converting ctor: `return {my_sol{...}};`.
3. Wire it into `manufactured_solution::from_lua` in `mms.cpp` by adding an `else if (ms_t == "myname")` branch that parses the `sol::table` and calls your factory. For a Gaussian-family variant, register through `build_ms_gauss` instead and add a `case` to the dimension `switch`.
4. Add hardcoded reference numerics to `src/mms/mms.t.cpp` following the existing `gauss1d/2d/3d` cases.

### Consume MMS in a new system

Mirror `heat`:
- Include `mms/manufactured_solutions.hpp`; add a `manufactured_solution m_sol;` member.
- In your `from_lua`, build via `manufactured_solution::from_lua(tbl, mesh.dims(), logger)` and keep the `ms_opt ? MOVE(*ms_opt) : manufactured_solution{}` fallback.
- Derive the forcing as `ddt − L(u*)` where `L` is your spatial operator, and evaluate over the mesh with `detail::eval_at_locations(m, lambda, span, m_sol.is_thread_safe())` (see `src/systems/detail/scalar_system_utils.hpp`). **Always thread `is_thread_safe()` through** so a Lua MMS does not race its single Lua state.
- Guard optional MMS use with `if (m_sol)`.

## Gotchas & invariants

- **`divergence()` has zero real consumers.** It is required by the concept, plumbed through the type-erasure layer, and gated in the Lua factory, yet every shipped backend hard-returns `0.0` ("this is a scalar field"). It is forward-looking scaffolding for a not-yet-implemented vector/Euler MMS. You cannot delete the bodies without also dropping the concept requirement and the Lua `div` gate, or the subsystem stops compiling. The Lua factory still **requires** a `div` function — an enforced-but-unused contract.
- **The single-arg, view-returning overloads are dead.** `ddt(real)`, `gradient(real)`, `gradient(int, real)`, `divergence(real)`, `laplacian(real)` (`manufactured_solutions.hpp:228-261`) return `std::views::transform` adaptors but have zero callers anywhere. They are range-v3-era leftovers stranded by the Kokkos migration. (`operator()(real time)` at line 221 is a sibling but is NOT dead — it is exercised by `mms.t.cpp:50`; do not lump it in.) See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`lua_mms` is not thread-safe** (`thread_safe = false`). A consumer that forgets to pass `is_thread_safe()` to its parallel-eval helper would race the single Lua state.
- **`lua_mms` copy/move re-bind the callbacks from the table by key**, not by copying the `std::function` objects. The Lua functions `call`/`ddt`/`grad`/`div`/`lap` must remain resident in that `sol::table`, and the table must outlive the bound functions.
- **`build_ms_gauss(dims, ...)` silently returns an empty (null) `manufactured_solution` for `dims ∉ {1,2,3}`** (`mms.cpp:30`), with no error logged. In practice `dims` comes from `mesh.dims()`, so this is a defensive default rather than a reachable path.
- **`from_lua` returns `std::nullopt` (not an empty value type) on failure** (unknown type, missing required Lua functions, or zero Gaussian entries). The "no MMS" *value* is a default-constructed `manufactured_solution`; callers convert nullopt → default.
- **A default-constructed `manufactured_solution` is null** (`operator bool == false`). `heat::rhs`/`initialize` guard with `if (m_sol)` — but `heat`'s constructor `assert(!!(this->m_sol))` (`heat.cpp:43`), so in practice `heat` does not run without an MMS. Do not assume the no-MMS branch in `rhs` is exercised in a real `heat` run.
- **MMS is heat-only.** `scalar_wave` does its own verification with a hardcoded `solution_at()` and does not touch this subsystem; `empty`/`inviscid_vortex` use no exact solution at all. Do not assume "all systems use MMS."
- **Lua arrays are 1-indexed.** `center`/`variance` are read as `t["center"][1..3]`; missing components default to `0.0`.

## Maturity & known gaps

**Verdict: mature.** Evidence: built as the `shoccs-mms` library, linked by the top-level executable and by `shoccs-systems`; covered by a dedicated, passing unit test (`t-mms`, label `mms`) with hardcoded reference numerics for value/ddt/gradient/laplacian in 1/2/3D and the Lua backend; and used in production by `heat` for source term, initial condition, Dirichlet/Neumann boundaries, and error statistics. The thread-safety mechanism (`is_thread_safe`) is actively consumed by `heat` to choose parallel vs serial evaluation.

Items within this subsystem flagged by the audit:

- **`divergence()` (concept method, gauss `0.0` bodies, Lua `div` requirement)** — *experimental*. No real consumers; deliberate scaffolding for a future vector/Euler MMS whose would-be consumer (`src/operators/divergence.hpp`) was already deleted as dead code in Phase 18. Structurally load-bearing (the concept + type erasure require it), so not freely deletable. Documented-as-experimental; deleting it is a coordinated multi-site change and a product decision. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **Single-arg range-adaptor overloads** `ddt(real)`, `gradient(real)`, `gradient(int, real)`, `divergence(real)`, `laplacian(real)` — *dead* (zero callers, range-v3 residue). Safe to delete `manufactured_solutions.hpp:228-261`, but **not** `operator()(real time)` (lines 221-226), which is live. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`gauss` default ctor + empty/`nullopt` fallback paths** — *mature*, not partial (audit refuted the "partial" flag). The default ctor is load-bearing via inherited constructors in the three derived backends, and the empty/`nullopt` outcomes are deliberately handled by `heat`. Only gap: no dedicated negative-path regression test for the `dims`-out-of-range or empty-center branches.

## Tests

- **`t-mms`** (`src/mms/mms.t.cpp`, label `mms`) — the only direct test. Four `TEST_CASE`s: `gauss1d`, `gauss2d`, `gauss3d`, `lua`. Each builds via `from_lua` and asserts `operator()`, `ddt`, `gradient`, `laplacian` against hardcoded constants. Run with `ctest --test-dir build -R t-mms` or `-L mms`. This test loads no `libkokkoscore` at runtime, so it is unaffected by the stale-Kokkos link breakage currently hitting other test binaries.
- **Not covered:**
  - `divergence()` output is **never asserted** — the Lua case defines `div` only to satisfy the `from_lua` gate; the Gaussian backends return 0.0 but it is not checked.
  - The single-arg range-adaptor overloads (`ddt(time)`, `gradient(real)`, `gradient(int, time)`, `divergence(time)`, `laplacian(time)`) have zero coverage; only `operator()(real time)` is touched (one line, `mms.t.cpp:50`).
  - The `dims ∉ {1,2,3}` and empty-center fallbacks (returning empty/nullopt) are untested.
  - Lua error paths (missing `call`/`ddt`/`grad`/`div`/`lap`) beyond the happy path are untested.
  - Copy/move semantics of `manufactured_solution` and `lua_mms` (the function re-binding) are not directly unit-tested.
- **Integration coverage** comes indirectly through `heat.t.cpp`, `simulation/simulation_cycle.t.cpp`, and `temporal/{euler_v2,rk4_v2}.t.cpp` (which configure `manufactured_solution` Lua tables), but the audit reports those binaries currently fail to load due to a stale `libkokkoscore.so.5.0` link (toolchain upgraded to Kokkos 5.1.1) — integration behavior is unverifiable without a clean rebuild. No disabled/skipped tests in `t-mms`.

## Related docs

- [Systems reference](systems.md) — `heat` is the sole MMS consumer and shows the full verification flow; note `scalar_wave` uses its own hardcoded `solution_at()` and `empty`/`inviscid_vortex` are experimental stubs that do no MMS verification.
- [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md) · [Cleanup Plan](../CLEANUP_PLAN.md).
- The MMS forcing pattern (`m.xyz | m_sol.ddt(time)` etc.) appears in the pre-Kokkos design archive `SHOCCS_ARCHITECTURE_AND_KOKKOS_MIGRATION_SPEC.md` (sections ~542); that is a historical rationale only — the live code uses `eval_at_locations` + lambdas, and the piped forms are the now-dead single-arg overloads.
