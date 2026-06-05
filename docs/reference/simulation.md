# Simulation (`src/simulation/`)

> **Maturity:** partial · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose
The simulation layer is the data-flow spine that turns a parsed Lua config into a runnable simulation and drives the time-stepping loop. `simulation_cycle` owns the four top-level components (a `system`, a `step_controller`, an `integrator`, and a `field_io`) plus a `logs` member, and its `run()` method allocates registry slots, initializes the system, builds the RHS Kokkos graph once, and loops calling the integrator until either the `step_controller` reaches its limits or the system stops being valid. It is the layer reached from the production executable; everything below it (mesh, operators, stencils) is built one level deeper inside each concrete system's own `from_lua`.

## Where it lives
| File | Role |
| --- | --- |
| `src/simulation/simulation_cycle.cpp` | The live spine. `simulation_cycle::from_lua` assembles `system`/`integrator`/`step_controller`/`field_io`; `run()` does registry slot allocation, builds the RHS graph once, runs the time-stepping loop with `deep_copy_slot`, and returns a `real3`. |
| `src/simulation/simulation_cycle.hpp` | `simulation_cycle` class declaration: members, 5-arg move ctor, default ctor, static `from_lua`, `run()`. |
| `src/simulation/CMakeLists.txt` | Builds `shoccs-simulation` (currently from BOTH `simulation_builder.cpp` and `simulation_cycle.cpp` — the dead builder is still compiled in); registers `t-simulation_cycle` under label `simulation`. |
| `src/simulation/simulation_cycle.t.cpp` | End-to-end tests (heat+rk4, heat+euler) driving `from_lua` + `run()` with a full Lua config (mesh, cut-cell sphere, lua MMS). |
| `src/simulation/simulation_builder.{hpp,cpp}` | **DEAD stub.** `build()` ignores its Lua argument and returns a default-constructed cycle. Not on the data path; zero callers. See [Maturity & known gaps](#maturity--known-gaps). |
| `src/lib/run_from_sol.cpp` | Production wrapper `ccs::simulation_run` that calls `simulation_cycle::from_lua` then `run()`; the real bridge from the executable to this subsystem. |
| `src/app/shoccs.cpp` | The `shoccs` executable `main`: parses CLI + Lua, calls `ccs::simulation_run(lua["simulation"])`. |

## Public API / entry points

### The real entry point
```cpp
// src/simulation/simulation_cycle.hpp
class simulation_cycle {
public:
    simulation_cycle() = default;
    simulation_cycle(system&&, step_controller&&, integrator&&, field_io&&,
                     bool enable_logging = false);   // 5-arg move ctor

    static std::optional<simulation_cycle> from_lua(const sol::table&);
    real3 run();
};
```
- `simulation_cycle::from_lua(const sol::table& tbl)` — assembles the four components by delegating to each subsystem's own `from_lua` factory; returns `std::nullopt` if any of them fails. This is the genuine top-level assembly point.
- `simulation_cycle::run()` — runs the time loop and returns a `real3`:
  - `{(real)controller, e, e}` on clean completion, where `e` is the L∞ error from `sys.summary(stats)` (so `res[0]` = final time, `res[1]` = `res[2]` = L∞ error).
  - `{(real)controller, null_v<real>, null_v<real>}` if the loop exited while the controller still reports valid (premature end).
  - `{null_v<real>}` (i.e. `{huge, 0, 0}`) if a requested timestep was below `min_dt`.

### Production wrapper
```cpp
// src/lib/run_from_sol.cpp
std::optional<real3> ccs::simulation_run(const sol::table& lua);
// internally: simulation_cycle::from_lua(lua) -> run()
```
`src/app/shoccs.cpp` `main` calls `ccs::simulation_run(lua["simulation"])`.

### Component factories the cycle depends on (each builds itself from the SAME `simulation` table)
- `system::from_lua(const sol::table&, const logs& = {})` → `std::optional<system>` (`src/systems/system.cpp`)
- `integrator::from_lua(const sol::table&, const logs& = {})` → `std::optional<integrator>` (`src/temporal/integrator.cpp`)
- `step_controller::from_lua(const sol::table&, const logs& = {})` → `std::optional<step_controller>` (`src/temporal/step_controller.hpp`)
- `field_io::from_lua(const sol::table&, const logs& = {})` → `std::optional<field_io>` (`src/io/field_io.hpp`)

### Dead API (do not use)
```cpp
// src/simulation/simulation_builder.hpp  — DEAD, zero callers
class simulation_builder {
public:
    std::optional<simulation_cycle> build(const sol::table& lua) &&;  // ignores lua, returns {simulation_cycle{}}
};
```

## How it works

The real, end-to-end data flow is:

```
app/shoccs.cpp main
  -> ccs::simulation_run(lua["simulation"])        (src/lib/run_from_sol.cpp)
     -> simulation_cycle::from_lua(tbl)            (assemble 4 components)
        -> system::from_lua(tbl, logs)             (heat/scalar wave/eigenvalues/inviscid vortex)
             -> mesh::from_lua / stencil::from_lua / bcs::from_lua / mms::from_lua   (built HERE, inside the system)
        -> integrator::from_lua(tbl, logs)
        -> step_controller::from_lua(tbl, logs)
        -> field_io::from_lua(tbl, logs)
     -> simulation_cycle::run()                    (time loop)
```

Important: the simulation layer assembles only `system + integrator + step_controller + field_io`. Mesh and operators are NOT built here — they are constructed one level deeper inside each concrete system's `from_lua` (e.g. `heat::from_lua` at `src/systems/heat.cpp:78` calls `mesh::from_lua`, `bcs::from_lua`, `stencil::from_lua`, `manufactured_solution::from_lua`). This corrects the CLAUDE.md "Lua → builder → mesh+operators+system+integrator" diagram.

### `run()`'s registry / slot model (`src/simulation/simulation_cycle.cpp`)
- A single `sim_registry reg;` is created on the stack. `sim_registry` is `field_registry<8, 8, 4>` (8 slots, up to 8 scalars / 4 vectors per slot) defined in `src/fields/field_registry.hpp`.
- `sys.size()` returns a `system_size` carrying `nscalars`, `nvectors`, and the four buffer sizes (`d_size`, `rx_size`, `ry_size`, `rz_size`).
- Four logical slots are allocated, one set per scalar and per vector field:
  - slot 0 → `u0_ref` (current solution / RHS-graph base)
  - slot 1 → `u1_ref` (next solution; RHS graph **input** slot)
  - slot 2 → `rk_ref` (integrator scratch)
  - slot 3 → `srhs_ref` (RHS **output** slot)
- Per-slot allocation goes through `reg.allocate_scalar(slot, index, d,rx,ry,rz)` / `allocate_vector(...)`, which returns an updated `field_ref` for that slot.
- The pre-loop sequence is: `sys.initialize(reg, u0_ref, controller)` → `reg.deep_copy_slot(u1, u0)` → `sys.update_boundary(reg, u0_ref, controller)` → `sys.stats(...)` → `sys.log(...)` → initial `sys.write(io, reg, u0_ref, controller, 0.0)`.
- `sys.build_rhs_graph(reg, u1_ref, reg, srhs_ref)` builds the Kokkos graph **once**, capturing the View data pointers of slots 1 (input) and 3 (output). Only graph-capable systems (heat, scalar_wave) build a real graph; the `system` dispatch guards with `if constexpr (requires { ... })` and is a no-op otherwise (`src/systems/system.cpp:41`).

### The time loop
```
while (controller && sys.valid(stats)) {
    dt = sys.timestep_size(reg, u0_ref, controller);   // nullopt -> return {null_v<real>}
    integrate(sys, reg, u0_ref, u1_ref, rk_ref, srhs_ref, controller, *dt);
    controller.advance(*dt);                            // time += dt; step += 1
    stats = sys.stats(reg, u0_ref, u1_ref, controller);
    sys.write(io, reg, u1_ref, controller, *dt);
    sys.log(stats, controller);
    reg.deep_copy_slot(u0_ref.slot, u1_ref.slot);       // NOT swap_slots — keeps graph pointers stable
}
```
- `controller`'s `operator bool()` is the loop's termination test (max step / max time), and its `operator real()` / `operator int()` supply the current time/step at the call sites.
- The integrator (`rk4` or `euler`) repeatedly calls back into `sys.submit_rhs_graph(...)` / `sys.rhs(...)` through the slots it was handed.

## How to extend

The simulation layer itself rarely needs changing — it delegates to component `from_lua` factories. The two common extensions both happen one layer down:

### Add a new PDE system that `simulation_cycle` can run
1. Implement the registry-based system interface in `src/systems/` (copy `src/systems/heat.{hpp,cpp}` as the template). Required members (match the signatures the variant dispatches to in `src/systems/system.cpp`):
   `size()`, `initialize(reg, ref, controller)`, `update_boundary(reg, ref, time)`, `rhs(creg, in, reg, out, time)`, `stats(reg, u0, u1, controller)`, `timestep_size(reg, u, controller)`, `write(io, reg, ref, c, dt)`, `valid(stats)`, `summary(stats)`, `log(stats, controller)`. Add `build_rhs_graph(scalar_view, scalar_span)` + `submit_rhs_graph()` (+ optional `fill_source(time)`) if the system is graph-capable.
2. Add the concrete type to the `std::variant` in `src/systems/system.hpp` (the `v` member).
3. Add a dispatch branch in `system::from_lua` (`src/systems/system.cpp`), matching on `system.type` and calling your `YourSystem::from_lua`. Build the mesh/operators inside your `from_lua` (call `mesh::from_lua`, `stencil::from_lua`, `bcs::from_lua`), exactly as `heat::from_lua` does.
4. No change to `simulation_cycle` is needed — `from_lua` and `run()` are system-agnostic.

### Add a new time integrator
1. Implement it in `src/temporal/` (copy `euler` / `rk4`).
2. Add it to the `std::variant` in `src/temporal/integrator.hpp` and a dispatch arm in `integrator::operator()` and `integrator::from_lua` (`src/temporal/integrator.cpp`).
3. Again, no change to `simulation_cycle`.

Do NOT extend `simulation_builder` — it is dead. New top-level orchestration logic belongs in `simulation_cycle::from_lua` / `run()`.

### Lua `simulation` table schema (from the working test config)
```lua
simulation = {
    mesh = { index_extents = {21, 22}, domain_bounds = { min = {...}, max = {...} } },
    domain_boundaries = { xmin = "dirichlet", ymin = "neumann", ymax = "neumann" },
    shapes = { { type = "sphere", center = {...}, radius = 0.25, boundary_condition = "floating" } },
    scheme = { order = 2, type = "E2" },
    system = { type = "heat", diffusivity = 1.0 },     -- type keys are SPACE-separated (see gotchas)
    integrator = { type = "rk4" },                      -- or "euler"
    step_controller = { max_step = 5 },
    manufactured_solution = { type = "lua", call=..., ddt=..., grad=..., lap=..., div=... },
    -- optional: logging = true|false, logging_dir = "logs"
}
```
`mesh`, `domain_boundaries`, `shapes`, `scheme`, `manufactured_solution` are consumed inside the system's `from_lua` (not by the simulation layer).

## Gotchas & invariants
- **`system.type` dispatch keys are space-separated, not underscore:** `"scalar wave"`, `"inviscid vortex"` (`src/systems/system.cpp`). The tests use `"heat"`. Writing `"scalar_wave"` will silently fail to match → `from_lua` returns `nullopt` → `simulation_run` returns `nullopt`.
- **`"inviscid vortex"` is accepted but is a complete stub.** It is wired into the variant and dispatch but `valid()` hard-returns `false`, so `run()` exits the `while` loop before the first integration step and reports nothing useful. See [Maturity & known gaps](#maturity--known-gaps).
- **`simulation_builder` is dead code** (last touched 2021 "namespace reorg", never instantiated). Do not assume it is the entry point despite CLAUDE.md; the real entry is `simulation_cycle::from_lua`.
- **`from_lua` passes a `logs` object into the `bool enable_logging` constructor parameter.** It compiles only because `logs::operator bool()` exists (`src/io/logging.hpp:29`), and it silently discards the `logging_dir`. The constructor then rebuilds its own `logs{enable_logging, "cycle"}` with no directory.
- **`run()` uses `deep_copy_slot`, never `swap_slots`, to copy `u1`→`u0`.** This is deliberate: the RHS Kokkos graph was built once and captured stable View data pointers; a swap would change which buffer those pointers refer to and silently corrupt graph execution. (`field_registry` exposes both; only `deep_copy_slot` is safe here.)
- **Zero-field systems (`nscalars==0 && nvectors==0`)**: the `field_ref`s keep their initial `{slot, 0, 0}` state and `slot_ops` no-op. The assert at `simulation_cycle.cpp:59` (`u0_ref.n_scalars == sz.nscalars && u0_ref.n_vectors == sz.nvectors`) encodes this invariant.
- **`step_controller` is passed where systems declare a `real time` parameter** (e.g. `update_boundary(reg, ref, real time)` in the headers). This works only because `step_controller::operator real()` returns its current time.
- **Component ordering in the 5-arg ctor differs from the assembly order**: `from_lua` constructs `system`, `integrator`, `step_controller`, `field_io`, but calls the ctor as `simulation_cycle{sys, step_controller, integrator, field_io, logs}`. Match the ctor's parameter order, not the construction order.

## Maturity & known gaps
**Verdict: partial.** `simulation_cycle` itself is mature and is the live spine — it is reached from production (`app/shoccs.cpp` → `ccs::simulation_run` → `simulation_cycle::from_lua` → `run()`), `simulation_cycle.cpp` was last edited 2026-03-27 (Phase 19), and the registry/graph mechanics in `run()` are complete and current. Two end-to-end tests exercise the full `from_lua` + `run()` path. The subsystem is rated **partial**, not mature, because of dead/incomplete neighbors it is wired to.

Verified flagged items in this subsystem:
- **`simulation_builder` class (`src/simulation/simulation_builder.cpp` / `.hpp`) — DEAD (zero callers, safe to delete).** `build()` ignores its Lua argument and returns a default-constructed `simulation_cycle{}`. No instantiation or `.build()` call anywhere in `src/`, scripts, tests, or Lua; the only reference is a leftover `#include "simulation_builder.hpp"` in `simulation_cycle.t.cpp:10` that never names the class. Single commit each, last touched 2021-05-04, while the sibling `simulation_cycle.cpp` moved to 2026. Fully superseded by `simulation_cycle::from_lua`. **Recommendation: delete** both files, drop `simulation_builder.cpp` from `src/simulation/CMakeLists.txt:1`, and replace the stale include in the test with `#include "simulation_cycle.hpp"`. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`inviscid_vortex` system (`src/systems/inviscid_vortex.cpp`) reachable via `system::from_lua` (`type="inviscid vortex"`) — EXPERIMENTAL (research-only stub).** Reachable through the variant + dispatch, but `valid()` returns `false`, `size()` returns `{}`, and all hooks are empty, so a run does zero integration steps. It carries a complete analytic Brady & Livescu vortex `solution` namespace but no `rhs`/`initialize`/BC implementation. **Recommendation: document-as-experimental** (do NOT delete — the analytic solution is the intended Euler-verification entry point to be finished later; a defensible alternative is to make `from_lua` warn/return `nullopt` for this type until `rhs()` exists). Lives outside `src/simulation/` but directly limits how complete the builder is across advertised systems. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`t-simulation_cycle` builds and passes (build fixed 2026-06-04).** The test and the code it covers are mature and load-bearing. This was previously blocked by the Kokkos 5.0→5.1.1 Graph API change, now resolved: all 17 `create_graph` call sites were migrated to the templated 1-arg form `create_graph<execution_space>(closure)` (in `derivative.cpp`, `laplacian.cpp`, `heat.cpp`, `scalar_wave.cpp`, `graph_poc.t.cpp`); node-building methods and numerics are unchanged. `cmake --build build` is green and `ctest --test-dir build` is 47/48 (`t-csr` and `t-E2_1` have since been fixed; the 1 remaining failure — `t-laplacian` — is pre-existing and unrelated to simulation). See [Cleanup Plan](../CLEANUP_PLAN.md).

## Tests
- **`t-simulation_cycle`** (`src/simulation/simulation_cycle.t.cpp`, ctest label `simulation`). Custom `main()` with `Kokkos::ScopeGuard`; links `Catch2::Catch2` (not `WithMain`). Two cases:
  - `"cycle - 2D"` — heat + rk4, 21×22 grid, sphere cut-cell, lua MMS; asserts `res[0] == 0.0125` (final time) and `res[1] < 0.05` (L∞ error).
  - `"cycle - 2D euler"` — heat + euler, same grid/MMS; asserts `res[1] < 0.05`.
  Both drive the complete `simulation_cycle::from_lua` + `run()` chain.
- **Current run status:** PASSES (build fixed 2026-06-04). This was previously blocked by the project-wide Kokkos 5.0→5.1.1 Graph API break described above; the `create_graph` migration to the templated 1-arg form resolved it.
- **Not covered:** `simulation_builder` is never exercised; `scalar_wave` and `hyperbolic_eigenvalues` are never run through `simulation_cycle` (only their own unit tests exist); `inviscid_vortex` is never tested; the `from_lua` failure/`nullopt` paths (missing/invalid `system`/`integrator`/`step_controller`/`field_io` tables) have no negative tests; the "ended prematurely" and "timestep too small" branches of `run()` are uncovered.
- **Disabled/removed:** a ~75-line commented-out 3D test case was removed in Phase 18.

## Related docs
- [Capability Audit](../CAPABILITY_AUDIT.md) and [Onboarding](../ONBOARDING.md).
- [Cleanup Plan](../CLEANUP_PLAN.md) — `simulation_builder` deletion, `inviscid_vortex` documentation, and the Kokkos 5.1.1 `create_graph` build-break.
- Reference docs for the subsystems the cycle wires together: **systems** (`system` variant + `system::from_lua`, heat/scalar_wave/eigenvalues/inviscid_vortex), **temporal** (`integrator`, `step_controller`, rk4/euler), **fields** (`field_registry`/`sim_registry`, `field_ref`, slot ops), **io** (`field_io`, logging), and **mesh**/**operators**/**stencils** (built transitively inside each system's `from_lua`).
- The live assembler is the static factory `simulation_cycle::from_lua` (not a builder object), and the simulation layer does NOT build mesh+operators — each system's own `from_lua` does that transitively. (Earlier `CLAUDE.md` revisions described a `simulation::builder` data flow; that has since been corrected.)
- The pre-Kokkos `SHOCCS_ARCHITECTURE_AND_KOKKOS_MIGRATION_SPEC.md` (§5.9, ~line 614) documents a `simulation_builder::from_lua` flow that never existed in code — a stale design rationale, useful only as historical context.
