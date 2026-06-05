# App & Build (`src/app/`, `src/lib/`, `src/CMakeLists.txt`, `CMakeLists.txt`)

> **Maturity:** mature · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose
This subsystem turns the SHOCCS library graph into a runnable program and an installable, linkable library. `src/app/shoccs.cpp` is the one-and-only `main()`: it initializes Kokkos, parses a CLI, loads a Lua config file, and hands the config's `simulation` table to the library. `src/lib/` is a thin public shim — a single function `ccs::simulation_run(const sol::table&)` that bridges from a Lua table into the well-tested `simulation_cycle::from_lua().run()` path. The top-level and `src/CMakeLists.txt` define the whole build: third-party discovery via a spack view, the dependency-ordered list of `shoccs-<subsystem>` libraries, the `add_unit_test` helper, and the install/export rules.

## Where it lives
| File | Role |
| --- | --- |
| `src/app/shoccs.cpp` | The `shoccs` executable `main()`. `Kokkos::ScopeGuard`, cxxopts CLI parse (`input-file`/`script`/`check`/`help`), a `sol::state` Lua load (base + math libs only), then `ccs::simulation_run(lua["simulation"])`. 56 lines. |
| `src/app/CMakeLists.txt` | Defines `add_executable(shoccs-exe shoccs.cpp)`; `OUTPUT_NAME "shoccs"`; links `cxxopts`, `shoccs-run_sol`, `spdlog`, `Kokkos`; `install(TARGETS shoccs-exe)`. |
| `src/lib/shoccs.hpp` | Public header. Declares `ccs::simulation_run(const sol::table&) -> std::optional<real3>`. Installed as a `PUBLIC_HEADER`. |
| `src/lib/run_from_sol.cpp` | Implements `simulation_run`: `simulation_cycle::from_lua(lua)`, return `run()` result or `std::nullopt`. ~20 lines, interface-stable since 2022. |
| `src/lib/CMakeLists.txt` | Defines `shoccs-run_sol` static lib. `OUTPUT_NAME`/`EXPORT_NAME "shoccs"` → `libshoccs.a`. Links `lua`/`sol2` PUBLIC, `shoccs-simulation` PRIVATE. Installs + exports into the `shoccs` EXPORT set. |
| `CMakeLists.txt` (top level) | C++20, TPL `find_package`s, `add_unit_test` helper (lines 51-58), RPATH (`$ORIGIN`), `add_subdirectory(src)`, the install `TARGETS` list of every `shoccs-*` lib, and the `shoccs::` namespaced export. |
| `src/CMakeLists.txt` | `SOL_ALL_SAFETIES_ON=1` define; the dependency-ordered `add_subdirectory` list (`fields` … `simulation` … `lib` … `app`); the `indexing` INTERFACE lib and first three unit tests. |
| `heat.lua` (repo root) | Canonical example of the `simulation` table schema consumed via `lua["simulation"]`. See also `scalar_wave.lua`, `eigenvalues.lua`, `scalar_wave_1d.lua`, and `lua-configs/*.lua`. |

## Public API / entry points

### Executable
```cpp
int main(int argc, char* argv[]);   // src/app/shoccs.cpp; binary OUTPUT_NAME "shoccs"
```
Invoke it as:
```bash
./build/src/app/shoccs path/to/config.lua          # positional input-file (happy path)
./build/src/app/shoccs config.lua --script "dx=0.1" # inline lua, takes precedence over file
./build/src/app/shoccs config.lua --check           # parse/validate lua, exit 0 before running
./build/src/app/shoccs --help                        # print usage and exit 0
```
CLI surface (cxxopts, defined in `options.add_options()`):
- `input-file` — main Lua file. Bound positionally via `options.parse_positional("input-file")`, so it is supplied without a flag.
- `script` — supplementary inline Lua string, run *after* the file, so it overrides file values.
- `check` — `bool`, default `false`. Parse the Lua, then return `0` without running the simulation.
- `help` — print usage.

### Library
```cpp
// src/lib/shoccs.hpp
namespace ccs {
std::optional<real3> simulation_run(const sol::table& lua);
}
```
`lua` is the `simulation` sub-table (not the whole Lua state). Returns the cycle's final `real3` result (the MMS / final-solution metric from `simulation_cycle::run()`), or `std::nullopt` if `simulation_cycle::from_lua` rejected the config. This is the installed public API, exported as the `shoccs::shoccs` target via `libshoccs.a` and `shoccs.hpp`/`shoccs_config.hpp` headers.

### CMake API (used when adding code)
```cmake
add_unit_test(<name> "<label>" [link-libs...])   # CMakeLists.txt:51
```
Compiles `<name>.t.cpp` into `t-<name>`, links `Catch2::Catch2WithMain` plus `link-libs`, registers it with ctest under `LABELS "<label>"`. Guarded by `if (BUILD_TESTING)`. (Tests that need a custom Kokkos `main()` bypass this helper and link `Catch2::Catch2` directly — see the `temporal` CMakeLists for the `t-rk4_v2`/`t-euler_v2` pattern.)

CMake options:
- `BUILD_TESTING` (from `include(CTest)`, default ON) — gates `find_package(Catch2 3)` and all `add_unit_test`/test targets.
- `BUILD_BENCHMARKS` (default `OFF`) — gates `find_package(benchmark)` and `add_subdirectory(benchmarks)`.
- `SHOCCS_TPL_DIR` — optional third-party prefix prepended to `CMAKE_PREFIX_PATH`.

## How it works

### Runtime data flow (one `shoccs config.lua` invocation)
1. `Kokkos::ScopeGuard kokkos(argc, argv)` initializes Kokkos for the whole process lifetime (RAII; finalized at `main` exit). **`main` owns Kokkos init/finalize — the library does not.**
2. cxxopts parses argv; `input-file` is consumed positionally.
3. Early exit: if `--help` is set OR `result.arguments().size() == 0`, print usage and `return 0`.
4. `sol::state lua; lua.open_libraries(sol::lib::base, sol::lib::math);` — only base + math are opened.
5. If `input-file` present → `lua.script_file(...)`. If `--script` present → `lua.script(...)` (runs after, so it overrides).
6. If `--check` → `return 0` here (before any simulation work).
7. `spdlog::info("Starting shoccs")`, then `ccs::simulation_run(lua["simulation"])`.
8. `simulation_run` → `simulation_cycle::from_lua(lua)`; on success `cycle->run()` returns `real3`, otherwise `std::nullopt`. `main` discards this return value.

Everything below `simulation_run` (mesh, operators, stencils, systems, integrators, I/O) is assembled inside `simulation_cycle::from_lua` and the concrete system's own `from_lua`. See [Simulation](simulation.md).

### Build / link structure
- The top-level `CMakeLists.txt` finds every TPL, defines the `add_unit_test` helper, then `add_subdirectory(src)`.
- `src/CMakeLists.txt` lists subdirectories in dependency order; `lib` and `app` come last so they can link the assembled `shoccs-simulation`.
- Each subsystem dir builds a `shoccs-<name>` library; `app` links `shoccs-run_sol` (which transitively pulls `shoccs-simulation` and the rest).

Target naming, as actually defined:
- INTERFACE libs (header-only): `fields`, `indexing`, `shoccs-utils`.
- Static libs: `shoccs-mesh`, `shoccs-matrices`, `shoccs-stencils`, `shoccs-operators`, `shoccs-bcs`, `shoccs-io`, `shoccs-logging`, `shoccs-mms`, `shoccs-system`, `shoccs-integrate` (the temporal subsystem — note the directory is `src/temporal/` but the target is `shoccs-integrate`), `shoccs-simulation`, `shoccs-random`, `shoccs-run_sol`.
- Executable: `shoccs-exe` (binary renamed to `shoccs`).

### Third-party discovery (spack view)
There is no vendored dependency tree. TPLs are found by plain `find_package` against a spack environment view. The active build's `Kokkos_DIR` points at `/home/user/spack/var/spack/environments/shoccs-dev/.spack-env/view/lib/cmake/Kokkos`, and the view symlinks all CMake config packages (Boost, Catch2, cxxopts, fmt, Kokkos, lapackpp, pugixml, spdlog, benchmark, …). Required packages and minimum versions, from the top-level `CMakeLists.txt`:

| `find_package` | Notes |
| --- | --- |
| `Lua REQUIRED` | wrapped into an INTERFACE `lua` target (`LUA_INCLUDE_DIR` / `LUA_LIBRARIES`). |
| `sol2 REQUIRED` | C++ Lua binding; `SOL_ALL_SAFETIES_ON=1` is set globally in `src/CMakeLists.txt`. |
| `fmt 8 REQUIRED` | formatting. |
| `pugixml 1.10 REQUIRED` | XDMF/XML I/O. |
| `spdlog 1.9 REQUIRED` | logging. |
| `cxxopts REQUIRED` | CLI parsing. |
| `Catch2 3 REQUIRED` | only if `BUILD_TESTING`. |
| `Boost REQUIRED` | header-only `mp11` only. |
| `lapackpp REQUIRED` | dense linear algebra. |
| `Kokkos REQUIRED` | parallel execution (host-only today). |
| `benchmark REQUIRED` | only if `BUILD_BENCHMARKS`. |

## How to extend

**Add a CLI option.** Edit `options.add_options()` in `src/app/shoccs.cpp` and branch on `result.count("name")` in `main`. Decide where in the sequence it fires relative to the `--check` early-return (step 6 above).

**Add a public library entry point.** Declare it in `src/lib/shoccs.hpp`, define it in `src/lib/run_from_sol.cpp` (or a new `.cpp` added to `add_library(shoccs-run_sol ...)`), and make sure the providing subsystem is linked — `shoccs-run_sol` currently links `shoccs-simulation` PRIVATE.

**Add a new subsystem library.** Create `src/<name>/CMakeLists.txt` with `add_library(shoccs-<name> ...)`; add the directory to `src/CMakeLists.txt` **in dependency order** (before anything that links it, after everything it links); and add the target to the top-level `install(TARGETS ...)` list (`CMakeLists.txt` lines ~71-87) or it will be missing from the exported package. Copy an existing subsystem's CMakeLists as a template (e.g. `src/temporal/CMakeLists.txt`).

**Add a test.** `add_unit_test(<name> "<label>" <link-libs...>)` for a simple Catch2 test (`<name>.t.cpp` → `t-<name>`). Use the manual `add_executable + Catch2::Catch2 + add_test + set_tests_properties LABELS` pattern (see `src/temporal/CMakeLists.txt`) if the test needs a custom Kokkos `main()`.

**Add a third-party dependency.** Add a `find_package(...)` near the top of `CMakeLists.txt` (lines ~16-39). Guard with `if (BUILD_TESTING)` / `if (BUILD_BENCHMARKS)` if it is only needed for those. Then `spack add` it to the `shoccs-dev` environment and refresh the view so the config package is discoverable.

**Add a config field.** The Lua `simulation` schema is defined by the downstream consumers (`simulation_cycle::from_lua` and each system's `from_lua`), not here. `main` just passes `lua["simulation"]` straight through. Use `heat.lua` as the canonical shape (keys: `mesh`, `domain_boundaries`, `shapes`, `scheme`, `system`, `integrator`, `step_controller`, `manufactured_solution`, `io`).

## Gotchas & invariants
- **Kokkos lifetime lives in `main`, not the library.** `simulation_run` does **not** call `Kokkos::initialize`. Anyone embedding `libshoccs.a` must create a `Kokkos::ScopeGuard` (or call init/finalize) themselves before invoking `simulation_run`.
- **`--check` is shallow.** It validates that the Lua *executes* without error, but returns `0` *before* `simulation_run` is ever called — so a malformed `simulation` table (one `from_lua` would reject) still passes `--check`.
- **`main` discards the result.** `simulation_run`'s `std::optional<real3>` (and therefore a `nullopt` from a bad config) is thrown away. The process exit status does not reflect whether the run succeeded; a bad config silently does nothing after logging `"Starting shoccs"`. (Minor known follow-up: log + return nonzero on `nullopt`.)
- **Help/empty-arg trigger uses `arguments().size()`,** which counts parsed *option occurrences*. An invocation with only a positional input-file still proceeds (the file is consumed into `input-file`, not counted as an argument).
- **Only `sol::lib::base` and `sol::lib::math` are opened.** Lua configs cannot rely on `io`, `os`, `string`, or `table` stdlib modules being present.
- **`OUTPUT_NAME "shoccs"` is used by two targets** — `shoccs-exe` (→ `build/src/app/shoccs`) and `shoccs-run_sol` (→ `build/src/lib/libshoccs.a`). They don't collide on disk (exe vs `.a`, different dirs) but the shared logical name is confusing.
- **`shoccs-run_sol` links `shoccs-simulation` PRIVATE.** Consumers of `libshoccs.a` get `simulation_run` and the `lua`/`sol2` headers, but not the rest of the `shoccs-*` graph as usable headers.
- **Two install sites feed one export set.** `shoccs-run_sol` is installed in `src/lib/CMakeLists.txt` (`EXPORT shoccs`); the other libs are installed in the top-level `CMakeLists.txt` `install(TARGETS ...)` (also `EXPORT shoccs`). The top-level list is hand-maintained — adding a subsystem lib without listing it there breaks the exported `shoccs::` package.

## Maturity & known gaps
**Verdict: mature.** The subsystem is small but load-bearing and correctly wired. `shoccs.cpp` was touched recently (2026-03-10, Phase 0 Kokkos init), the binary is the documented run entry point (`./build/src/app/shoccs config.lua`) used by `scripts/profile.sh` and the Python `cpp_bridge`, real Lua configs exist at the repo root and in `lua-configs/`, and `simulation_run` delegates to the well-tested `simulation_cycle`. The `src/lib/` files are old (2021/2022) but stable, not stale — their `sol::table -> optional<real3>` interface survived the Kokkos migration untouched precisely because a thin, correct delegate has no reason to change.

Flagged items (with audit verdicts):
- **Build artifacts rebuilt — was the Kokkos 5.0 → 5.1.1 `.so` mismatch, fixed 2026-06-04.** Earlier the on-disk `build/src/app/shoccs` failed to launch with `error while loading shared libraries: libkokkoscore.so.5.0` because spack upgraded Kokkos 5.0 → 5.1.1 *after* the last build. A reconfigure + rebuild resolved this; `cmake --build build` now builds the whole tree including `build/src/app/shoccs`, and the binary launches. The binary is a gitignored artifact. **Verdict: resolved.**
- **C++ build blocker resolved — Kokkos 5.1.1 `create_graph` API change, fixed 2026-06-04.** Kokkos 5.1.1 dropped the 2-argument `create_graph(execution_space{}, closure)` overload (only `create_graph(Closure&&)` remains). All 17 call sites were migrated to the templated 1-argument form `create_graph<execution_space>(closure)` across `src/operators/derivative.cpp`, `src/operators/laplacian.cpp`, `src/systems/heat.cpp`, `src/systems/scalar_wave.cpp`, and `src/fields/graph_poc.t.cpp`; node-building methods and numerics are unchanged. The build is green and produces a working `shoccs` binary. See [Cleanup Plan](../CLEANUP_PLAN.md) §0a.
- **Reconfigure note.** If `build/build.ninja` predates the current spack view, regenerate it from the current view CMake: `cmake -S /workspace -B /workspace/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON`. The reconfigure and subsequent compile both succeed.
- **No automated tests for `src/app/` or `src/lib/` — partial coverage gap.** Confirmed partial: the `--check`, `--script` precedence, empty-args/`--help` trigger, and the `from_lua -> nullopt` path have zero ctest coverage; the happy path is exercised only indirectly (Python `cpp_bridge` subprocess + `simulation_cycle.t.cpp`). The audit recommends *finishing* this (a small `t-app`/`t-run_from_sol` covering the four branches), not deprecating it. See [Cleanup Plan](../CLEANUP_PLAN.md).

Nothing in this subsystem is dead (zero-caller) — `simulation_run` is the sole call from `main`, and both `lib` and `app` are fully build-wired and installed.

## Tests
- **No dedicated `*.t.cpp`** under `src/app/` or `src/lib/`; none of `main`'s CLI logic or `simulation_run`'s `nullopt` branch is unit-tested.
- **Indirect C++ coverage:** `t-simulation_cycle` (label `simulation`) drives `simulation_cycle::from_lua().run()` — the exact path `simulation_run` wraps — and the system tests `t-heat`, `t-scalar_wave`, `t-hyperbolic_eigenvalues` exercise it end to end. All only take the happy (`!!cycle_opt`) branch.
- **End-to-end:** the Python `scripts/stencil_gen/.../cpp_bridge.py` subprocess-invokes the `shoccs` binary (positional file only) and `test_cpp_bridge.py` parses `logs/system.csv`; its real-binary smoke test is `@pytest.mark.slow` and skips if the binary is unbuilt.
- **Build-graph wiring** is implicitly validated by the registered ctest entries; the project-wide audit confirms 0 orphan sources.
- **Current ctest status:** after the 2026-06-04 build fix, `ctest --test-dir build` is 47/48 pass. There is 1 remaining failure (`t-laplacian`); `t-csr` and `t-E2_1` are now fixed. The remaining failure is a pre-existing issue surfaced (not caused) by the build fix — tracked in [Cleanup Plan](../CLEANUP_PLAN.md) §0a. None are in `src/app/` or `src/lib/`.

## Related docs
- [Simulation](simulation.md) — the `simulation_cycle` this subsystem invokes (the next layer down).
- [Core Types](core-types.md) — `real3` and the config aliases used in the public signature.
- [Systems](systems.md), [Operators](operators.md), [Stencils](stencils.md), [Temporal](temporal.md), [Mesh](mesh.md), [Fields](fields.md), [I/O](io.md), [MMS](mms.md) — the `shoccs-*` libraries linked transitively through `shoccs-simulation`.
- `docs/handoff/operating_conventions.md` — documents the `./build/src/app/shoccs config.lua` run workflow and the `logs/system.csv` output.
- `docs/brady2d_cpp_bridge_reference.md` and `scripts/stencil_gen/` — the Python pipeline that subprocess-invokes this binary.
- `build_blocker.md` (audit) — full detail on the Kokkos 5.1.1 `create_graph` compile failure.
- Note: `docs/handoff/framework_architecture.md` documents the *Python* stencil pipeline, not this C++ app/lib layer, despite its title.
