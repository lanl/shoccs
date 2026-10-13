# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SHOCCS (Stable High-Order Cut-Cell Solver) is a C++ Cartesian cut-cell solver for time-dependent PDEs (heat equation, scalar wave, Euler equations). It uses high-order finite difference operators on structured grids with embedded boundaries. The codebase uses Kokkos for parallel execution.

## Documentation map

- `docs/ONBOARDING.md` — start here if you are new to the codebase.
- `docs/CAPABILITY_AUDIT.md` — per-subsystem maturity status (what actually works vs. scaffolding).
- `docs/reference/` — per-subsystem reference docs.
- `docs/CLEANUP_PLAN.md` — known issues and remediation plan (§0a: the one failing test).
- `docs/handoff/MASTER.md` — Python framework (stencil derivation / sweeps) handoff.

## Build Commands

> Known failure: `t-laplacian` (47/48 pass; `docs/CLEANUP_PLAN.md` §0a). If `build.ninja` references a missing cmake path (stale tree), rerun the reconfigure command below.

```bash
# Build (from repo root; build/ is pre-configured with Ninja)
cmake --build build

# Build a single target
cmake --build build --target t-heat

# Run all tests
ctest --test-dir build

# Run tests by label (fields, matrices, operators, stencils, mesh, systems, temporal, simulation)
ctest --test-dir build -L fields

# Run a single test by name
ctest --test-dir build -R t-dense

# Reconfigure from scratch
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON -DBUILD_BENCHMARKS=ON
```

### Native macOS build (Apple silicon)

- Optional host build in `build-host/` against a user-managed spack environment `shoccs-host` (nothing checked in); setup, manifest, the kokkos override and pitfalls: `docs/reference/macos-host-build.md`.
- `spack` runs outside the Bash sandbox (it mis-detects the CPU inside it): bare name, one command per call, no `cd`, pipe or redirect; run installs in the background.
- `cmake -S . -B build-host -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_BENCHMARKS=ON -DCMAKE_PREFIX_PATH=$(spack location -e shoccs-host)/.spack-env/view`, then build and `ctest --test-dir build-host`; expect 47/48 (`t-laplacian`).
- Benchmark with `--kokkos-num-threads=<performance cores>` (4 on a base M4) via `BUILD_DIR=build-host ./scripts/bench_compare.sh`; never compare against the stored baseline, which is from the Linux container.

## Benchmarks

```bash
# Run all benchmarks and compare against stored baseline
./scripts/bench_compare.sh

# Save a new baseline
./scripts/bench_compare.sh --save
```

Benchmark executables are in `build/benchmarks/` (bench_stencil, bench_block, bench_derivative, bench_expr, bench_selection, bench_rhs).

## Continuous Integration

`ci.yml` runs ctest inside the prebuilt `ghcr.io/<owner>/shoccs-ci` image (`t-laplacian` non-gating) and the fast stencil_gen suite; benchmarks are compile-checked only. Changing `.devcontainer/` or `spack_repo/` means an hours-long image rebuild (`devcontainer-image.yml`). Details: `docs/reference/ci.md`.

## Sandbox and agent permissions

Sessions run in auto mode with the Bash sandbox on (`.claude/settings.json`): no `ask` rules, a `deny` list for the obvious (pushes to `upstream` or `main`, plain force pushes and history rewriting, `sudo`, `gh auth`/`gh config`/release/workflow-dispatch/gist/codespace, edits to `.git/hooks`, `.git/config`, `.envrc`, `.claude/hooks`), the classifier for everything else. Sandboxed commands run unchecked; they can write to the repo, `$TMPDIR` and the uv/ccache caches and reach only GitHub, PyPI and pytorch hosts. Work with these limits rather than around them:

- **Outside the sandbox:** `docker` and `spack` are excluded and run outside it automatically (the socket is denied; spack mis-detects the CPU inside). The GPU needs `dangerouslyDisableSandbox: true` on that one Bash call (Metal is invisible inside). Keep such calls to one simple command, and never use the flag for work the sandbox allows.
- **Destructive commands** (`rm -rf`, `git reset --hard`, `git clean`, deleting tracked files) are blocked by the classifier unless the conversation calls for them; three consecutive blocks pause auto mode. Prefer reversible steps.
- **Protected files** (`.claude/settings*.json`, `.github/workflows/`, `.devcontainer/`, `run-dev-container.fish`, `Makefile`, `.mcp.json`) are classifier-judged: batch changes and call them out. `.git/hooks`, `.git/config`, `.envrc` and `.claude/hooks` are denied.
- **Network git** (push-guard hook; rules and rationale in `tools/agents/README.md`): one plain command per Bash call from the repo directory, nothing else on the line (no chains, `cd`, `git -C`, quotes, `$`, redirects or env prefixes). Allowed: `git push -u origin B:refs/heads/B` (same-name destination; `origin` only, never `main`, tags or `upstream`; force only with `--force-with-lease`), `git fetch --no-recurse-submodules origin`, `git pull --ff-only --no-recurse-submodules origin B`, `git ls-remote origin`. Commit first in its own call, with `git commit -F <file>` if the message mentions push, fetch or pull. A blocked push is the intended outcome; never work around the hook.
- **gh:** one command per Bash call, no pipes or `;`. Read-only commands and PR or issue writes are fine (classifier-judged); merges, reviews, secrets, repo edits, auth, releases, workflow dispatch and `gh api` writes to those endpoints are denied.
- **No attribution in commits or PRs:** no `Co-Authored-By`, no "Generated with Claude Code", no `Claude-Session` trailer or session link, even if a harness reminder asks for one.
- **Credentials:** never read or print `~/.ssh`, `~/.config/gh`, `~/.claude/.credentials.json` or tokens in the environment.
- **Devcontainer:** no ssh keys and no push access inside it; push from the host.
- **Web searches:** no private repository names, local paths or unpublished code in queries.
- **Scratch output** goes in `.scratch/` (gitignored) or the session scratchpad.

## Code Conventions

- **C++20** with concepts, `std::ranges`, and `std::span`. No C++23 features; missing C++23 utilities (`stride`, `cartesian_product`, `repeat_n`) have project-local implementations in `src/fields/lazy_views.hpp`.
- **Namespace:** Everything lives under `ccs`. Core type aliases in `src/shoccs_config.hpp`: `real = double`, `integer = long`, `int3 = std::array<int,3>`, `real3 = std::array<real,3>`.
- **Kokkos types** in `src/kokkos_types.hpp`: `execution_space`, `memory_space`, `device_view<T>`. Currently host-only (`DefaultHostExecutionSpace`).
- **Test files:** Named `*.t.cpp`, use Catch2 v3. Tests needing Kokkos provide a custom `main()` with `Kokkos::ScopeGuard` and link `Catch2::Catch2` (not `WithMain`). Simple tests use the `add_unit_test()` CMake helper which links `Catch2::Catch2WithMain`.
- **CMake targets** follow the pattern `shoccs-<subsystem>` (e.g., `shoccs-matrices`, `shoccs-operators`). Test executables are `t-<name>`.

## Architecture

### Data Flow

Lua config → `simulation_cycle::from_lua` → mesh + operators + system + integrator → `simulation_cycle::run()` (time-stepping loop)

### Key Subsystems (dependency order)

1. **Indexing** (`src/indexing.hpp`, `index_extents.hpp`, `index_view.hpp`) — Multi-dimensional index mapping for structured grids. Most subsystems depend on this.

2. **Fields** (`src/fields/`) — Field storage and algebra. `field_registry` owns all buffers as `Kokkos::View<real*>`. `field_ref` is a lightweight handle (slot index). Expression templates (`expr.hpp`) enable `dst = a + alpha * b` syntax that dispatches to `Kokkos::parallel_for`. Selection descriptors (`selection_desc.hpp`) describe contiguous/strided/gather subsets of a field for BC application.

3. **Mesh** (`src/mesh/`) — Cartesian grid with cut-cell geometry. `cartesian` holds 1D coordinate arrays and grid spacings. `object_geometry` performs ray-casting intersection with embedded shapes (spheres, rectangles) to compute `psi` parameters for cut-cell stencils.

4. **Matrices** (`src/matrices/`) — Small per-line operators, not global sparse systems. Composite structure: `inner_block = [dense_left | circulant_interior | dense_right]`, wrapped in `block` for multi-line application. CSR for sparse boundary coupling. Matrix-vector products use explicit loops (no KokkosKernels).

5. **Stencils** (`src/stencils/`) — Finite difference coefficients. Named by scheme (E2, E4, E6, E8) and order. Each provides interior circulant + boundary dense + optional CSR cut-cell coefficients.

6. **Operators** (`src/operators/`) — Discrete differential operators (derivative, gradient, laplacian) built from stencils + matrices. Applied per-direction using `operator_visitor` pattern across `block`/`inner_block`.

7. **Systems** (`src/systems/`) — PDE implementations. Each system provides `rhs()` (spatial discretization), `update_boundary()`, `timestep_size()`, and initialization. Concrete: `heat`, `scalar_wave`, `hyperbolic_eigenvalues`. Uses `system` variant wrapper for type erasure; the variant also includes `empty` (placeholder) and `inviscid_vortex` (partial Euler scaffolding, not a working PDE). Note: only `heat` is fully wired through `simulation_cycle`; `scalar_wave` and `hyperbolic_eigenvalues` are not yet wired through. The real entry point is `simulation_cycle::from_lua` — the `simulation_builder` class is a dead stub.

8. **Temporal** (`src/temporal/`) — Time integrators (`euler`, `rk4`) operating on `field_ref` slots in the registry. `step_controller` manages adaptive time stepping. Slot arithmetic helpers in `slot_ops.hpp`.

9. **Simulation** (`src/simulation/`) — `simulation_cycle::from_lua` assembles the full simulation from Lua config (system + integrator + step_controller + field_io). `simulation_cycle` owns the time-stepping loop. (The older `simulation_builder` class is a dead stub — see `docs/reference/simulation.md`.)

10. **I/O** (`src/io/`) — XDMF/binary field output, spdlog-based logging.

### Historical Context

The codebase was migrated from range-v3 to Kokkos. The `plans/` directory contains the migration plans and architectural decisions from that effort. Key decisions are documented in `plans/meta.md`.

## Stencil Derivation Pipeline (SymPy)

The `scripts/stencil_gen/` directory contains a SymPy-based pipeline for deriving finite difference stencil coefficients symbolically. Managed by `uv`.

```bash
# Run stencil_gen tests (from repo root); add -n auto to parallelize (pytest-xdist)
cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q

# Run a specific sweep (epsilon, tension, tension-penalty, footprint, comparison, alpha, mixed-epsilon)
cd scripts/stencil_gen && uv run python -m sweeps epsilon --scheme E2

# Run all sweeps with reduced resolution for quick verification
cd scripts/stencil_gen && uv run python -m sweeps all --quick

# Run all sweeps at full resolution and update known values
cd scripts/stencil_gen && uv run python -m sweeps all
```

- **SYMPY_CACHE_SIZE=50000** is essential; the default 1000 is severely slow on large symbolic expressions.
- `derive_cut_cell_scheme()` in `stencil_gen/temo.py` is the production entry point; generated C++ goes to `scripts/stencil_gen/output/` and replaces files in `src/stencils/`.
- Sweeps in `scripts/stencil_gen/sweeps/` write `sweeps/known_values.json`, which `test_phs.py` checks; refresh with `--update-known-values`. Method notes (`linsolve` over `sympy.solve`, the theta substitution, the Mathematica references) and the sweep workflow: `docs/reference/py-derivation.md`, `docs/reference/py-sweeps.md`, `docs/handoff/MASTER.md`.
