# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SHOCCS (Stable High-Order Cut-Cell Solver) is a C++ Cartesian cut-cell solver for time-dependent PDEs (heat equation, scalar wave, Euler equations). It uses high-order finite difference operators on structured grids with embedded boundaries. The codebase uses Kokkos for parallel execution.

## Documentation map

- `docs/ONBOARDING.md` — start here if you are new to the codebase.
- `docs/CAPABILITY_AUDIT.md` — per-subsystem maturity status (what actually works vs. scaffolding).
- `docs/reference/` — per-subsystem reference docs.
- `docs/CLEANUP_PLAN.md` — known issues and remediation plan (see §0/§0a for the Kokkos 5.1 build fix and the one remaining test failure).
- `docs/handoff/MASTER.md` — Python framework (stencil derivation / sweeps) handoff.

## Build Commands

> ✅ **Build status (2026-06-04):** Green. A Kokkos 5.0→5.1.1 `create_graph` API break was fixed by migrating the call sites to `create_graph<execution_space>(...)`; `ctest` passes 47/48, with one remaining failure (`t-laplacian`); `t-csr` and `t-E2_1` are fixed. Tracked in `docs/CLEANUP_PLAN.md` §0a.
> Also: if `build.ninja` errors with a missing cmake path (stale tree), re-run `cmake -S . -B build -G Ninja ...` (see "Reconfigure from scratch" below) to regenerate it.

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

## Benchmarks

```bash
# Run all benchmarks and compare against stored baseline
./scripts/bench_compare.sh

# Save a new baseline
./scripts/bench_compare.sh --save
```

Benchmark executables are in `build/benchmarks/` (bench_stencil, bench_block, bench_derivative, bench_expr, bench_selection, bench_rhs).

## Continuous Integration

GitHub Actions workflows in `.github/workflows/`:

- `ci.yml` — runs on PRs and pushes to main. The `cpp` job builds and tests inside the prebuilt dependency image `ghcr.io/lanl/shoccs-ci` (visibly skipped if the image hasn't been published yet); the gating ctest run excludes `t-laplacian` (known failure, `docs/CLEANUP_PLAN.md` §0a), which runs separately as a non-gating step. The `python` job runs the stencil_gen suite (slow tests deselected) via uv. Benchmarks are compile-checked, never executed in CI.
- `devcontainer-image.yml` — builds the lean `ci` stage of `.devcontainer/Dockerfile` and pushes it to GHCR. Triggers on pushes to main touching `.devcontainer/` or `spack_repo/`, weekly cron, or manual dispatch. Dispatch it manually after changing the dependency stack; a cold build takes hours (full Spack source build).
- `python-extended.yml` — weekly/manual run of the stencil_gen suite including `--run-slow` tests. Full-resolution sweeps stay out of CI.

## Code Conventions

- **C++20** with concepts, `std::ranges`, and `std::span`. No C++23 features; missing C++23 utilities (stride, zip_transform, cartesian_product, bind_back) have project-local implementations in `src/fields/lazy_views.hpp`.
- **Namespace:** Everything lives under `ccs`. Core type aliases in `src/shoccs_config.hpp`: `real = double`, `integer = long`, `int3 = std::array<int,3>`, `real3 = std::array<real,3>`.
- **Kokkos types** in `src/kokkos_types.hpp`: `execution_space`, `memory_space`, `device_view<T>`. Currently host-only (`DefaultHostExecutionSpace`).
- **Test files:** Named `*.t.cpp`, use Catch2 v3. Tests needing Kokkos provide a custom `main()` with `Kokkos::ScopeGuard` and link `Catch2::Catch2` (not `WithMain`). Simple tests use the `add_unit_test()` CMake helper which links `Catch2::Catch2WithMain`.
- **CMake targets** follow the pattern `shoccs-<subsystem>` (e.g., `shoccs-matrices`, `shoccs-operators`). Test executables are `t-<name>`.

## Architecture

### Data Flow

Lua config → `simulation_cycle::from_lua` → mesh + operators + system + integrator → `simulation_cycle::run()` (time-stepping loop)

### Key Subsystems (dependency order)

1. **Indexing** (`src/indexing.hpp`, `index_extents.hpp`, `index_view.hpp`) — Multi-dimensional index mapping for structured grids. All subsystems depend on this.

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

- **SYMPY_CACHE_SIZE=50000** is essential for performance — default 1000 causes severe slowdowns with large symbolic expressions.
- For linear systems with symbolic parameters, use `linear_eq_to_matrix` + `linsolve` (not `sympy.solve`). Linearize bilinear terms first with the theta-substitution trick (`theta = w*phi`).
- Reference Mathematica code in `mathematica-files/finitedifferences/` (`taylor.wl` for functions, `explicitr-E4d1.nb.pdf` for workflow).
- Key entry points: `derive_cut_cell_mathematica()` (singularity-free), `derive_cut_cell_scheme()` (legacy with psi-clamping).
- Generated C++ goes to `scripts/stencil_gen/output/` and replaces files in `src/stencils/`.
- **Sweep workflow:** Parameter-space explorations live in `scripts/stencil_gen/sweeps/` as standalone scripts. Sweeps discover optimal parameters and write results to `sweeps/known_values.json`. Regression tests in `test_phs.py` load from `known_values.json` to verify stability at those values. To update known values after a sweep: `uv run python -m sweeps epsilon --scheme E2 --update-known-values`.
