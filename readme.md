# Stable High-Order Cut-Cell Solver (shoccs)

SHOCCS is a Cartesian cut-cell solver for time-dependent PDEs (heat equation,
scalar wave, Euler/hyperbolic systems), using high-order finite-difference
operators on structured grids with embedded boundaries. It is the code
counterpart to the numerical algorithm in the
[JCP cut-cell paper](https://doi.org/10.1016/j.jcp.2020.109794). The solver is
written in C++20 and built on [Kokkos](https://github.com/kokkos/kokkos) for
parallel execution (currently host-only).

## Status

See [docs/ONBOARDING.md](docs/ONBOARDING.md) to get started and
[docs/CAPABILITY_AUDIT.md](docs/CAPABILITY_AUDIT.md) for a current,
code-verified picture of what works.

The build is green as of 2026-06-04 (a Kokkos 5.0→5.1.1 `create_graph` API break was
fixed); `ctest` passes 47/48, with one known failure (t-laplacian); t-csr and t-E2_1 fixed, tracked in
[docs/CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md) §0a. If a stale `build/` tree errors with
`cmake: not found`, regenerate it (see Building below).

## Dependencies

- [Lua](https://www.lua.org) and [sol2](https://github.com/ThePhD/sol2)
- [fmt](https://github.com/fmtlib/fmt)
- [pugixml](https://pugixml.org/)
- [spdlog](https://github.com/gabime/spdlog)
- [cxxopts](https://github.com/jarro2783/cxxopts)
- [Catch2 v3](https://github.com/catchorg/Catch2)
- Boost (header-only [mp11](https://github.com/boostorg/mp11))
- [lapackpp](https://github.com/icl-utk-edu/lapackpp)
- [Kokkos](https://github.com/kokkos/kokkos)
- [google-benchmark](https://github.com/google/benchmark) (optional, for benchmarks)

These are provisioned by the project's devcontainer, which uses
[spack](https://spack.io/) to build the toolchain (see `.devcontainer/`). The
recommended workflow is to build and develop inside that container.

## Building

```shell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
cmake --build build
```

Add `-DBUILD_BENCHMARKS=ON` to also build the benchmark executables.

Note: the root `Makefile` is a wrapper for building the devcontainer Docker
images, not the solver itself. Use the CMake + Ninja commands above to build
the solver.

## Running

```shell
./build/src/app/shoccs heat.lua          # run the solver on a config
./build/src/app/shoccs heat.lua --check  # validate the config and exit
```

Several example `.lua` configs are provided at the repository root
(`heat.lua`, `scalar_wave.lua`, `scalar_wave_1d.lua`, `eigenvalues.lua`) and
under `lua-configs/`.

Run the test suite with ctest:

```shell
ctest --test-dir build              # all tests
ctest --test-dir build -L systems   # by label
ctest --test-dir build -R t-heat    # single test by name
```

## Documentation

- [docs/ONBOARDING.md](docs/ONBOARDING.md) — start here
- [docs/CAPABILITY_AUDIT.md](docs/CAPABILITY_AUDIT.md) — current capabilities
- [docs/reference/](docs/reference/) — per-subsystem reference
- [docs/handoff/MASTER.md](docs/handoff/MASTER.md) — entry point for the Python
  stencil-derivation / optimization framework (`scripts/stencil_gen/`)

## Misc

Copyright assertion C20039
