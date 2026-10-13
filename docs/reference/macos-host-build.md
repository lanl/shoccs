# Native macOS build (Apple silicon, spack)

## Purpose

An optional host build of the C++ solver outside the devcontainer, with Apple clang and a spack environment that each developer manages on their own machine. It exists so that C++ figures can be measured on the same hardware as Python or GPU experiments. The devcontainer (`docs/ONBOARDING.md` §2) remains the reference environment and the one CI uses.

Nothing environment-specific is checked in: every host carries its own concessions for its spack and package-repo versions, so the manifest lives with the user. What the repository contributes is the recipe side (`spack_repo/shoccs`) and this page.

## Requirements (as of spack 1.2.2, builtin repo at `releases/v2026.06`)

- Apple clang for C and C++ (Command Line Tools; clang 21 at the time of writing), registered with `spack compiler find`. There is no Fortran compiler, so OpenBLAS needs `~fortran`; it then builds its own C LAPACK and still provides the `blas` and `lapack` virtuals that lapackpp needs.
- Externals, once per machine: `spack external find cmake ninja curl`. Homebrew cmake and ninja save two builds; the system curl keeps lua's luarocks fetcher from rebuilding curl, openssl and perl.
- The project recipe repo `spack_repo/shoccs`, listed **before** `builtin` in the environment's `repos:`. It carries a one-line override of the builtin kokkos recipe (`packages/kokkos/package.py`) that adds `depends_on("llvm-openmp", when="+openmp %apple-clang")`. Apple clang ships no OpenMP runtime and the builtin recipe does not depend on llvm-openmp, so without it `kokkos +openmp` fails in Kokkos's own `find_package(OpenMP)` with `Could NOT find OpenMP_CXX`. Writing `^llvm-openmp` in the spec does not help: a spec cannot add a dependency to a package, so llvm-openmp lands only under blaspp. The override is inert for gcc and can go once spack-packages carries the same line.
- `benchmark ~performance_counters`: its default pulls the Linux-only libpfm4, and the concretizer otherwise falls back to benchmark 1.5.0 instead of flipping the variant. The shoccs recipe requires 1.6 or newer because the benchmark mains call `benchmark::Shutdown()`.
- `readline @8.2`, for the same reason as the devcontainer manifest (only 8.2's patches are mirrored on mirror.spack.io).

## The environment

Example manifest for a named environment `shoccs-host`; adjust the paths:

```yaml
spack:
  specs:
    - shoccs +tests +benchmarks ^kokkos@5.1.1 ^llvm-openmp
  repos:
    shoccs: /Users/me/projects/shoccs/spack_repo/shoccs   # precedes builtin: carries the kokkos override
  view: true
  concretizer:
    unify: true
  packages:
    all:
      require: "%[when=%c]c=apple-clang %[when=%cxx]cxx=apple-clang"
      prefer: [cxxstd=20]
    openblas: {require: "~fortran"}
    benchmark: {require: "~performance_counters"}
    readline: {require: "@8.2"}
```

```bash
spack env create shoccs-host shoccs-host.yaml
spack -e shoccs-host concretize
spack -e shoccs-host install --only dependencies   # about 15 minutes on an M4 the first time
```

`+profiling` is left off: kokkos-tools has only an unpinned `develop` branch. The spec matches the devcontainer otherwise (`.devcontainer/spack.yaml`).

## Building against it

```bash
cmake -S . -B build-host -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_BENCHMARKS=ON -DCMAKE_PREFIX_PATH=$(spack location -e shoccs-host)/.spack-env/view
cmake --build build-host
ctest --test-dir build-host
```

Expected: `AppleClang`, `Found OpenMP_CXX: -Xclang -fopenmp`, `Enabled Kokkos devices: OPENMP;SERIAL`; 47 of 48 tests pass, `t-laplacian` failing as in the container (`docs/CLEANUP_PLAN.md` §0a). Kokkos propagates `-march=armv8.4-a` (spack maps the m4 target to its ARMV84 architecture). `build-host/` is gitignored; benchmarks run with `BUILD_DIR=build-host ./scripts/bench_compare.sh`, but never compare them against the stored baseline, which comes from the Linux container.

## Threads

Thread count is `--kokkos-num-threads=N` on any executable (`OMP_NUM_THREADS` works too). macOS libomp has no thread affinity (`OMP_PROC_BIND` only warns) and the default team includes the efficiency cores, so benchmark with N = number of performance cores (4 on a base M4); 10 threads was slower in every case tried. Idle libomp threads sleep between parallel regions, which costs small kernels dearly: the shape-free 16^3 heat RHS takes 0.32 ms at 4 threads against 0.04 ms serial. `KMP_BLOCKTIME=infinite OMP_WAIT_POLICY=active` brings that to 0.06 ms and makes the whole shipped heat.lua step 0.27 ms, but slows the largest kernels (64^3 RHS from 1.4 to 2.3 ms), so choose per workload and measure. Full native figures: `docs/python-dsl-survey/05-measurements.md` Part IV E.

## Pitfalls

- **Spack and the Bash sandbox.** Inside the macOS sandbox used by agent sessions, `sysctl` is denied; archspec then cannot read the CPU brand string and reports x86_64, and writes to `~/.spack` and the spack tree are denied. `spack` is therefore excluded from the sandbox in the project settings and runs outside it: call it by its bare name, one command per call, no `cd`, redirect or pipe (any of those keeps the call sandboxed), and run long installs in the background. A sandboxed `spack arch` printing `darwin-*-x86_64` is the symptom.
- **The builtin repo ref.** Spack keeps one clone per package-repo git URL under `~/.spack/package_repos`, and only `spack repo update builtin` moves its ref. A tag pinned under `repos: builtin::` in an environment applies only after that command, and two environments pinning different refs share one checkout. Spack 1.2's default is branch `releases/v2026.06`; the devcontainer pins tag `v2026.06.0` from that branch and runs the update in its Dockerfile.
- **`$env` in paths.** Build subprocesses have no active environment, so `$env` in a `repos:` entry is left unexpanded and then resolved as a relative path one level short ("Unknown namespace: shoccs"). Use absolute paths or paths relative to the manifest.
- **Updating spack itself** is `git fetch --tags` plus `git checkout <tag>` in the spack clone; keep it on the tag the devcontainer pins (`ARG SPACK_VERSION` in `.devcontainer/Dockerfile`).

## Related docs

- `docs/ONBOARDING.md` §2 (devcontainer environment), `docs/reference/app-and-build.md` (third-party discovery through a spack view).
- `spack_repo/shoccs/packages/kokkos/package.py` and `packages/shoccs/package.py`.
- `docs/python-dsl-survey/05-measurements.md` Part IV E for the native measurements.
