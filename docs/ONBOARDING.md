# SHOCCS — New Developer Onboarding

> Welcome! This guide gets a strong C++/numerics developer from `git clone` to
> "I built it, ran it, broke it, and know where everything lives" in an afternoon.
> It is honest about the sharp edges. The Kokkos 5.1 break that blocked a fresh
> build has been **fixed** (2026-06-04); `ctest` now passes **47/48**, with one
> remaining failure (`t-laplacian`) — `t-csr` and `t-E2_1` have since been fixed
> (see §3). Skim §3 before your first build.
>
> **Audited:** 2026-05-29; **build fixed:** 2026-06-04. If these dates are months
> old by the time you read this, re-verify the build status in §3.

---

## 1. What SHOCCS is (the two-codebases mental model)

SHOCCS = **Stable High-Order Cut-Cell Solver**. There are really *two* codebases
living in one repo, and keeping them straight is the single most useful thing you
can do on day one.

### Codebase A — the C++ solver (`src/`, `build/`)

A Cartesian **cut-cell, high-order finite-difference** PDE solver for
time-dependent problems (heat equation, scalar wave, hyperbolic eigenvalue
analysis). This is the implementation of the **Brady–Livescu JCP** line of work
(see `papers/BradyLivescu2019.pdf`, `papers/BradyLivscu2021.pdf`). It discretizes
on a structured Cartesian grid with **embedded boundaries** (spheres,
axis-aligned rectangles): grid lines are ray-cast against the geometry, the
cut-cell distance `psi` is computed per intersection, and special boundary
closures are applied there. It is **method-of-lines**: spatial operators produce
an RHS, time integrators (Euler, RK4) march it forward.

- Parallelism is via **Kokkos** (the codebase was migrated from range-v3 to
  Kokkos in early 2026; that migration is *done*).
- It is **host-only today** — `DefaultHostExecutionSpace`, serial + OpenMP. There
  is no GPU build yet (a plan exists, it was never carried through `src/`).
- Configuration is **Lua** (via sol2). You run `./build/src/app/shoccs foo.lua`.

### Codebase B — the Python stencil/stability/optimization framework (`scripts/stencil_gen/`)

A separate, `uv`-managed Python project that does the *math behind* the C++
stencils:

- **Derivation:** a SymPy pipeline that derives finite-difference coefficients
  symbolically (interior + boundary closures + cut-cell), implementing the **TEMO**
  (Truncation-Error-Matching-Optimization) cut-cell method from Brady–Livescu
  (2021). It emits C++ that is then copied into `src/stencils/`.
- **Stability scoring:** an 8-layer analytical cascade (group-velocity dispersion
  → Kreiss → 1D eigenvalues → 2D → non-normality → sparse 2D eigenvalues → a full
  end-to-end C++ `shoccs` run) evaluated against the Brady–Livescu 2019 §4.3
  benchmark.
- **Optimization:** finds optimal boundary-closure parameters (Nelder–Mead,
  COBYQA, NSGA-II Pareto fronts, multi-fidelity Bayesian optimization with
  BoTorch).

**Where the action is right now:** the C++ core has been frozen and stable since
~2026-03-27 (migration complete, all cleanup plans closed). Essentially all
recent development is in Codebase B. The only live C++ touchpoint is a thin
"stencil installation" bridge: the Python side derives a new closure and installs
its `.cpp` into `src/stencils/`. So expect to read a lot of C++, but to *write*
most new code in Python — unless your task is GPU enablement or finishing one of
the partial C++ features in §8.

The two codebases meet at exactly one place: **`cpp_bridge.py`** renders a Lua
config with runtime-parameterized stencil parameters, runs the compiled `shoccs`
binary, and parses `logs/system.csv`. That is "Layer 8" of the stability cascade.

---

## 2. Environment setup (devcontainer + spack)

You do not install dependencies by hand. The whole toolchain is provisioned by a
**devcontainer** whose dependencies are built by **spack**.

- **`.devcontainer/Dockerfile`** — multi-stage, based on `python:3.11-slim`.
  Installs GCC (the production compiler: default `CMAKE_CXX_COMPILER=/usr/bin/c++`,
  which is GCC 14 in the current image) and Clang (for clangd/clang-format / IDE
  tooling). It bootstraps **spack** and builds all C++ third-party libraries from
  `.devcontainer/spack.yaml`. It also installs `swig` + `cmake` so the Python
  `nlopt` dependency (no aarch64 wheel) can build from source.
- **`.devcontainer/spack.yaml`** — the C++ dependency manifest. A unified spack
  environment named **`shoccs-dev`** with `view: true`, so all libraries appear
  under a single view directory
  (`~/spack/var/spack/environments/shoccs-dev/.spack-env/view/`). CMake's
  `find_package` resolves everything from that view. Packages: lua, lua-sol2,
  fmt, pugixml, spdlog, cxxopts, catch2@3, boost, lapackpp, **kokkos** (+serial
  +openmp, cxxstd=20), kokkos-tools, google-benchmark, cmake, ninja.
- **`.devcontainer/devcontainer.json`** — bind-mounts the repo at `/workspace`,
  uses the Ninja generator, and its `postCreateCommand` runs the canonical
  configure:
  `cmake -S /workspace -B /workspace/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON`.
- **`run-dev-container.fish`** — convenience launcher (`./run-dev-container.fish
  [gcc|clang|<image:tag>]`). It `docker run`s the `shoccs-devcontainer:<flavor>`
  image, bind-mounts your `pwd` → `/workspace`, mounts `~/.gitconfig` / `~/.ssh`
  (read-only) and a Claude config dir, caps `NET_ADMIN`/`NET_RAW`, and drops you
  into `zsh`.
- **`Makefile`** (repo root) — **does NOT build the solver.** It is a Docker-image
  wrapper only (`make build-gcc`, `make shell`, etc.). Ignore it for code work; it
  operates one layer up, on container provisioning.

The Python side has its own environment managed by **`uv`** inside the container
(`scripts/stencil_gen/pyproject.toml`, Python ≥ 3.11). No separate setup — `uv
run` provisions it on first use.

---

## 3. Building the C++ solver

The correct, from-scratch procedure (run from `/workspace`):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
cmake --build build
```

Useful variants:

```bash
cmake --build build --target shoccs-exe   # just the solver binary
cmake --build build --target t-heat       # a single test executable
```

Build options (CMake): `BUILD_TESTING` (default ON, gates Catch2 + all `t-*`
targets), `BUILD_BENCHMARKS` (default OFF; turn on to build `benchmarks/`),
`SHOCCS_TPL_DIR` (optional extra prefix for finding TPLs; unused in the
devcontainer since spack provides everything). C++ standard is **C++20**.

**Real dependencies** (all from the spack view via `find_package`): Lua, sol2,
fmt (≥8), pugixml (≥1.10), spdlog (≥1.9), cxxopts, Catch2 **v3**, Boost (only the
header-only **mp11**), lapackpp, and **Kokkos**. Catch2 is required only when
`BUILD_TESTING=ON`; google-benchmark only when `BUILD_BENCHMARKS=ON`.

> ✅ **BUILD STATUS — fixed 2026-06-04 (was a known blocker).** A fresh build now
> compiles and `ctest` passes **47/48**. Full history and the one remaining failure
> are in **`docs/CLEANUP_PLAN.md` §0/§0a**.
>
> **What was wrong (Kokkos 5.1.1 Graph API).** The `shoccs-dev` spack env's Kokkos
> floated from the pinned `5.0` up to **5.1.1**, which changed the
> `Kokkos::Experimental::create_graph` overload set. The code called the old
> **2-argument** `create_graph(execution_space{}, closure)` form — 18 errors across
> `operators`, `systems`, and `t-graph_poc`. **Fix applied:** all 17 call sites were
> migrated to the templated 1-arg form `create_graph<execution_space>(closure)`
> (in `derivative.cpp`, `laplacian.cpp`, `heat.cpp`, `scalar_wave.cpp`,
> `graph_poc.t.cpp`); the node-building methods and all numerics are unchanged.
>
> **One remaining failure** (`t-laplacian`). Two earlier failures surfaced by the
> fix have since been fixed:
> - `t-csr` — **FIXED**: now uses a custom `main()` with `Kokkos::ScopeGuard` and
>   links `Catch2::Catch2` + `Kokkos::kokkos` (previously `Catch2WithMain` with no
>   `ScopeGuard`, which aborted under Kokkos 5.1's "OpenMP space constructed before
>   initialize()" check).
> - `t-E2_1` — **FIXED**: added `.margin(1e-12)` to its `Approx` comparisons (the
>   exact-rational E2 coefficients matched to ~15 digits, but a ~4e-16 roundoff
>   value could not match an exact 0 without a margin).
> - `t-laplacian` — **still fails**: cut-cell R-point values off ~2-3% (the interior
>   `d_vec` passes); a real cut-cell numerics question worth understanding before
>   relying on R-point results. See **CLEANUP_PLAN §0a**.
>
> **Stale `build/` tree (still possible on an older checkout).** If
> `build/build.ninja` errors with `cmake: not found` (exit 127) — a tree generated by
> a now-deleted spack-hashed CMake — `cmake --build build` cannot self-recover.
> Regenerate it (wiping `build/` first is safest):
> `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON`.

---

## 4. Running the solver

Once it builds, the binary lives at **`/workspace/build/src/app/shoccs`**. It
takes a positional Lua config file:

```bash
./build/src/app/shoccs heat.lua            # run a simulation
./build/src/app/shoccs heat.lua --check    # validate config and exit (no run)
./build/src/app/shoccs heat.lua --script "..."   # inline supplementary Lua override
./build/src/app/shoccs --help
```

`main()` (`src/app/shoccs.cpp`) opens a `Kokkos::ScopeGuard`, parses CLI with
cxxopts, builds a `sol::state` (base + math libs), runs the input Lua, then calls
`ccs::simulation_run(lua["simulation"])`.

### Example Lua configs (at repo root and `lua-configs/`)

| File | System | Scheme / integrator | Notes |
|------|--------|---------------------|-------|
| `heat.lua` | `heat` (diffusivity 1/30) | E2 order 2, RK4 | **The known-good starter.** 2D 51² grid, sphere obstacle, gaussian MMS, io on. |
| `scalar_wave.lua` | `scalar wave` | E2-poly order 1, RK4 | 2D 71², two spheres, hardcoded floating/dirichlet alphas. |
| `scalar_wave_1d.lua` | `scalar wave` | E2 order 1, RK4 | 1D 71, yz_rect cut. |
| `eigenvalues.lua` | `eigenvalues` (`hyperbolic_eigenvalues`) | E2-poly order 1 | Analysis config — eigenvalue extraction, no time-stepping (no integrator/io). |
| `lua-configs/brady_livescu_4_3_n61.lua` | `scalar wave` | E4u order 1, RK4 | 61² convergence variant. |
| `lua-configs/brady_livescu_4_3_long.lua` | `scalar wave` | E4u order 1, RK4 | 31², `max_time=100` long-stability variant. |
| `lua-configs/brady_livescu_4_3.lua` | `scalar wave` | placeholder | **Template, not standalone-runnable** — contains `--{{N}}--` / `--{{T_FINAL}}--` / `--{{SCHEME_TABLE}}--` tokens that `cpp_bridge.py` substitutes. |

**Config schema** (see `heat.lua` for a worked example): a top-level `simulation`
table with `mesh{index_extents, domain_bounds}`,
`domain_boundaries{xmin/xmax/ymin/ymax}`, `shapes{}` (embedded geometry),
`scheme{order, type, ...}`, `system{type, ...}`, `integrator{type}`,
`step_controller{max_time, cfl{...}}`, `manufactured_solution{...}`, and
`io{write_every_time | write_every_step}`.

### Where output and logs go

- **`logs/system.csv`** — per-step diagnostics (time, error norms). The Python
  bridge parses exactly this file.
- **XDMF + binary fields** — when an `io` table is present, the io subsystem
  writes an `.xmf` index file plus little-endian binary `double` data files, one
  per dump/variable, openable in ParaView/VisIt. Dump cadence is set by
  `write_every_time` (simulation-time interval) or `write_every_step` (step
  count).

---

## 5. Testing

C++ tests use **Catch2 v3** and are driven by ctest:

```bash
ctest --test-dir build                 # all 48 registered tests
ctest --test-dir build -L fields       # by label
ctest --test-dir build -R t-heat       # single test by name
```

Labels: `indexing`, `real3`, `fields`, `mesh`, `matrices`, `stencils`, `bcs`,
`operators`, `utils`, `io`, `systems`, `temporal`, `simulation`, `mms`. (Heads-up:
the io labels are slightly inconsistent — `t-interval`/`t-xdmf` are tagged
`shoccs-io` while `t-logging`/`t-field_io` are tagged `io`; CLEANUP_PLAN §5 has
the fix.)

### Green vs. red right now

After a fresh rebuild, `ctest` is **47 PASS / 1 FAIL** (98%). The one failure is a
pre-existing issue that the §3 build fix *revealed* (it was previously hidden
behind the loader error); it is not caused by the `create_graph` migration. Two
other failures surfaced by the fix have since been fixed:

- `t-csr` (matrices) — **FIXED**: now uses a custom `main()` with
  `Kokkos::ScopeGuard` and links `Catch2::Catch2` + `Kokkos::kokkos` (previously
  `Catch2WithMain` with no `ScopeGuard`, which `SIGABRT`ed under Kokkos 5.1).
- `t-E2_1` (stencils) — **FIXED**: `.margin(1e-12)` added to its `Approx`
  comparisons (one coefficient was off by ~4e-16, a sub-ulp `Approx` tolerance).
- `t-laplacian` (operators) — **still fails**: cut-cell R-point ("floating object"
  sphere) values off ~2-3%; the interior `d_vec` passes. A real cut-cell numerics
  question.

`t-laplacian` is tracked in **CLEANUP_PLAN §0a** — a good first task and a genuine
numerics investigation. There is also one
intentionally-disabled entry: a commented-out `#add_unit_test(divergence ...)` in
`src/operators/CMakeLists.txt:20` whose source no longer exists — deliberate, not a
bug (CLEANUP_PLAN §1).

### Python tests

```bash
cd scripts/stencil_gen
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q
```

1174 tests collected; 1035 run by default (139 are `@pytest.mark.slow`, run only
with `--run-slow`). The **deterministic symbolic/codegen/PHS core** (test_temo,
test_interior, test_boundary, test_codegen, test_phs, …) is fast (~13s) and
reliably green — treat it as ground truth. The **BoTorch/optimizer** suites
(`test_bo.py`, `test_optimizer.py`, `test_sweep_bo.py`) are stochastic and can
**flake under `-x`** (a failing acquisition-optimization retry); re-run a flaky
test in isolation before believing it. There is 1 documented `xfail` (E4_1
conservation infeasibility at `nextra=0`) and a few `skipif`/`pytest.skip` guards
when an optional dep or a `known_values.json` sub-key is absent — all expected.

---

## 6. Repo map

The repo is documented subsystem-by-subsystem under **`docs/reference/`**. Each
file is a deep dive (purpose, evidence of maturity, test coverage, sharp edges).
Read them in **dependency order** — bottom of the stack first:

| # | Subsystem | `src/` path | Reference doc |
|---|-----------|-------------|---------------|
| 1 | Indexing / core types | `src/shoccs_config.hpp`, `kokkos_types.hpp`, `indexing.hpp`, `index_extents.hpp`, `index_view.hpp` | [reference/core-types.md](reference/core-types.md) |
| 2 | Fields (storage + algebra) | `src/fields/` | [reference/fields.md](reference/fields.md) |
| 3 | Matrices (per-line operators) | `src/matrices/` | [reference/matrices.md](reference/matrices.md) |
| 4 | Stencils (FD coefficients) | `src/stencils/` | [reference/stencils.md](reference/stencils.md) |
| 5 | Operators (derivative/grad/lap) | `src/operators/` | [reference/operators.md](reference/operators.md) |
| 6 | Systems (PDE implementations) | `src/systems/` | [reference/systems.md](reference/systems.md) |
| 7 | Temporal (time integrators) | `src/temporal/` | [reference/temporal.md](reference/temporal.md) |
| 8 | Mesh (Cartesian + cut-cell geometry) | `src/mesh/` | [reference/mesh.md](reference/mesh.md) |
| 9 | Simulation (builder + cycle) | `src/simulation/` | [reference/simulation.md](reference/simulation.md) |
| 10 | I/O (XDMF/binary + logging) | `src/io/` | [reference/io.md](reference/io.md) |
| 11 | MMS (manufactured solutions) | `src/mms/` | [reference/mms.md](reference/mms.md) |

> Dependency note: `mesh` sits at #8 in the *reading* order for narrative reasons,
> but it is foundational — operators and every PDE system consume it. Think of
> core-types → fields → {matrices, mesh} → stencils → operators → systems →
> temporal → simulation → {io, mms} as the true DAG.

Supporting / cross-cutting subsystems with their own reference docs:
[random](reference/random.md) (test-support RNG),
[utils](reference/utils.md) (`bounded<T>`),
[app-and-build](reference/app-and-build.md) (`main()`, library, CMake).

Python side:
[py-derivation](reference/py-derivation.md) (SymPy/TEMO pipeline),
[py-brady2d](reference/py-brady2d.md) (8-layer stability cascade),
[py-sweeps](reference/py-sweeps.md) (parameter-sweep CLI),
[py-cpp-bridge](reference/py-cpp-bridge.md) (Python→C++ L8 bridge),
[py-tests-and-skills](reference/py-tests-and-skills.md) (test suite + Claude skills).

For the bird's-eye maturity picture (which subsystems are mature / partial / have
dead code), see **[CAPABILITY_AUDIT.md](CAPABILITY_AUDIT.md)** and the actionable
follow-ups in **[CLEANUP_PLAN.md](CLEANUP_PLAN.md)**.

---

## 7. The Python stencil framework — quickstart

```bash
cd scripts/stencil_gen
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q      # run the test suite
uv run python -m sweeps epsilon --scheme E2            # run one parameter sweep
uv run python -m sweeps all --quick                    # all sweeps, reduced resolution
uv run python -m sweeps all                            # full resolution + update known values
```

- **`SYMPY_CACHE_SIZE=50000` is essential** — the default of 1000 causes severe
  slowdowns with the large symbolic expressions this pipeline builds. Set it for
  every SymPy/pytest invocation.
- **Entry point doc:** start at **`docs/handoff/MASTER.md`** — it is the
  explicitly-designated front door for the framework, with the L1–L8 cascade,
  plan status (40–47), and the file-reading order. It links
  `framework_architecture.md` (module map + APIs),
  `operating_conventions.md` (commands), `scientific_findings.md` (results), and
  the `completed_plans.md` / `known_limitations.md` / `next_steps.md` trio. The
  `docs/handoff/*` files are well-maintained and self-flag their own stale
  sections — **trust them** over the older legacy design docs.
- **Sweeps workflow:** sweeps discover optima and write them to
  `sweeps/known_values.json`; regression tests in `test_phs.py` load that JSON to
  re-verify stability. To refresh after a sweep:
  `uv run python -m sweeps epsilon --scheme E2 --update-known-values`.
- **Claude Code skills** under `.claude/skills/` describe how to drive the
  pipeline: `stencil-pipeline`, `stencil-sweeps`, `stencil-testing`,
  `group-velocity-analysis`, and `ralph-wiggum`. Note that **three of them
  (stencil-pipeline, stencil-testing, ralph-wiggum) and several reference docs are
  not yet tracked in git** (a harness permission layer blocks writes to
  `.claude/skills/**`); CLEANUP_PLAN §3/§5 covers committing them via the
  human-in-the-loop workaround.

---

## 8. First tasks / good first issues

A graded ramp, pulled from **`docs/CLEANUP_PLAN.md`**. The build is green (47/48),
so you can dive straight in.

> **Warm-up: investigate the one remaining test failure** (CLEANUP_PLAN §0a). The
> two quick fixes are already done — `t-csr` now uses the custom-`main` +
> `Kokkos::ScopeGuard` pattern, and `t-E2_1` got a `.margin(1e-12)` on its
> sub-ulp `Approx` comparisons. The remaining good task is `t-laplacian`'s ~2-3%
> cut-cell R-point discrepancy: a genuine numerics investigation and a good way to
> learn the cut-cell path. Do it on a branch and confirm with
> `ctest --test-dir build`.

Other good tasks:

1. **Safe deletions — learn the build by removing dead code** (CLEANUP_PLAN §1).
   Each is a one-symbol removal with zero production callers; rebuild + ctest to
   confirm nothing breaks. Good starters:
   - Delete the unused `int2` alias (`src/shoccs_config.hpp:14`).
   - Delete `index::transpose<>` and `index::bounds<I>` (`src/indexing.hpp`,
     pre-Kokkos leftovers, test-only / uncalled).
   - Delete the dead commented `#add_unit_test(divergence ...)` line
     (`src/operators/CMakeLists.txt:20`) and the spurious `shoccs-random` link on
     `t-shapes` (`src/mesh/CMakeLists.txt:14`).
   These teach you the CMake target graph and the edit→build→test loop with
   minimal risk.

2. **Finish a feature: wire `scalar_wave` through `simulation_cycle`**
   (CLEANUP_PLAN §3). All three loop-driving end-to-end tests use only `heat`,
   yet `scalar_wave` is a full implementation reachable via the loop and the real
   `lua-configs/brady_livescu_4_3*.lua`. Add a `simulation_cycle::run()`
   regression for `type="scalar wave"` (RK4 + MMS, assert an error bound). This is
   the **biggest end-to-end coverage gap** and a great way to learn the data-flow
   spine. (Size: S–M.)

3. **Finish: add `xy_rect`/`xz_rect` geometry orientations** (CLEANUP_PLAN §3).
   `make_yz_rect` is production-reachable via `from_lua`; the `xy`/`xz`
   orientations exist in the generic `rect<I>` machinery but have no `from_lua`
   dispatch. Mirror the `yz_rect` branch and add a test for each. Completes the
   embedded-geometry orientation set. (Size: S.)

4. **Refresh a stale doc: `stencil-testing` SKILL.md** (CLEANUP_PLAN §3, doc-only,
   no C++ build needed). Its test counts are wrong (lists 10 of 21 test files;
   headline "~400 tests" vs the real 1174 collected / 1035 run). Update the table
   to all 21 files and fix the headline. A gentle Python-side first contribution
   that also forces you to actually run the suite. (Size: S.)

5. **Add output-content tests for the io subsystem** (CLEANUP_PLAN §3). Current io
   tests only smoke-check return values; nobody reads the binary payload back or
   asserts the XDMF Seek offsets. Add a `field_data.t.cpp` that writes then reads
   back and validates the D/Rx/Ry/Rz layout. Catches silent format drift. (Size:
   M; depends on Task 0.)

6. **Stretch / known incomplete C++ thread: GPU enablement.** `kokkos_types.hpp`
   is host-only (`DefaultHostExecutionSpace`) and a GPU plan exists but was never
   carried through `src/`. This is the clearest large intended-but-unfinished C++
   work item — a good "earn your stripes" project after the smaller tasks.

See [CLEANUP_PLAN.md](CLEANUP_PLAN.md) for the full ordered list (19 deletions, 3
deprecations, 13 finishes, 28 document-as-experimental), each with file:line
evidence.

---

## 9. Gotchas (the sharp edges)

- **`SYMPY_CACHE_SIZE=50000`** — always set it for any SymPy/pytest run in
  `scripts/stencil_gen`. Default 1000 is pathologically slow on the large
  symbolic expressions. (Repeated here from §7 because it bites everyone once.)
- **Kokkos is host-only.** No GPU build exists today. Don't expect CUDA/HIP; the
  execution space is `DefaultHostExecutionSpace` (serial + OpenMP).
- **Build is green (47/48), but two failure *modes* are worth recognizing** (§3).
  If `cmake --build build` errors with `create_graph` (you're on an un-migrated
  checkout) or a test fails to load with `libkokkoscore.so.5.0 not found` (stale
  `build/` tree), that's the historical Kokkos 5.1 issue — re-run the configure +
  build; see CLEANUP_PLAN §0. The one remaining red test (`t-laplacian`) is in §0a
  (`t-csr` and `t-E2_1` have been fixed).
- **Strong BC enforcement, NOT SBP-SAT.** The boundary closures enforce boundary
  conditions *strongly* (directly imposing values/derivatives). The
  pre-migration architecture spec
  (`SHOCCS_ARCHITECTURE_AND_KOKKOS_MIGRATION_SPEC.md` §4.2) calls the operators
  "summation-by-parts (SBP)" — **treat that label as misleading**; the project
  deliberately does not use SBP-SAT. When you read stability/closure code, think
  strong enforcement.
- **For symbolic linear solves, use `linear_eq_to_matrix` + `linsolve`, not
  `sympy.solve`.** And linearize bilinear terms first via the theta-substitution
  trick (`theta = w*phi`). `sympy.solve` does not scale to these systems.
- **The `plans/` directory + `ralph_wiggum.sh` workflow.** A large fraction of
  this codebase was produced by an automated plan-executor: `ralph_wiggum.sh`
  drives Claude Code non-interactively against numbered plan files in `plans/` in
  a work/review/commit cycle. Architectural decisions are logged in
  `plans/meta.md`. If you see commits like "47.8d: …" that is the plan/item
  numbering. (Caveat: `ralph_wiggum.sh` still embeds some range-v3-migration-era
  prompt text and a stale default plan path — CLEANUP_PLAN §5.) See the
  `ralph-wiggum` skill / `docs/ralph_wiggum_reference.md`.
- **Doc trust guide.** The repo has a lot of documentation, of mixed freshness.
  - **Trust:** `CLAUDE.md` (authoritative for the C++ side), everything under
    `docs/handoff/*` and `docs/reference/*`, `docs/brady2d_cpp_bridge_reference.md`.
  - **Use with heavy caveats:** `SHOCCS_ARCHITECTURE_AND_KOKKOS_MIGRATION_SPEC.md`
    — good for subsystem-level architecture and the PDE/method overview, but its
    *premise* (range-v3→Kokkos migration as future work) is obsolete (it's done),
    and specifics like a `cc_elliptic` system, the `mesh_object_info` struct
    members, and the stencil file list are wrong.
  - **Do NOT trust for current API:** `readme.md` (advertises range-v3 + a `make`
    build — both stale), `docs/field.md`, `docs/lazyness.md`,
    `docs/expression_templating.md`. Treat `docs/{design,discrete_operators,`
    `matrices,stencils,geometry}.md` as design-rationale archives: concepts hold,
    types/interfaces are pre-Kokkos.

---

## 10. Where to get help / key docs

| If you want… | Read |
|--------------|------|
| Authoritative C++ build/test/convention facts | **`CLAUDE.md`** (repo root) |
| Bird's-eye maturity of every subsystem | **`docs/CAPABILITY_AUDIT.md`** |
| Actionable cleanup/finish/delete tasks (with file:line) | **`docs/CLEANUP_PLAN.md`** |
| Deep dive on one C++ subsystem | `docs/reference/<subsystem>.md` (see §6) |
| The Python stencil/stability/optimization framework | **`docs/handoff/MASTER.md`** (entry point) → `framework_architecture.md` |
| How to run sweeps / derive stencils / analyze GV | `.claude/skills/{stencil-pipeline,stencil-sweeps,stencil-testing,group-velocity-analysis}/SKILL.md` |
| The automated plan-executor | `.claude/skills/ralph-wiggum/`, `docs/ralph_wiggum_reference.md` |
| Python→C++ bridge details | `docs/brady2d_cpp_bridge_reference.md`, `docs/reference/py-cpp-bridge.md` |
| The science (the actual papers) | `papers/BradyLivescu2019.pdf`, `papers/BradyLivscu2021.pdf`, plus the group-velocity/stability PDFs |
| Current build blocker, in full | `docs/CLEANUP_PLAN.md` §0 (and `/tmp/audit/build_blocker.md`) |

The repo owner is **Peter Brady** — author of the Brady–Livescu 2019/2021 papers
cited throughout. When a stability or boundary-closure question is ambiguous,
the papers and `docs/handoff/scientific_findings.md` are the canonical references.

Welcome aboard — fix the Graph API, watch `ctest` go green, then pick a §8 task.
