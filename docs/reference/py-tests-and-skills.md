# Python Stencil Framework: Tests & Skills (`scripts/stencil_gen/tests/`, `scripts/stencil_gen/stencil_gen/benchmarks/`, `.claude/skills/`, `ralph_wiggum.sh`)

> **Maturity:** mature · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose
This is the documentation + verification surface for the SymPy stencil-derivation pipeline (`scripts/stencil_gen/`). It bundles four things: (a) the **pytest suite** that proves the symbolic/codegen/PHS core is correct and that discovered-optimal stencil parameters stay stable; (b) the **`benchmarks/` sub-package** holding the Brady & Livescu 2019 reference problems used to score stencil stability; (c) five **Claude Code skills** (`.claude/skills/`) that orient a developer in how to drive the pipeline, sweeps, tests, and group-velocity analysis; and (d) **`ralph_wiggum.sh`**, the Bash plan-executor loop that ran the automated development that produced most of `scripts/stencil_gen`. It exists so a new developer can run, extend, and trust the Python tooling. The Python side already has detailed prose docs under `scripts/stencil_gen/docs/*` and `docs/handoff/*`; this doc orients and cross-links rather than duplicating them.

## Where it lives
| File | Role |
|------|------|
| `scripts/stencil_gen/tests/conftest.py` | Shared fixtures (`e2_1_uniform`, `e2_2_uniform`, `e4u_pipeline`, `e6u_pipeline`, `e8u_pipeline`, `assert_taylor_accuracy`), the `run_pipeline(p, nu, s)` helper, and the `--run-slow` option + slow-skip collection hook. |
| `scripts/stencil_gen/tests/test_phs.py` | Largest regression file. `_load_known_values()` / module-level `_KNOWN` read `sweeps/known_values.json` and assert stability/GV at discovered optima (E2_1, E4_1, footprint). |
| `scripts/stencil_gen/tests/test_brady2d_stability.py` | Second regression file. `KNOWN_VALUES_PATH` / `_load_known_values()` drive Brady-Livescu 2D layered-stability regressions; `pytest.skip` guards for absent sub-keys. |
| `scripts/stencil_gen/tests/test_benchmarks.py` | The only tests for the `benchmarks/` sub-package (PDE-residual, exact-solution-at-t0, FAMILIES enumeration, `run_calibration(max_layer=1)`). |
| `scripts/stencil_gen/tests/test_optimizer.py` | `save/load_known_values` roundtrip + `--update-known-values` CLI (via `tmp_path`/monkeypatch) and the `alpha_basin_survey` unit + slow end-to-end tests. |
| `scripts/stencil_gen/tests/fixtures/*_e4u1_reference.py` | Reference RBF boundary coefficient arrays (gaussian / multiquadric / tension E4u_1) hard-pasted into the C++ `t-*_E4u_1` tests; the Python copies guard cross-language consistency to 1e-12. |
| `scripts/stencil_gen/sweeps/known_values.json` | Single source of truth for discovered-optimal stencil parameters; written by sweeps `--update-known-values`, read by the regression tests. |
| `scripts/stencil_gen/stencil_gen/benchmarks/brady_livescu_2d.py` | Brady & Livescu 2019 §4.3 2D varying-coefficient advection reference. |
| `scripts/stencil_gen/stencil_gen/benchmarks/brady_livescu_4_2.py` | Brady & Livescu 2019 §4.2 reflecting-hyperbolic eigenvalue reference (purely imaginary continuous spectrum) — strictest stability discriminator. |
| `scripts/stencil_gen/stencil_gen/benchmarks/brady2d_calibration.py` | Enumerates all `(scheme, kernel, params)` `FAMILIES` and runs layered stability scoring; produces the calibration table. |
| `scripts/stencil_gen/stencil_gen/benchmarks/alpha_basin_survey.py` | Plan-43.9c multi-seed basin survey (research analog of Brady & Livescu Table 4). |
| `.claude/skills/{stencil-pipeline,stencil-sweeps,stencil-testing,group-velocity-analysis,ralph-wiggum}/SKILL.md` | Claude Code onboarding skills (auto-triggered by their `description`); each delegates detail to a `scripts/stencil_gen/docs/*_reference.md`. |
| `ralph_wiggum.sh` | Bash plan-executor: loops Claude Code through work/plan/review/commit passes against a plan file, enforcing the `RALPH_STATUS` protocol and a worktree-drift baseline guard. |

## Public API / entry points

**Running the suite** (always from `scripts/stencil_gen`):
```bash
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q              # default: 1035 of 1174 run
SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q --run-slow   # + the 139 @pytest.mark.slow
```

**conftest fixtures** (`tests/conftest.py`) — module-scoped derivation caches, reuse rather than recompute:
- `e2_1_uniform`, `e2_2_uniform` → `derive_e2_uniform_boundary(nu=1|2)`
- `e4u_pipeline`, `e6u_pipeline`, `e8u_pipeline` → `run_pipeline(p=2|3|4)` (returns `(updated_rows, solution_dict, w_syms, result)`)
- `assert_taylor_accuracy` (session) → callable checking row-wise Taylor moment conditions on a `B_u` matrix
- `run_pipeline(p, nu=1, s=0)` — module-level helper: `derive_boundary` + `build_conservation_system` + `solve_conservation`
- Hooks: `pytest_addoption(--run-slow)` + `pytest_collection_modifyitems` (skips `slow`-keyword items unless `--run-slow`)

**Regression machinery:**
- `test_phs.py`: `_load_known_values() -> dict | None` and module global `_KNOWN` (test classes only defined `if _KNOWN is not None`). Reads `sweeps/known_values.json`.
- `test_brady2d_stability.py`: `KNOWN_VALUES_PATH`, `_load_known_values()` (same file).
- `sweeps/known_values.json` top-level keys: `E2_1`, `E4_1`, `footprint`, `brady2d_calibration`, `brady2d_sweep`. (Nested `*_gv` keys live under `E2_1`/`E4_1`.)

**benchmarks package** (`from stencil_gen.benchmarks import ...`):
- `brady_livescu_2d`: `exact_solution(x,y,t)`, `initial_condition(x,y)`, `c_x(x,y)`, `c_y(x,y)`, `psi(x,y)`, `make_coefficient_field(N)`, `inflow_bc_x(y,t)`, `inflow_bc_y(x,t)`, constants `L_DOMAIN = sqrt(2)`, `PSI_OFFSET = 0.25`.
- `brady_livescu_4_2`: `initial_u(x)`, `initial_v(x)`, `exact_solution(x,t) -> (u,v)`, `continuous_eigenvalues(k_max=20) -> np.ndarray`, constant `L_DOMAIN = 1.0`.
- `brady2d_calibration`: `FAMILIES: list[tuple[str,str,dict,str]]`, `run_calibration(...)`, `format_calibration_table(results) -> str`, `_report_to_dict(report)`, `_E4_CLASSICAL_ALPHA`.
- `alpha_basin_survey`: `run_survey(...)`, `format_survey_table(survey) -> str` (wired into `stencil_gen.brady2d_cli --alpha-basin-survey`).

**Skills** (invoke via the Skill tool; each is a `SKILL.md` with YAML `name`/`description` + body): `stencil-pipeline`, `stencil-sweeps`, `stencil-testing`, `group-velocity-analysis`, `ralph-wiggum`.

**`ralph_wiggum.sh` CLI:**
```bash
ralph_wiggum.sh --plan FILE --mode {plan|work|full} --max-iterations N \
                [--model M] [--log-dir DIR] [--allow-dirty]
```
Env overrides: `RALPH_WIGGUM_{PLAN,MODE,MAX_ITERATIONS,MODEL,LOG_DIR,ALLOW_DIRTY,CLAUDE_FLAGS}`. Status protocol: each Claude pass must print `RALPH_STATUS: committed|done|blocked` and `RALPH_SUMMARY: <one-line>`. Default log dir is `.git/ralph-wiggum/`.

## How it works

**Test suite — two layers.** The bulk is *deterministic unit/property tests* over the symbolic core: `test_temo`, `test_interior`, `test_boundary`, `test_codegen`, `test_codegen_e4u`, `test_printer`, `test_eval_e2_1`, `test_phs` verify derivation, Taylor accuracy, and C++ codegen. Layered on top is a *file-driven regression contract*: sweeps discover optimal parameters and write `sweeps/known_values.json`; the regression test classes in `test_phs.py` and `test_brady2d_stability.py` load that JSON and re-assert stability / GV error / objective values at the stored optima. The sweep → JSON → regression loop is the project's mechanism for pinning research results so they cannot silently regress.

**Slow-test gating.** `pyproject.toml` registers the `slow` marker; `conftest.pytest_collection_modifyitems` deselects every `slow`-keyword item unless `--run-slow` is passed. So a default run collects 1174 tests but runs ~1035 — a green default run is **not** full coverage.

**Skip-when-absent regression contract.** Some regression sub-keys (the secondary footprint/tension-penalty `*_gv` keys, and the top-level `brady2d_optima` key) are intentionally not yet populated in `known_values.json`. The corresponding tests `pytest.skip()` rather than fail — a documented "activate once sweeps populate the keys" contract, not a stub. The primary `*_gv` keys (`tension_gv`, `gaussian_gv`, `multiquadric_gv` under E2_1/E4_1) **are** present and their tests actively pass.

**benchmarks → stability scoring.** `brady_livescu_4_2.py` (§4.2 reflecting hyperbolic) and `brady_livescu_2d.py` (§4.3 varying-coefficient advection) encode the reference PDEs. `brady2d_calibration.py` enumerates `FAMILIES` and runs the layered scoring (L1–L8). The L8 layer drives the compiled C++ solver via `stencil_gen/cpp_bridge.py` (which shells `build/src/app/shoccs` with a generated Lua config). Production callers are `stencil_gen/brady2d_stability.py` and `stencil_gen/brady2d_cli.py`.

**ralph_wiggum loop.** Each iteration builds a prompt heredoc (one of `plan` / `plan-review` / `work` / `review`), invokes `claude -p --dangerously-skip-permissions --no-session-persistence`, then parses `RALPH_STATUS`/`RALPH_SUMMARY` from the output. It snapshots `git status` as a baseline and aborts (exit 2) on worktree drift, and dies if Claude claims `committed` but `HEAD` did not advance. `--mode full` interleaves plan-refinement and work passes.

## How to extend

**Add a test.** Create `tests/test_<module>.py` mirroring the source module name; use `pytest.approx` for floats; cache expensive SymPy derivations behind `@pytest.fixture(scope="module")` (copy the `e4u_pipeline`/`e6u_pipeline` pattern in `conftest.py`, or add a new shared fixture there). Mark anything doing >10 eigendecompositions or a TEMO/Groebner derivation `@pytest.mark.slow`. Always export `SYMPY_CACHE_SIZE=50000`.

**Add a regression test.** Run a sweep with `--update-known-values` to populate the relevant key in `sweeps/known_values.json`, then add a `TestRegression*` class in `test_phs.py` that reads `_KNOWN[...]` and asserts stability. Guard with `pytest.skip` if the sub-key may be absent (follow the existing skip-when-absent classes).

**Add a benchmark.** Drop a module under `stencil_gen/benchmarks/`, then add a test class to `tests/test_benchmarks.py` with a finite-difference PDE-residual check + an exact-solution-at-t0 check (copy the existing classes).

**Add a skill.** Create `.claude/skills/<name>/SKILL.md` with YAML frontmatter (`name` + `description` — the description drives auto-triggering) and a concise body; keep a one-line pointer to the detailed `scripts/stencil_gen/docs/*_reference.md`. **Note the harness write-block below** — committing a new skill currently requires a manual human-in-the-loop step.

**Extend ralph_wiggum.** Edit one of the four `build_*_prompt` heredocs (plan / plan-review / work / review) or add a mode in `main()`'s `case` statement.

## Gotchas & invariants
- **`SYMPY_CACHE_SIZE=50000` is mandatory.** The default 1000 causes severe slowdowns on the large TEMO/Groebner symbolic expressions.
- **Default `pytest tests/` runs 1035 of 1174 collected** — 139 `@pytest.mark.slow` are silently deselected unless `--run-slow`. A green default run is partial coverage. The full default selection takes **minutes**, not seconds.
- **Non-determinism in the BO/optimizer tests.** `test_bo.py` and `test_optimizer.py` wrap stochastic BoTorch/pymoo; a `-x` run can stop on a flaky acquisition-optimization failure that passes on isolated re-run. Do not treat a single `-x` stop in `test_bo` as a real regression.
- **Some regression tests `pytest.skip()` when a sub-key is missing from `known_values.json`.** The secondary footprint/tension-penalty `*_gv` keys and the top-level `brady2d_optima` key are absent right now, so those guards silently skip — coverage looks green but those specific paths are untested until a sweep populates the keys. (The *primary* `*_gv` keys are present and asserted.)
- **Three of five skills are UNTRACKED in git** (`stencil-pipeline`, `stencil-testing`, `ralph-wiggum`); only `stencil-sweeps` and `group-velocity-analysis` are committed. Their companion reference docs (`docs/ralph_wiggum_reference.md`, `scripts/stencil_gen/docs/pipeline_reference.md`, `scripts/stencil_gen/docs/testing_reference.md`) are also untracked. A harness permission layer (see `docs/handoff/operating_conventions.md`) blocks all writes to `.claude/skills/**`; the two committed skills were landed via a manual human-in-the-loop workaround. A clean checkout at an older commit may lack the untracked assets.
- **`stencil-testing/SKILL.md` test-count table and "~400 tests/~7s" claim are STALE.** Real numbers: printer 16 (not 10), codegen 41 (not 29), group_velocity 64 (not 52), phs 101 (not 82); the table lists only 10 of 21 test files. Trust live `pytest --collect-only`, not that table.
- **cpp_bridge L8 path depends on a runnable `build/src/app/shoccs`.** As of 2026-06-04 the C++ build is fixed and green and the binary builds & loads (was the Kokkos 5.0→5.1.1 `create_graph` API break, fixed 2026-06-04). The slow smoke test still requires a runnable binary; it guards only on *binary absence* (`if not SHOCCS_BINARY.exists(): pytest.skip`), so under `--run-slow` it exercises the real solver. Default runs deselect it.
- **`ralph_wiggum.sh` is strict and container-only.** It refuses a dirty worktree (exit 2) unless `--allow-dirty`, aborts on worktree drift between passes, and dies if `committed` is claimed without `HEAD` advancing. It unsets `CLAUDECODE` (so it can run inside an existing Claude session) and uses `--dangerously-skip-permissions` — intended for container use only.
- **`ralph_wiggum.sh:152` still embeds range-v3-migration prompt text** ("the specific range-v3 patterns to replace") in `build_plan_prompt` — leftover from the original migration, misleading for today's non-migration plans. Default plan is still the stale `doc/hypre-semi-struct-plan.md`.

## Maturity & known gaps
**Verdict: mature.** Evidence: 1174 tests collect cleanly with no collection errors (1035 default / 139 slow). The deterministic core (`test_temo`, `test_interior`, `test_boundary`, `test_codegen`, `test_codegen_e4u`, `test_printer`, `test_eval_e2_1`, `test_phs`) runs fully green in ~13s; the only observed failure was a known stochastic BoTorch flake in `test_bo` that passes on isolated re-run. Every benchmark module has real production callers (`brady2d_stability.py`, `brady2d_cli.py`) and tests. All five skills' referenced functions/CLI/docs resolve on disk. `ralph_wiggum.sh` is git-tracked, executable, and actively maintained (commit `9f09ecf`, 2026-04-27, fixed a kill-0 polling hang). The drift is in narrative documentation, not broken references — so "mature but documentation-drifting."

Verified flagged items in this subsystem:
- **Three untracked skills + companion docs** (`stencil-pipeline`, `stencil-testing`, `ralph-wiggum`) — **partial-in-the-narrow-sense / verdict: finish (commit them).** Content is complete and every reference resolves; the only deficiency is they have never been committed (blocked by the `.claude/skills/**` harness write-layer). Remediation = commit via the documented manual workaround. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`stencil-testing/SKILL.md` stale test counts + "~400/~7s"** — **partial / verdict: finish.** Run instructions and fixture docs are accurate; the quantitative table is provably wrong. Regenerate from live `pytest --collect-only`. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`ralph_wiggum.sh` range-v3 prompt text + no test harness** — **partial / verdict: finish.** Functional and in active use, but `:152` carries misleading range-v3 phrasing and the script has zero automated tests / no CMake wiring. Fix = generalize the prompt line (and optionally the stale default plan). See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`*_gv` / `brady2d_optima` keys absent from `known_values.json`** — **verdict: keep (data backfill, not a code defect).** Primary `*_gv` regression keys are present and their tests pass; `brady2d_optima` is absent but its producer (`optimize.py --update-known-values`), CLI wiring, and consumer test are all implemented and unit-tested with mocks. Skip-when-absent is a documented populate-later contract.
- **`alpha_basin_survey.py` is research-oriented** — **verdict: keep.** Feature-complete: real CLI entry point (`brady2d_cli --alpha-basin-survey`), documented invocation, 5 passing monkeypatched unit tests + 1 slow end-to-end regression against the published Brady-Livescu α. "Slow + research" is its intended nature, not immaturity.
- **`test_cpp_bridge.py` / cpp_bridge L8** — **verdict: keep.** The test file (30 hermetic cases via a fake-shell binary) and the `cpp_bridge.py` module (4 production callers) are mature. The C++ build is fixed and green as of 2026-06-04 (was the Kokkos 5.1 `create_graph` break), so the L8 smoke test now drives a runnable binary. See [Cleanup Plan](../CLEANUP_PLAN.md).

No dead (zero-caller, safe-to-delete) items in this subsystem.

## Tests
This subsystem *is* largely the test surface: 21 `test_*.py` modules, 1174 tests collected. There is no `ctest` label here — these are Python tests run via `uv run pytest`, not the C++ Catch2 suite. Coverage highlights: deterministic symbolic/codegen core (green), file-driven regression of discovered optima (`test_phs.py`, `test_brady2d_stability.py`), benchmark-package sanity (`test_benchmarks.py`, 17 tests), optimizer/BO/pareto/sweep machinery (`test_optimizer.py`, `test_bo.py`, `test_pareto.py`, `test_sweep_*`), and the cpp_bridge contract (`test_cpp_bridge.py`, hermetic). **Not covered:** `ralph_wiggum.sh` has no automated tests (Bash); `SKILL.md` files have no validation harness (which is why the stale counts went unnoticed). **Conditional/disabled:** 139 `@pytest.mark.slow` (deselected without `--run-slow`); 1 xfail (`test_e4_1_conservation_fails_without_zeros`, documented infeasibility); 1 skipif (COBYQA/scipy≥1.14 in `test_optimizer`); runtime `pytest.skip` guards for absent `known_values.json` sub-keys; the cpp_bridge L8 smoke test (skips on binary absence; the C++ build is fixed and green as of 2026-06-04, so under `--run-slow` it drives the runnable binary).

## Related docs
- **Detailed Python references** (the deep docs this doc deliberately does not duplicate): `scripts/stencil_gen/docs/testing_reference.md` (test organization, fixtures), `scripts/stencil_gen/docs/pipeline_reference.md` (derivation pipeline), `scripts/stencil_gen/docs/sweeps_reference.md`, `scripts/stencil_gen/docs/group_velocity_reference.md`, `scripts/stencil_gen/docs/brady2d_stability_reference.md`, `scripts/stencil_gen/docs/optimization_reference.md`, `scripts/stencil_gen/docs/pareto_reference.md`, `scripts/stencil_gen/docs/mfbo_reference.md`, `scripts/stencil_gen/docs/bl42_reference.md`. Note: `pipeline_reference.md` and `testing_reference.md` are currently **untracked** (see Gotchas).
- **ralph_wiggum reference:** `docs/ralph_wiggum_reference.md` (untracked).
- **Handoff narrative:** `docs/handoff/{MASTER.md, framework_architecture.md, operating_conventions.md, completed_plans.md, scientific_findings.md, known_limitations.md, next_steps.md}` — `operating_conventions.md` documents the `.claude/skills/**` write-block; `scientific_findings.md` records the stability/GV results the regression tests pin.
- **Sibling reference docs:** the stencil-pipeline / matrices / operators / stencils references under `docs/reference/` cover the C++-side and the SymPy derivation internals consumed here.
- **Project entry:** `CLAUDE.md` (root) for the canonical sweep/pytest invocations.
