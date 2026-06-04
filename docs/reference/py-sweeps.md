# Python Stencil Sweeps & Optimization CLI (`scripts/stencil_gen/sweeps/`)

> **Maturity:** partial · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose
The `sweeps` package is the parameter-space exploration and stencil-optimization **driver layer** for the SymPy stencil pipeline. It exposes a single `python -m sweeps <subcommand>` CLI over 12 named subcommands plus `all`. It spans two eras: **(1) research sweeps** that scan PHS/RBF shape parameters for stable differentiation matrices and persist optima to `known_values.json`, which `tests/test_phs.py` consumes as a regression contract; and **(2) optimization drivers** (single-objective, NSGA-II Pareto, multi-fidelity BoTorch) that are thin CLI shims over backend libraries in `stencil_gen.{optimizer,pareto,bo}`. The package deliberately separates slow exploration from the fast test suite via the JSON calibration store.

This doc orients and cross-links. The how-to-run detail and the optimization science already live in dedicated docs — do not re-derive here. See [Related docs](#related-docs).

## Where it lives
| File | Role |
| --- | --- |
| `sweeps/__main__.py` | CLI entry: argparse for all 12 subcommands + `all`; lazy-import dispatch (forwards flattened argv to each module's `main`); `_run_all()` runs only the 14 cheaper research sweeps. |
| `sweeps/_common.py` | Shared infra: `load_known_values()`/`save_known_values()` (+ `_KnownValuesEncoder`), `SCHEME_PARAMS` (E2/E4 → p,q,nextra,nu), `STABILITY_TOL=1e-10`, `SweepResult`, table printers, `bisect_threshold()`. |
| `sweeps/known_values.json` | Calibration store / regression contract. Live top-level keys: `E2_1`, `E4_1`, `footprint`, `brady2d_calibration`, `brady2d_sweep` (NOTE: `brady2d_optima` is **absent** — see gaps). |
| **Research sweeps (tier 1)** | |
| `sweeps/epsilon_sweep.py` | Gaussian/MQ epsilon sweep; `CLI_DEFAULT_EPS_FLOOR=1.5`; coarse-then-fine + floor-snap-warning logic. |
| `sweeps/tension_sweep.py` | Tension-spline sigma sweep; `CLI_DEFAULT_SIGMA_FLOOR=1.0` keeps the optimum off the PHS k=2 (sigma=0) limit. |
| `sweeps/tension_penalty_sweep.py` | Tension + conservation-penalty 2D (sigma, gamma) sweep. |
| `sweeps/footprint_sweep.py` | Stencil-footprint (`nextra`) sweep over (sigma, gamma). |
| `sweeps/comparison.py` | Read-only multi-method comparison table. |
| `sweeps/alpha_extraction.py` | Boundary-alpha extraction at optimal epsilon; `PRODUCTION_ALPHAS` (E2 only). |
| `sweeps/mixed_epsilon_sweep.py` | Per-row epsilon coordinate-descent (multiple strategies). |
| `sweeps/gv_stability_pareto.py` | 1D parametric GV-vs-stability dominance scan (read-only research aid). |
| `sweeps/gv_objectives.py` | Scalar GV *secondary*-objective wrappers over `stencil_gen.group_velocity`; `print_gks_advisory`; `gv_score_from_matrix` (used by `test_phs`). |
| **Optimization drivers (tier 2)** | |
| `sweeps/brady2d_sweep.py` | Single-axis Brady-Livescu 2D layered stability sweep; `CLASSICAL_E4_ALPHA`; `rank_for_l8` + `--validate-with-cpp` top-K L8 re-run. |
| `sweeps/optimize.py` | Single-objective optimizer shim (Nelder-Mead/COBYQA/SHGO/DE/staged) over `stencil_gen.optimizer`. |
| `sweeps/pareto.py` | NSGA-II multi-objective shim over `stencil_gen.pareto.run_nsga2`; `--persist` via `_pareto_io`. |
| `sweeps/bo.py` | Multi-fidelity BoTorch qMFKG shim over `stencil_gen.bo.run_mfbo`; `--baseline staged` head-to-head. |
| `sweeps/_pareto_io.py` | `ParetoResult` JSON persistence → `sweeps/pareto_fronts/<scheme>_<kernel>_<mangled>.json`. |
| `sweeps/_bo_io.py` | `BOResult` JSON persistence → `sweeps/bo_runs/<scheme>_<kernel>_<mangled>_<seed>.json`; restores int dict keys on load. |

## Public API / entry points
**CLI:** `uv run python -m sweeps <subcommand> [flags]`. Subcommands, by tier:

- **Research sweeps** (write `known_values.json` under `--update-known-values`; consumed by `test_phs.py`): `epsilon`, `tension`, `tension-penalty`, `footprint`, `comparison` (read-only), `alpha` (read-only), `mixed-epsilon`, `gv-stability-pareto` (read-only).
- **Optimization drivers** (CLI shims over `stencil_gen.{optimizer,pareto,bo}`): `brady2d`, `optimize`, `pareto`, `bo`.
- **`all`** — `_run_all(quick=...)` runs the 14 cheaper research sweeps; **does NOT** run `optimize`/`pareto`/`bo` (by design — see gotchas).

Every module exposes `main(argv: list[str] | None = None) -> int` doing its own argparse; `__main__.main()` re-parses and forwards a flattened argv. So you can also call e.g. `python -m sweeps.epsilon_sweep --scheme E2` directly.

**`_common` (shared infra):**
- `load_known_values() -> dict`, `save_known_values(data: dict) -> None` (writes via `_KnownValuesEncoder`: numpy scalars/arrays, dataclasses, `Path`, `complex` → `[re, im]`).
- `SCHEME_PARAMS = {"E2": {p,q,nextra,nu,label}, "E4": {...}}`, `STABILITY_TOL = 1e-10`.
- `@dataclass SweepResult(parameter, eigenvalue, stable, n=None, label="", extra={})`.
- `print_sweep_table(...)`, `report_stable_ranges(...)`, `bisect_threshold(f, a, b, threshold, *, tol=1e-4, maxiter=60)`.

**`gv_objectives` (secondary objectives over `stencil_gen.group_velocity`):**
- `interior_gv_error_max(p, nu=1, n_xi=200)`, `boundary_gv_error_max(p, q, nextra, nu, sigma, kernel, n_xi=200)`.
- `gv_score_from_matrix(D, n_xi=200) -> {"max_gv_error", "min_cutoff_xi"}` (used by `test_phs`).
- `cutcell_gv_min_C(scheme_params, psi_values, alpha_values, n_xi=200) -> (min_C, has_sign_reversal)`.
- `print_gks_advisory(D, *, label, n_xi=200) -> int` (advisory; returns count of outgoing modes).

**Persistence helpers** (one file per run, glob-discovered by tests):
- `_pareto_io`: `save_pareto_front(result, directory=PARETO_FRONTS_DIR) -> Path`, `load_pareto_front(path) -> dict`, `iter_pareto_fronts(directory=PARETO_FRONTS_DIR) -> Iterator[Path]`.
- `_bo_io`: `save_bo_run(result, directory=BO_RUNS_DIR) -> Path`, `load_bo_run(path) -> dict` (re-int-ifies keys), `iter_bo_runs(directory=BO_RUNS_DIR) -> Iterator[Path]`.

**Imported-by-tests constants:** `epsilon_sweep.CLI_DEFAULT_EPS_FLOOR (=1.5)`, `tension_sweep.CLI_DEFAULT_SIGMA_FLOOR (=1.0)`, `brady2d_sweep.CLASSICAL_E4_ALPHA` + `_report_to_dict()`.

## How it works
**Tier 1 — research sweeps → JSON → regression test.** The central data-flow contract:

```
sweeps <name> --update-known-values
        │  (builds D = -D_bc via stencil_gen.phs.build_diff_matrix_*,
        │   computes stability_eigenvalue, keeps feasible-then-minimizes)
        ▼
sweeps/known_values.json   (scalar optima: E2_1, E4_1, footprint, ...)
        ▼
tests/test_phs.py          (loads the key, asserts STABLE at that optimum)
```

Stability is the **only** hard gate: a parameter is feasible iff `max Re(eig(-D_bc)) < STABILITY_TOL (1e-10)`. GV error and `--check-gks` are advisory secondary objectives that never move the chosen optimum. Sweeps print formatted tables (`print_sweep_table`) and, with `--update-known-values`, merge their optimum into `load_known_values()` then `save_known_values()`.

**Tier 2 — optimization drivers.** Each `optimize`/`pareto`/`bo`/`brady2d` shim:
1. parses/resolves bounds (`_parse_bounds`/`_resolve_bounds` → `stencil_gen.optimizer.DEFAULT_BOUNDS`) and validates kernel dimensionality (`_validate_kernel_bounds_dim`);
2. builds an objective from a **dotted-path report field** (`layerN.field`, `layer_bl42.*`, `kreiss.*`) over `stencil_gen.brady2d_stability.brady2d_stability_score`, with `_infer_max_layer` choosing how deep to run;
3. calls the backend (`make_objective` + `multi_start_optimize`/`run_scipy_shgo`/`run_staged_optimize`; `run_nsga2`; `run_mfbo`);
4. optionally re-validates winners at L8 via `cpp_bridge` (`--validate-with-cpp`), prints a summary, and persists (`--persist`/`--update-known-values`).

Kernel dimensionality matters: `classical` is a **2D** alpha vector (`DEFAULT_BOUNDS[("E4","classical")] = [(-2,2),(0.05,2)]`); `tension`/`gaussian`/`multiquadric` are **1D**. The sweeps CLI kernels are exactly `{tension, gaussian, multiquadric, classical}` — there is **no `phs` kernel choice here** (PHS k=2 is reached via `tension` with `sigma → 0`, or via `footprint`).

## How to extend
**Add a new research sweep** (the common case):
1. Create `sweeps/<name>_sweep.py` exposing `main(argv: list[str]) -> int` that does its own argparse and imports helpers from `._common` (`SCHEME_PARAMS`, `STABILITY_TOL`, `load_known_values`/`save_known_values`, `print_sweep_table`) and optionally `.gv_objectives`.
2. Register a subparser in `__main__.main()` and add a lazy-import dispatch branch forwarding flattened args to your `main`. Add it to `_run_all`'s `sweeps` list if it is cheap/deterministic.
3. Persist optima by merging into `load_known_values()` + `save_known_values()` under a `--update-known-values` flag; then add a regression test in `tests/test_phs.py` that loads the new key. Copy `epsilon_sweep.py` / `tension_sweep.py` as the pattern.

**Add a new optimization driver:** the heavy lifting belongs in `stencil_gen/{optimizer,pareto,bo}.py` (objective construction over `brady2d_stability_score`, `DEFAULT_BOUNDS`, `_infer_max_layer`). The `sweeps/` file is only a CLI shim: parse bounds (`_parse_bounds`/`_resolve_bounds`/`_validate_kernel_bounds_dim` — copy these gates), call the backend, print a summary, and persist via a dedicated `_<name>_io.py` mirroring `_pareto_io`/`_bo_io` (own subdirectory, `_<Name>Encoder` for numpy/complex/dataclass/Path, `OrderedDict` canonical key layout, glob-based `iter` for test discovery). A new BO cheap-surrogate metric needs a per-fidelity default in `bo._DEFAULT_FIDELITY_FIELDS`.

## Gotchas & invariants
- **Stability is the only hard feasibility gate** (`max Re(eig(-D_bc)) < STABILITY_TOL = 1e-10`). `--include-gv`, `--check-gks`, and the GKS advisory are secondary/advisory and never move the chosen optimum; `--check-gks` is necessary-not-sufficient for instability.
- **eps-floor / sigma-floor** defaults (epsilon 1.5, tension 1.0) intentionally keep the persisted regression optimum **off** the degenerate `eps → 0` polynomial-reproduction limit / PHS k=2 (sigma=0) limit. The gaussian `stab_eig(eps)` landscape has multiple disjoint stable basins; pass `--eps-floor 0.0` / `--sigma-floor 0.0` to allow the unconstrained grid-min. A floor-snap emits a `UserWarning` — the optimum is then determined by the floor, not the objective.
- **`sweeps all` (`_run_all`) runs only the 14 research sweeps** and does **not** run `optimize`/`pareto`/`bo` — these exclusions are deliberate and plan-documented (optimization runs are not smoke tests). Do not treat `all` as a full CLI smoke test. `brady2d` **is** included.
- **JSON round-trips downgrade int dict keys to str.** `_bo_io._restore_int_keys` re-int-ifies `report_fields_by_layer`/`cost_model`/`n_evals_per_fidelity`/`wall_time_per_fidelity`, else `make_multi_fidelity_objective`'s `inferred > layer` comparison raises `TypeError`. Any new int-keyed `BOResult` field must be added to `_INT_KEYED_TOP_LEVEL`.
- **`--validate-with-cpp` NEVER alters `best_objective`.** An L8 (`max_layer=8`) disagreement is purely diagnostic. It silently **skips** (returns None) for unsupported scheme/kernel (only E4 + classical/tension/gaussian/multiquadric are bridged), an empty/non-finite winner, or a missing `SHOCCS_BINARY` — so a "skipped" validation is **not** a pass. Validation runs **before** `--persist`, so the JSON captures the L8 verdict.
- **Kernel dimensionality is enforced.** `_validate_kernel_bounds_dim` rejects mismatched `--bounds`: a 1D start fed to a 2D `params_from_vector` would be silently swallowed and return `+inf`/sentinel (the CLI would exit 0 with an infeasible result).
- **Two parallel "pareto" concepts.** `gv-stability-pareto` is a 1D parametric dominance scan (read-only, no persistence); `pareto` is true NSGA-II over 2+ metrics (persists to `pareto_fronts/`). They are NOT interchangeable; the help text cross-warns.
- **`bo --cost-model empirical` raises `NotImplementedError`** (reserved stub at parse time). Only `constant` (plan-46 `DEFAULT_COST_TABLE`) works.
- **`_pareto_io`/`_bo_io` duplicate the mangle + encoder** rather than importing from `sweeps.pareto`/`sweeps.bo`, specifically to keep the persistence path off pymoo / torch.
- Pareto and BO use **one file per run** (`pareto_fronts/`, `bo_runs/`) to avoid concurrency races on a shared JSON; tests discover them by globbing.

## Maturity & known gaps
**Verdict: partial** — mixed maturity by tier. The tier-1 research sweeps + `gv_objectives` + `_common` are **mature**: stable, recently touched, and they back the `known_values.json` regression contract (`E2_1`, `E4_1`, `footprint` present and asserted). The tier-2 drivers are recent but well-tested (`test_optimizer.py` 142, `test_sweep_pareto.py` 20, `test_sweep_bo.py` 31, `test_sweep_gv_objectives.py` 28) and carry persisted artifacts (one `bo_runs` JSON, two `pareto_fronts` JSONs). The "partial" label is driven by specific incomplete signals below.

Verified flagged items (from the audit):
- **`bo --cost-model empirical` — partial (stub).** `_resolve_cost_table` raises `NotImplementedError` (`bo.py:493`); only `constant` works. Deliberate reserved enum that fails fast at parse time with an actionable message; no caller, no test. The `__main__.py:325` help text omits the "future item" caveat that `bo.py:596` carries. Document-as-experimental; finish or trim later. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`optimize` → `known_values.json["brady2d_optima"]` — write path mature, data absent.** The deep-set write (`optimize.py:564-574`) is implemented, CLI-wired, and unit-tested; but no run has ever been persisted (the key has 0 occurrences in `known_values.json`), so `test_phs.py::TestRegressionBrady2DOptima` skips. This is a **documented deferral** (each entry costs ~5-30 min + the shoccs binary), not a stub. Operational follow-up only: seed one entry with a `staged --update-known-values` run.
- **`mixed_epsilon_sweep.py` — partial (untested wrapper).** Fully CLI-wired and runs all four strategies end-to-end, but has **zero behavioral test coverage of its own logic**; its `mixed_<kernel>` `--update-known-values` output is never written nor asserted. Explicitly an "exploratory sweep" per plan 43. Document-as-experimental. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`comparison.py` & `alpha_extraction.py` — partial (untested read-only tools).** Both wired, documented, and run to exit 0, but **no test imports either** (apparent hits are substring false-positives). `alpha_extraction.PRODUCTION_ALPHAS` defines **E2 only** — the `--scheme E4` production-comparison block is silently a no-op. Functional research/table tools with no regression net. Document-as-experimental.
- **`gv_stability_pareto.py` — mature (no dedicated test).** Audit refutes the "partial" suspicion: fully wired, dispatched, auto-run by `sweeps all`, runs to exit 0, and **intentionally** read-only (mirrors the mature `comparison` sweep; plan 45 explicitly preserves it alongside the NSGA-II `pareto` driver). Only gap is the absence of a dedicated pytest. **Keep as-is.**
- **`_run_all` / `all` — mature.** Wired, executes end-to-end, documented as the full-verification entry point. The exclusion of `optimize`/`pareto`/`bo` is mandated by plans 43/45/47 (optimizers are not smoke tests), not a defect. Only gap: no test pins the `_run_all` entry list.

Related cross-subsystem note: the **brady2d C++ CLI** (`stencil_gen/brady2d_cli.py`, a *different* subsystem) advertises a broken `--kernel phs` choice — see [`py-brady2d`](py-brady2d.md). The `sweeps` CLI does **not** offer a `phs` kernel, so that bug does not reach this subsystem.

## Tests
Run: `cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q`.
- **Sweep-wrapper tests:** `test_optimizer.py` (142; `optimize` CLI dispatch, `_run_cpp_validation` banners, `brady2d_sweep._report_to_dict`, `_common` roundtrip, `__main__` forwarding), `test_sweep_pareto.py` (20; `pareto` CLI + `_pareto_io` save/load/iter, `run_nsga2` monkeypatched), `test_sweep_bo.py` (31; `bo` CLI + `_bo_io`), `test_sweep_gv_objectives.py` (28; `gv_objectives` + `print_gks_advisory`).
- **Regression consumer:** `test_phs.py` loads `known_values.json` and asserts stability at `E2_1`/`E4_1`/`footprint`; it imports `CLI_DEFAULT_EPS_FLOOR`/`CLI_DEFAULT_SIGMA_FLOOR`, `gv_score_from_matrix`, and `_bo_io.load_bo_run`.
- **NOT covered:** `mixed_epsilon_sweep.py`, `gv_stability_pareto.py`, `comparison.py`, `alpha_extraction.py` have **no dedicated module tests** (only underlying `stencil_gen.phs` library functions are tested); `_run_all` / the `all` subcommand has no test.
- **Conditional skips:** many `test_phs.py` regression tests `pytest.skip()` (not fail) when a `known_values` sub-key is absent (notably `*_gv` GV fields and `brady2d_optima`), so a green run can mask un-exercised regression branches if the JSON lacks those keys.

## Related docs
- [`scripts/stencil_gen/docs/sweeps_reference.md`](../../scripts/stencil_gen/docs/sweeps_reference.md) — primary how-to for the tier-1 research sweeps. **NOTE: stale** (dated Apr 11) — its package layout and subcommand list predate plans 43/45/47, so it omits `pareto`, `brady2d`, `optimize`, `bo` and the `pareto_fronts/`/`bo_runs/` directories. Use this doc for the full subcommand map.
- [`scripts/stencil_gen/docs/optimization_reference.md`](../../scripts/stencil_gen/docs/optimization_reference.md) — the single-objective `optimize` driver and `stencil_gen.optimizer` backend (methods, bounds, objective dotted-paths, `--validate-with-cpp`).
- [`scripts/stencil_gen/docs/pareto_reference.md`](../../scripts/stencil_gen/docs/pareto_reference.md) — the NSGA-II `pareto` driver and `_pareto_io` persistence.
- [`scripts/stencil_gen/docs/mfbo_reference.md`](../../scripts/stencil_gen/docs/mfbo_reference.md) — the multi-fidelity `bo` driver (qMFKG), `DEFAULT_COST_TABLE`, fidelity fields.
- [`py-brady2d`](py-brady2d.md) — the `brady2d_stability_score` objective and L8 C++ bridge that **all** tier-2 drivers call; see also [`brady2d_stability_reference.md`](../../scripts/stencil_gen/docs/brady2d_stability_reference.md) for the objective/fidelity-field naming.
- [`py-derivation`](py-derivation.md) — the `stencil_gen.phs` differentiation-matrix builders and `group_velocity` used by the sweeps.
- `.claude/skills/stencil-sweeps/SKILL.md` — quick recipes (note: line ~105's `optimize` "persists to brady2d_optima" claim is currently un-realized — key absent).
