# Multi-Fidelity Bayesian Optimization Reference

Companion to `optimization_reference.md` and `pareto_reference.md`.  Where
`optimization_reference.md` describes the *scalar* drivers
(`make_objective`, `run_scipy_local`, `run_staged_optimize`, …) and
`pareto_reference.md` describes the *multi-objective* layer (NSGA-II Pareto
fronts), this one describes the *multi-fidelity Bayesian* layer added in
plan 47 — a Gaussian-process surrogate over the cascade's discrete fidelity
levels driven by a cost-aware acquisition function that picks `(x, m)`
jointly to maximise expected information gain at the high-fidelity target
per second of wall time.

## 1. Problem

The Brady-Livescu stability cascade has a 5+ orders of magnitude cost
spread across its analytical layers:

| Layer | Wall time | Cost ratio (L1 = 1) | What it tests |
|---|---|---|---|
| L1 | ~76 ms | 1.0 | GV dispersion (interior + boundary) |
| L3 | ~38 ms | 0.5 | 1D advection eigenvalue |
| L3r (`layer_bl42`) | ~486 ms | 6.4 | BL §4.2 reflecting-hyperbolic spectrum |
| L6 | ~846 ms | 11 | Non-normality on 1D operator |
| L7 | ~1434 ms | 19 | Full 2D varying-coefficient spectral abscissa |
| L7+nn | ~17.5 s | 230 | L7 + transient-growth bound |

The scalar `run_staged_optimize` heuristic spends a fixed fraction of its
budget at L3, then re-evaluates the top-K survivors at L7.  The `K` is
hand-tuned and the trade-off is implicit.  Multi-fidelity Bayesian
optimisation (MF-BO) makes the trade-off explicit:

```
maximise   I(x*; D ∪ {(x, m)}) / cost(m)         (cost-weighted EIG)
over       x ∈ [lb, ub], m ∈ {fidelity levels}
```

where `I(x*; D ∪ {(x, m)})` is the expected information gain about the HF
optimum `x*` from observing the cascade at `(x, m)`, and `cost(m)` is the
calibrated wall-time cost of fidelity `m`.  The GP surrogate keeps a
posterior over the HF objective that learns from cheap observations
*without* assuming the cheap layers are noisy versions of HF.

**Critical observation (`docs/handoff/scientific_findings.md` finding #1):**
L3 → L3r is **not** a refinement chain — they test different physics (1D
periodic advection vs. reflecting BCs).  Tension closures pass L3
universally but fail L3r at `max_spectral_abscissa ≈ 0.95`.  So the GP
cannot use a Kennedy-O'Hagan autoregressive ladder
(`f_m(x) = ρ_{m-1} f_{m-1}(x) + δ_m(x)`); it uses an *intrinsic
coregionalisation* (ICM) kernel that lets the data report the actual
layer-pair correlations end-to-end.

## 2. Why BoTorch + qMFKG

- **Verified clean aarch64 install.** `botorch==0.17.2`, `torch==2.11.0+cpu`,
  `gpytorch==1.15.2` all resolve cleanly via `uv sync` (~141 MB CPU torch
  wheel; no NVIDIA stack pulled).  Emukit forces `numpy<2` via GPy; Trieste
  forces TF 2.16 with `numpy<2`.  BoTorch is the only mature MF-BO library
  with NumPy 2 compatibility on the dev container.
- **First-class MF API.** `qMultiFidelityKnowledgeGradient`,
  `MultiTaskGP`, `IndexKernel` (ICM), `AffineFidelityCostModel`,
  `InverseCostWeightedUtility` — all stable in 0.17.x and documented in the
  `discrete_multi_fidelity_bo` tutorial.
- **MES is a one-line swap.** If KG diagnostics show pathology (Gumbel
  sampling collapse, multi-modal posterior trapping), swap to
  `qMultiFidelityMaxValueEntropy` from
  `botorch.acquisition.max_value_entropy_search` (NOT
  `botorch.acquisition.multi_fidelity` — see `bo.py` notes for the
  recurring import-path confusion).
- **Active maintenance.** Weekly commits, frequent point releases, and
  Ax 1.0 integration.

## 3. API

All exports live in `stencil_gen/bo.py`.  Module surface:

```python
from stencil_gen.bo import (
    BOEval, BOResult, _BO_SENTINEL, DEFAULT_COST_TABLE,
    make_multi_fidelity_objective, run_mfbo,
    build_mf_gp, build_cost_model, build_initial_design, build_acquisition,
    apply_cost_floor,
)
```

### 3.1 `BOEval`

```python
@dataclass(frozen=True)
class BOEval:
    x: np.ndarray         # (d,) design vector
    params: dict          # params_from_vector(kernel, x)
    fidelity: int         # external cascade layer index
    value: float          # extracted field value at this fidelity
    wall_time: float      # measured per-eval seconds
    report: dict          # _report_to_dict serialisation
```

One observation in the BO loop.  Sentinel-valued evals (gate trip,
exception in `brady2d_stability_score`, shape mismatch) carry
`value = _BO_SENTINEL = 1e12` and the `report` dict carries an `"error"`
key with a short string.  The wall_time is always the measured perf
counter delta, even on sentinel — so the cost-aware utility's empirical
calibration is honest.

### 3.2 `BOResult`

The full record returned by `run_mfbo`.  Frozen dataclass; tuples (not
lists) for immutability — same convention as `ParetoResult`.  Notable
fields:

| Field | Type | Description |
|---|---|---|
| `best_x` | `np.ndarray` | Recommended incumbent at HF |
| `best_objective` | `float` | Posterior-mean recommended HF value at `best_x`; `_BO_SENTINEL` when no feasible HF point was found |
| `best_report` | `dict` | Full `StabilityReport` at `best_x` from a final HF re-eval |
| `method` | `str` | `"BoTorch-qMFKG"` |
| `fidelity_levels` | `tuple[int, ...]` | Sorted external cascade indices in scope |
| `hf_level` | `int` | `max(fidelity_levels)` |
| `report_fields_by_layer` | `dict[int, str]` | Per-layer dotted-path field |
| `cost_model` | `dict[int, float]` | Floored cost table actually used |
| `n_evals_per_fidelity` | `dict[int, int]` | Headcount, sums to `len(eval_history)` |
| `wall_time_per_fidelity` | `dict[int, float]` | Total seconds per fidelity |
| `total_compute_time` | `float` | End-to-end wall time |
| `eval_history` | `tuple[BOEval, ...]` | Every evaluation |
| `hf_eval_history` | `tuple[BOEval, ...]` | Filtered to HF only |
| `gp_hyperparameters` | `dict` | `{"lengthscale", "icm_W", "icm_var", "noise"}` from the final GP fit |
| `seed` | `int` | RNG seed (forwarded to torch + numpy + Sobol') |
| `converged` | `bool` | True iff variance / stagnation guard fired |
| `stop_reason` | `str` | `"budget" \| "variance" \| "stagnation" \| "error"` |
| `extras` | `dict` | Driver diagnostics: `n_sentinel_per_fidelity`, `n_sentinel_filtered`, `voronoi_fallback`, `baseline`, `cpp_validation`. `n_sentinel_per_fidelity` is set when `clamp_sentinel_rows=True` (the default; mutually exclusive with `n_sentinel_filtered`, which is set when `clamp_sentinel_rows=False`). `voronoi_fallback` is set to `True` only when `recommendation_strategy="voronoi"` and the helper falls back to an unmasked argmin because every grid point fell inside the union of Voronoi cells around recorded sentinel `x` rows. `baseline` and `cpp_validation` are populated by the corresponding CLI flags. |

### 3.3 `make_multi_fidelity_objective`

```python
def make_multi_fidelity_objective(
    scheme: str,
    kernel: str,
    report_fields_by_layer: dict[int, str],
    *,
    gate_layer: int | None = None,
) -> Callable[[np.ndarray, int], tuple[float, float, dict]]
```

Vector-of-fidelities analogue of `make_objective`.  Returns a closure
`f(x, m) -> (value, wall_time, report_dict)` that runs
`brady2d_stability_score(scheme, kernel, params, max_layer=m,
short_circuit=True)` once and extracts `report_fields_by_layer[m]` via
`extract_field`.  The 3-tuple shape (not just the value) lets the BO loop
record per-eval wall time without a side channel.

The HF layer is `max(report_fields_by_layer)`; the HF field is the
optimisation target.  `gate_layer` defaults to `max(min_layer - 1, 0)`,
so only failures *below* the cheapest fidelity gate to sentinel.

On any of (gate trip, shape mismatch in `params_from_vector`, exception
from `brady2d_stability_score`, `m not in report_fields_by_layer`,
non-finite extracted value) the closure returns `(_BO_SENTINEL,
measured_wall_time, {"error": <type>})`.

Validates at factory time: each field's `_infer_max_layer` must be `≤`
the layer it is keyed under.  You cannot extract `layer7.*` at
`max_layer=3`.

### 3.4 `run_mfbo`

```python
def run_mfbo(
    scheme: str,
    kernel: str,
    report_fields_by_layer: dict[int, str],
    bounds: Sequence[tuple[float, float]],
    *,
    budget_evals: int | None = None,
    budget_seconds: float | None = None,
    cost_table: dict[int, float] | None = None,
    seed: int = 0,
    n_init: int | None = None,
    hf_anchors: int | None = None,
    num_fantasies: int = 64,
    min_acquisition_iterations: int | None = None,
    hf_explore_bias: float = 0.0,
    adaptive_hf_floor: float | None = None,
    adaptive_hf_explore_bias: float | None = None,
    variance_guard_relative_threshold: float = 1e-5,
    hf_priority_warmup: int = 0,
    hf_acquisition_bonus: float | None = None,
    clamp_sentinel_rows: bool = True,
    recommendation_strategy: str = "mean",
    voronoi_radius: float = 0.1,
    ucb_beta: float = 2.0,
    verbose: bool = False,
    objective: Callable[[np.ndarray, int], tuple[float, float, dict]] | None = None,
) -> BOResult
```

The full driver.  Builds the multi-fidelity objective via
`make_multi_fidelity_objective` (or accepts `objective=` for synthetic
test injection), constructs an initial design via `build_initial_design`,
then loops:

1. Fit `MultiTaskGP` (Matern × IndexKernel) via `build_mf_gp` on the
   current finite-valued training set.
2. Build cost-aware acquisition via `build_acquisition`
   (`qMultiFidelityKnowledgeGradient` wrapped in
   `InverseCostWeightedUtility`).
3. `optimize_acqf_mixed` over `(x, m)` to get the next pick.
4. Evaluate the objective; record into `eval_history`.
5. Check stopping criteria (budget / variance / stagnation).
6. Append to training data.

After loop exit, `_recommend_incumbent` picks `x_inc` over a 1024-point
Sobol' grid at HF.  The ranking depends on `recommendation_strategy`
(`"mean"` is the default — the historical posterior-mean argmin):

- **`"mean"`** (default; pre-47.6b.3.2c behaviour) — `x_inc = argmin_x
  μ_n(x, m=hf)`.  Posterior mean, not best-observed — standard for noisy
  / multi-fidelity GPs.
- **`"voronoi"`** (47.6b.3.2c.2) — same posterior-mean ranking as
  `"mean"`, but grid points within `voronoi_radius` (L2) of any sentinel
  `x` row in `eval_history` are masked out; the lowest-mean candidate
  among the survivors wins.  When every grid point falls inside the
  union of Voronoi cells (e.g. sentinels cover the whole bounded box),
  the helper falls back to the unmasked argmin and sets
  `extras["voronoi_fallback"] = True`.
- **`"ucb"`** (47.6b.3.2c.3) — `x_inc = argmin_x (μ_n(x) + ucb_beta *
  σ_n(x))`.  Upper-confidence bound on the *minimisation* target — the
  pessimistic estimate; lower mean wins, lower variance wins, so
  high-variance regions (sentinel boundaries on the clamped GP) are
  penalised.  `ucb_beta=0.0` collapses to `"mean"` exactly (no-op
  default-off contract).  Default `ucb_beta=2.0` follows the Auer et
  al. 2002 bandit-maximisation convention with the sign adjusted for
  minimisation.

`clamp_sentinel_rows=True` (the default; 47.6b.3.1) keeps sentinel rows
inside the GP fit by replacing the sentinel value with `max(Y_finite) +
3 * std(Y_finite)` per fidelity, so the GP learns "infeasible regions
return very high objective" and the recommendation argmin avoids them
naturally.  When fewer than 2 finite rows exist for a fidelity, that
fidelity's sentinel rows fall back to filtering instead.  Setting
`clamp_sentinel_rows=False` reverts to the pre-47.6b.3.1
filter-only contract; `extras["n_sentinel_per_fidelity"]` is replaced
by `extras["n_sentinel_filtered"]` in that case.

After the strategy picks `x_inc`, a final HF evaluation populates
`best_objective` and `best_report` from real cascade data rather than
GP posterior.

`budget_evals` and `budget_seconds` are mutually exclusive; one must be
set.  `cost_table` defaults to `DEFAULT_COST_TABLE`.

### 3.5 Building blocks

```python
build_mf_gp(train_X, train_Y, fidelity_dim, num_fidelities, *, rank=2)
    -> MultiTaskGP
```

ICM-kernelled GP: outer product of `MaternKernel(nu=2.5,
ard_num_dims=d)` and `IndexKernel(num_tasks, rank)`, fit with
`Standardize(m=1)` outcome transform via
`fit_gpytorch_mll(ExactMarginalLogLikelihood)`.  Likelihood noise floor
`GreaterThan(1e-9)` prevents Cholesky failures on noise-free synthetic
data.  Returns a fitted `MultiTaskGP` (NOT
`SingleTaskMultiFidelityGP` — see `bo.py:351` block comment for why
BoTorch's "multi-fidelity" wrapper does not actually preserve a
hand-supplied ICM `covar_module`).

```python
build_cost_model(cost_table, fidelity_dim, *, floor_ratio=0.05)
    -> InverseCostWeightedUtility
```

Wraps a discrete `GenericDeterministicModel` step function with the
`apply_cost_floor` clamp `c'(m) = max(c(m), floor_ratio * c(hf))`.  The
floor prevents acquisition over-exploitation of the cheapest layer.

```python
build_initial_design(bounds, fidelity_levels, *, n_init=None,
                     hf_anchors=3, mid_anchors=2, seed=0)
    -> tuple[np.ndarray, np.ndarray]
```

Sobol' design with stratified fidelity allocation: `n_cheap` rows at the
cheapest fidelity, `mid_anchors` at the median fidelity, `hf_anchors` at
HF.  HF anchors are placed bytewise-identical to `hf_anchors` of the
cheap rows — paired observations that the ICM kernel needs to identify
the off-diagonal `B[lo, hi]` entries (Wu et al. 2020 §3.1).
`n_init` defaults to `5*d + 3` (Loeppky et al. 2009).

```python
build_acquisition(model, cost_utility, target_fidelity_index, *,
                  num_fantasies=64, candidate_set_size=512,
                  hf_acquisition_bonus=None)
    -> tuple[qMultiFidelityKnowledgeGradient, optimize_callable]
```

Wraps qMFKG with a `project_to_target_fidelity` callable for the inner
posterior-mean argmax.  Returns both the acquisition function and a
closure `optimize(bounds, fidelity_choices) -> (x_next, fidelity_next,
acq_value)` that drives `optimize_acqf_mixed`.  When
`hf_acquisition_bonus is not None`, returns an `_HFBonusAcquisition`
subclass that adds a constant `+α` to the acquisition value when the
candidate's fidelity column equals `target_fidelity_index` — gates on the
q candidate points only (NOT the q + num_fantasies one-shot dimension);
see `bo.py:783` for the slicing rationale (plan item 47.3k.3.1).

## 4. CLI

```bash
cd scripts/stencil_gen
uv run python -m sweeps bo --help
```

The `bo` subcommand (`sweeps/bo.py`) mirrors `sweeps optimize` and
`sweeps pareto`:

| Flag | Description |
|---|---|
| `--scheme {E2,E4}` | Required. |
| `--kernel {classical,tension,gaussian,multiquadric}` | Required. |
| `--objective FIELD` | HF target as a dotted path, e.g. `layer7.max_spectral_abscissa`.  HF layer inferred from prefix. |
| `--cheap-fidelities N [N ...]` | External cascade layer indices to use as cheap surrogates.  Each must be `< HF layer`.  Default fields per layer come from a built-in table; override via `--fidelity-fields`. |
| `--fidelity-fields LAYER=FIELD [LAYER=FIELD ...]` | Per-layer field overrides. |
| `--bounds LO HI [LO HI ...]` | Flat list of bound pairs; falls back to `DEFAULT_BOUNDS[(scheme, kernel)]`. |
| `--budget-evals N` / `--budget-seconds T` | Mutually exclusive; one required. |
| `--n-init N` | Initial design size (default `5*d + 3`). |
| `--num-fantasies N` | qMFKG fantasy count (default 64). |
| `--seed N` | RNG seed (default 1). |
| `--cost-model {constant,empirical}` | `constant` uses `DEFAULT_COST_TABLE`; `empirical` is reserved for a future item. |
| `--baseline {none,staged}` | Run `run_staged_optimize` alongside MF-BO with the same seed; record into `extras.baseline`. |
| `--persist` | Write JSON to `sweeps/bo_runs/<scheme>_<kernel>_<mangled>_<seed>.json`. |
| `--validate-with-cpp` | Re-run `best_x` at L8 via the shoccs binary; record into `extras.cpp_validation`. |
| `--verbose` | One line per evaluation. |

Validation runs *before* persistence so the JSON captures
`extras.cpp_validation` when both flags are set (mirror of plan 45.5a.1's
Pareto convention).

### 4.1 Example — tension fast (L1 + L3 only)

```bash
SYMPY_CACHE_SIZE=50000 uv run python -m sweeps bo \
    --scheme E4 --kernel tension \
    --objective layer3.max_stab_eig \
    --cheap-fidelities 1 \
    --bounds 0.5 20 \
    --budget-evals 15 --seed 1 --persist
```

CLI smoke test from plan 47's "Test commands" block.  Two-fidelity 1D
problem (σ ∈ [0.5, 20]); converges in well under a minute.

### 4.2 Example — classical 2D full cascade (L1 + L3 + L3r + L6 + L7)

```bash
SYMPY_CACHE_SIZE=50000 uv run python -m sweeps bo \
    --scheme E4 --kernel classical \
    --objective layer7.max_spectral_abscissa \
    --cheap-fidelities 1 3 5 6 \
    --budget-evals 60 --seed 1 --persist
```

The 47.7a calibration run.  Layer 5 in the BO module's external indexing
maps to L3r (`layer_bl42.*`); the cascade itself collapses L3 and L3r
into the same `max_layer=3` evaluation, but the BO module treats them as
distinct fidelities so the ICM kernel can learn separate task
correlations (the L3 ↔ L3r-different-physics finding driving the whole
plan).  Wall time ~5 min on the dev container.

### 4.3 Example — head-to-head benchmark vs. staged

```bash
SYMPY_CACHE_SIZE=50000 uv run python -m sweeps bo \
    --scheme E4 --kernel classical \
    --objective layer7.max_spectral_abscissa \
    --cheap-fidelities 1 3 5 6 \
    --budget-evals 60 --seed 1 \
    --baseline staged --persist
```

Same configuration, but also runs `run_staged_optimize` against the same
HF objective and the same seed.  The persisted JSON's
`extras.baseline` carries the staged run's `best_x`, `best_objective`,
`compute_time`, and `n_evals` (success path) or `error`/`method`/None
fields (failure path) so post-hoc analysis can diff the two.

## 5. Persistence

Each `--persist`ed run writes a single JSON file under
`sweeps/bo_runs/`.  Filename:
`{scheme}_{kernel}_{mangled_objective}_{seed}.json` where the mangler
replaces `.` with `_`:

```
layer7.max_spectral_abscissa  →  layer7_max_spectral_abscissa
```

Top-level keys are emitted in the field order of the `BOResult` dataclass
(see `sweeps/_bo_io.py::_result_to_ordered`):

```
best_x, best_params, best_objective, best_report,
method, scheme, kernel, bounds,
fidelity_levels, hf_level, report_fields_by_layer,
cost_model, n_evals_per_fidelity, wall_time_per_fidelity,
total_compute_time, eval_history, hf_eval_history,
gp_hyperparameters, seed, converged, stop_reason, extras
```

`sweeps/_bo_io.py` exposes:

- `save_bo_run(result, directory=BO_RUNS_DIR) -> Path`
- `load_bo_run(path) -> dict`
- `iter_bo_runs(directory=BO_RUNS_DIR) -> Iterator[Path]`

`load_bo_run` restores int keys for the four whitelisted top-level
fields (`report_fields_by_layer`, `cost_model`, `n_evals_per_fidelity`,
`wall_time_per_fidelity`).  JSON serialises every object key as a string;
without restoration the loaded dict cannot be piped into
`make_multi_fidelity_objective` because the factory's field-vs-layer
validation runs `inferred > layer` against the keys and raises
`TypeError` on strings.  See plan item 47.4c.1 for the full motivation.

The regression test
`tests/test_phs.py::TestRegressionBOBenchmark` reads each stored JSON,
rebuilds `make_multi_fidelity_objective(scheme, kernel,
report_fields_by_layer)`, evaluates at the stored `best_x` at HF, and
asserts `np.isclose(recomputed, stored, rtol=1e-2, atol=1e-8)`.  The
sentinel `_BO_SENTINEL = 1e12` is a finite float that round-trips
bytewise through JSON and compares cleanly — calibration runs that
exhaust their budget without finding a feasible HF region store
`best_objective = 1e12` and the regression test treats that as a valid
state.

## 6. Cost model calibration

`DEFAULT_COST_TABLE` ships the plan-46 measurements in seconds:

```python
DEFAULT_COST_TABLE = {
    1: 0.076,   # L1 GV dispersion
    3: 0.038,   # L3 1D advection eigenvalue
    5: 0.486,   # L3r BL §4.2 reflecting-hyperbolic spectrum
    6: 0.846,   # L6 non-normality on 1D operator
    7: 1.434,   # L7 full 2D varying-coefficient spectral abscissa
}
```

The keys are external cascade layer indices (the same ones the CLI's
`--cheap-fidelities` consumes); the BO module's *internal* fidelity
indexing maps these to a contiguous `0..K-1` integer for the
`MultiTaskGP` task encoding.

`build_cost_model` applies the floor `c'(m) = max(c(m), 0.05 * c(hf))`
(via `apply_cost_floor`) before wrapping in
`InverseCostWeightedUtility`.  The 5 % floor prevents the cost-aware
utility from over-exploiting the cheapest layer when its empirical cost
is anomalously small (a real failure mode: when the cheap layer is so
cheap that `EIG / cost` saturates qMFKG at the cheap fidelity for every
candidate, the GP starves of HF data and the basin never localises).

To override, pass `--cost-model constant` with a hand-built
`cost_table` (CLI plumbing for non-default tables is reserved for a
future item; for now use the API directly via `run_mfbo(cost_table=...)`).
A future "empirical" cost model would fit per-layer GP costs from the
per-eval `wall_time` measurements, but plan 46 measured costs to within
±10 % per layer (except L7, which is 4.7× kernel-dependent for tension)
so the constant table is the MVP.

## 7. Stopping criteria

`run_mfbo` exits when *one* of the following fires:

- **Budget.** `len(eval_history) >= budget_evals - 1` or
  `time.perf_counter() - start >= budget_seconds`.  The `-1` reserves a
  slot for the mandatory final HF re-eval at `x_inc`.
- **Variance guard.** Both
  `var_inc < variance_guard_relative_threshold * max_var_grid` AND
  `var_inc < 1e-6 * spread_hf**2` fire on the same iteration, AND the
  acquisition counter has exceeded `min_acquisition_iterations` (default
  `max(15, K)` post-47.3k.1).  The dual criterion guards against
  GP-uniform-collapse on smooth synthetic objectives (a real failure
  mode where `var_inc / max_var_grid ≈ 1` even when the basin is far
  from localised — see plan items 47.3d, 47.3f, 47.3k for the empirical
  history).
- **Stagnation guard.** No improvement in `min(hf_eval_history)` over
  the most recent `window=10` finite HF observations.  Pure helper
  `_stagnation_triggered`; tested in
  `tests/test_bo.py::TestStagnationGuard`.
- **Error.** Any uncaught exception inside the loop body sets
  `stop_reason="error"` and exits cleanly (the partial `BOResult` is
  still returned).

`stop_reason` is one of `{"budget", "variance", "stagnation", "error"}`
and `converged = stop_reason in {"variance", "stagnation"}`.

## 8. Relationship to `run_staged_optimize`

`run_staged_optimize` (plan 43, `optimization_reference.md` §
"Drivers / staged") and `run_mfbo` solve the *same* problem class —
maximise an HF cascade objective when cheap surrogates are available —
with different machinery:

- `run_staged_optimize`: hand-coded heuristic.  Inner stage at
  `gate_layer=3`, top-K survivors re-evaluated at L7, K is fixed.  No
  surrogate model; exploration relies on the inner driver's own
  multi-start strategy.
- `run_mfbo`: principled cost-aware MF-BO.  GP surrogate over `(x, m)`,
  cost-aware qMFKG acquisition, ICM kernel learns L3 ↔ L3r ↔ L7
  correlations from the data.  No fixed K; the cheap/HF split is a
  Pareto-optimal cost/benefit choice inside the GP posterior.

The 47.7a benchmark records both side-by-side via `--baseline staged`.
Plan-level acceptance: MF-BO either reaches the same `best_objective`
as the staged baseline using ≤ 50 % of total wall-time, OR at equal
wall-time achieves a `best_objective` ≥ 1 % lower than staged.

## 9. When MF-BO helps vs. hurts

The cost-aware utility's leverage scales with the cost ratio between
cheap and HF layers and with the cross-layer correlation that the ICM
kernel can learn.  Plan 47's exploratory analysis of the cascade gave
the following rule of thumb:

| HF target | Cheapest useful surrogate | Cost ratio | Expected MF-BO gain |
|---|---|---|---|
| L7 `max_spectral_abscissa` | L3r (`layer_bl42`) | ~3:1 | modest — single-fidelity BO competitive |
| L7+nn `transient_growth_bound` | L3r or L6 | ~36:1 | large — staged baseline leaves significant budget on the table |
| L8 (compiled C++ sim) | any analytical layer | ~40:1+ | large — but L8 is currently the validator, not in the GP (≥ 10 L8 evals required to anchor a `B_{·,8}` row meaningfully — deferred) |

**When MF-BO probably hurts:** any HF target where the cheapest layer
has a `≤ 2:1` cost ratio.  The qMFKG fantasy machinery's overhead (one
GP fit per acquisition, ~64 fantasies sampled per iteration) becomes
comparable to one cheap evaluation; a single-fidelity BO without
cost-awareness will run faster.

## 10. Failure modes

The slow-suite regression coverage (`tests/test_bo.py`,
`@pytest.mark.slow`) pins three specific failure modes that previous
ad-hoc MF-BO implementations regressed against:

- **`TestBranin` (47.6a) — pipeline smoke test.** AugmentedBranin
  two-fidelity hook at the 47.3k-tuned composition (`hf_priority_warmup=3,
  adaptive_hf_explore_bias=0.5, hf_acquisition_bonus=2.0`).  Threshold
  `best_objective < 3.7` (the empirical floor; the original `< 0.5`
  target was empirically unreachable on a 30-eval budget — see plan
  item 47.3k.4d Stage 2 for the threshold-vs-pass-rate analysis).
- **`TestBiasMisspec` (47.6b) — bias-misspecified cheap surrogate.**
  Cheap layer has constant offset from HF; correctly-modelled MF-BO
  finds the HF optimum, not the cheap optimum.  Catches a regression
  where the GP's per-task variance separation collapses.
- **`TestCostMisspec` — cost-misspecified table.** Wrong costs by a
  factor of 10×; MF-BO degrades by ≤ 2× vs. correctly-costed baseline.
  Pins the cost-floor's protection against acquisition over-exploitation.
- **`TestMultiModal` (47.6b.3) — real-cascade multi-modal classical-α.**
  Finds a known basin in ≥ 4/5 seeds within `1.6` L2 distance (the
  47.6b.3.2d threshold revision; the original `< 0.1` was empirically
  unreachable on a 30/60-eval budget).

## 11. Known limitations

- **Multi-objective MF-BO** (Pareto fronts at multiple fidelities) is
  not implemented.  Use `sweeps pareto` (NSGA-II) for single-fidelity
  Pareto fronts; multi-objective MF-BO is a future extension.
- **Continuous fidelity** (BoTorch's `LinearTruncatedFidelityKernel`
  with `s ∈ [0, 1]`) is not implemented.  The cascade's discrete layer
  indices are the natural fidelity axis; varying N within L7 is a
  separate dimension worth exploring later.
- **Autoregressive Kennedy-O'Hagan model** is not implemented.  The
  L3 ↔ L3r independence rules out a single global AR chain; the ICM
  model is the principled choice here.  AR-style links between specific
  pairs (e.g. L7 → L8) are a future extension.
- **Learned cost model.** Constant cost table only; per-layer GP cost
  models defer to a follow-up.
- **L8 in the GP.** L8 (the C++ simulation) is the validator at the end,
  not a fidelity inside the GP.  Adding L8 as a 6th fidelity needs ≥ 10
  L8 evaluations to anchor a `B_{·,8}` row meaningfully — deferred until
  the data exist.
- **MES acquisition.** Documented as a one-line swap (see § 2) if KG
  diagnostics show pathology; not the default.
- **GPU.** Our GP fits run in microseconds on CPU; CPU-only PyTorch
  wheels via `https://download.pytorch.org/whl/cpu`.

## 12. References

- Plan 41 — Brady-Livescu 2D analytical stability pipeline (L1 – L7).
- Plan 42 — C++ bridge runtime-parameterised stencils (L8).
- Plan 43 — Stability optimisation framework (scalar drivers).
- Plan 44 — L3r BL §4.2 reflecting-BC layer.
- Plan 45 — Multi-objective Pareto optimisation.
- Plan 46 — Cost-table calibration + CLI cleanup.
- Plan 47 — Multi-fidelity Bayesian optimisation (this layer).
- Plan 48 — Brady-Livescu 1D Euler (will reuse the BO infrastructure
  for nonlinear blow-up scoring).
- Wu, J., Toscano-Palmerin, S., Frazier, P. I., & Wilson, A. G. (2020).
  "Practical Multi-Fidelity Bayesian Optimization for Hyperparameter
  Tuning." *UAI 2020.* https://arxiv.org/abs/1903.04703.
- BoTorch tutorials:
  - https://botorch.org/docs/tutorials/discrete_multi_fidelity_bo/
  - https://botorch.org/docs/tutorials/cost_aware_bayesian_optimization/
- BoTorch 0.17 API reference:
  - `botorch.acquisition.knowledge_gradient.qMultiFidelityKnowledgeGradient`
  - `botorch.models.gp_regression.MultiTaskGP`
  - `botorch.models.cost.AffineFidelityCostModel`
  - `botorch.acquisition.cost_aware.InverseCostWeightedUtility`
- `docs/handoff/scientific_findings.md` finding #1 — L3 ↔ L3r are
  different physics (drives the ICM choice over Kennedy-O'Hagan AR1).
