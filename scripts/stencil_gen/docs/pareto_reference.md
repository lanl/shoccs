# Multi-Objective Pareto Optimization Reference

Companion to `optimization_reference.md`.  Where that document describes the
*scalar* optimizer layer (`make_objective`, `run_scipy_local`,
`run_staged_optimize`, …), this one describes the *multi-objective* layer
added in plan 45 — NSGA-II driven Pareto fronts over two or more
Brady-Livescu 2D stability metrics.

## 1. Problem

The scalar optimizers collapse every trade-off onto a single scalar, so
genuinely conflicting metrics disappear.  Plan 44 made the conflict concrete:

| Metric | Measures | Tension E4 σ=3 | Classical E4 (BL α) |
|---|---|---|---|
| `layer1.boundary_gv_err` | dispersion quality at the boundary | ~3.6e-2 (excellent) | ~1e-1 |
| `layer_bl42.max_spectral_abscissa` | BL42 reflecting-BC stability | ~0.95 (fails) | ~3e-14 (passes) |
| `layer6.transient_growth_bound` | non-normal transient growth | ~3.3 | basin-dependent |

No single scalar captures "best closure."  The user's physics (acoustics
vs. advection-dominated vs. transient-growth-sensitive) decides.  The
multi-objective formulation

```
minimize  F(x) = [f_1(x), f_2(x), ..., f_m(x)]
subject to  x ∈ [lb, ub]
```

is solved in the sense of *Pareto dominance*: `x` dominates `y` iff
`F_i(x) ≤ F_i(y)` for all `i` and strictly `<` for at least one.  The
*Pareto front* is the set of non-dominated points — no better on one axis
without being worse on another.  NSGA-II (Deb et al. 2002) evolves a
population toward this front via fast non-dominated sort plus
crowding-distance survival; at convergence the population *is* the front.

## 2. Why pymoo

- Mature, stable 0.6 API with NSGA-II, NSGA-III, hypervolume indicator, and
  reference-direction helpers.
- `ElementwiseProblem` composes cleanly with the existing `make_objective`
  pattern — one vector-valued objective wrapping the per-layer extractor.
- Pure-Python numerics; no PyTorch/TensorFlow dependency (deferred to a
  later Bayesian-optimization plan).

## 3. API

All exports live in `stencil_gen/pareto.py`.

### 3.1 `ParetoPoint`

```python
@dataclass(frozen=True)
class ParetoPoint:
    x: np.ndarray          # (n_var,), float64 — optimizer vector
    params: dict           # params_from_vector(kernel, x) — kernel-native
    objectives: np.ndarray # (n_obj,), float64 — aligned with objective_fields
    report: dict           # _report_to_dict(StabilityReport) at x (may be {})
```

One non-dominated member of a Pareto front.  `x` and `params` are redundant
representations: carry whichever is convenient.  `objectives` is aligned
with `ParetoResult.objective_fields`.  `report` captures the full
short-circuit `StabilityReport` at `x` (including `layer_bl42` as of plan
45.6a.1.1); downstream analysis tooling walks this dict without re-running
the cascade.

### 3.2 `ParetoResult`

```python
@dataclass(frozen=True)
class ParetoResult:
    front: tuple[ParetoPoint, ...]
    objective_fields: tuple[str, ...]
    scheme: str
    kernel: str
    bounds: tuple[tuple[float, float], ...]
    method: str            # "NSGA-II"
    pop_size: int
    n_gen: int
    n_evals: int
    seed: int
    compute_time: float
    hv_trace: tuple[float, ...]  # hypervolume per generation
    ref_point: tuple[float, ...] # used for HV; shape (n_obj,)
    extras: dict                 # driver-specific diagnostics
```

`front` holds only finite, non-sentinel members; sentinel rows produced by
the gate (see `_PARETO_SENTINEL` below) are filtered out and counted in
`extras["n_sentinel_filtered"]`.  `hv_trace` has length `n_gen` and is
non-decreasing under NSGA-II's elitism.  `ref_point` dominates every front
member by construction (auto-picked at 1.1 × the max of a small random
feasible probe, clipped to `[1.0, _PARETO_SENTINEL)`).  `extras` also
carries `hv_n_nds` (per-generation non-dominated count) and, when
`--validate-with-cpp` is used, `cpp_validation`.

### 3.3 `make_multi_objective`

```python
def make_multi_objective(
    scheme: str,
    kernel: str,
    report_fields: Sequence[str],
    *,
    gate_layer: int | None = None,
    max_layer: int | None = None,
) -> Callable[[np.ndarray], np.ndarray]
```

Vector-valued analogue of `make_objective`.  Runs `brady2d_stability_score`
in short-circuit mode up to `max_layer` and extracts each dotted path via
`extract_field`.  Returns a length-`len(report_fields)` vector per call.

On any gate trip, parameter-vector shape mismatch, or exception from
`brady2d_stability_score`, the closure returns
`np.full(n_obj, _PARETO_SENTINEL)` with `_PARETO_SENTINEL = 1e12` — finite,
because pymoo's hypervolume indicator and `ftol` termination both reject
`+inf`.  Downstream NSGA-II filtering drops sentinel rows from the reported
front.

Requires `len(report_fields) >= 2`; raises `ValueError` otherwise.  Defaults
`max_layer = max(_infer_max_layer(f) for f in report_fields)` and
`gate_layer = max(max_layer - 1, 0)`, consistent with the 45.0b auto-infer
shared by the scalar `make_objective`.

### 3.4 `run_nsga2`

```python
def run_nsga2(
    scheme: str,
    kernel: str,
    report_fields: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    *,
    pop_size: int = 40,
    n_gen: int = 50,
    seed: int = 1,
    ref_point: Sequence[float] | None = None,
    gate_layer: int | None = None,
    max_layer: int | None = None,
    verbose: bool = False,
    objective: Callable[[np.ndarray], np.ndarray] | None = None,
) -> ParetoResult
```

Builds a private `ElementwiseProblem` whose `_evaluate` calls the closure
produced by `make_multi_objective` (unless `objective=` overrides it — a
test hook so unit tests can plug synthetic analytic problems in without
importing the Brady2D pipeline), constructs `NSGA2(pop_size=pop_size)` with
pymoo defaults (SBX crossover, polynomial mutation), and runs
`minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, callback=...)`.
After minimization, `res.F` is split into finite and sentinel rows; only
finite rows populate `ParetoResult.front`, paired with a
`_report_to_dict(brady2d_stability_score(...))` rebuild for each surviving
`x`.

Raises `ValueError` on `len(report_fields) < 2`, empty `bounds`, or a
`ref_point` with mismatched shape.

### 3.5 `_HVCallback`

Private callback attached to the NSGA-II algorithm.  On each generation it
pulls the current non-dominated set (`algorithm.opt`, not the full
population history — see pymoo's docs on `Algorithm.opt`), filters rows
whose objectives are non-finite or `≥ ref_point`, and records the
hypervolume under the configured reference point.  The resulting
`hv_trace` and `hv_n_nds` sequences end up in `ParetoResult.hv_trace` and
`ParetoResult.extras["hv_n_nds"]`.

## 4. CLI

```bash
cd scripts/stencil_gen
uv run python -m sweeps pareto --help
```

The `pareto` subcommand (`sweeps/pareto.py`) mirrors `sweeps optimize`:

| Flag | Description |
|---|---|
| `--scheme {E2,E4}` | Required. |
| `--kernel {classical,tension,gaussian,multiquadric}` | Required. |
| `--objectives FIELD [FIELD ...]` | Two or more dotted paths into `StabilityReport`. |
| `--bounds LO HI [LO HI ...]` | Flat list of bound pairs; falls back to `DEFAULT_BOUNDS[(scheme,kernel)]`. |
| `--pop-size N` | NSGA-II population. Default 40. |
| `--n-gen N` | NSGA-II generations. Default 50. |
| `--seed N` | Algorithm seed. Default 1. |
| `--ref-point V [V ...]` | Override the hypervolume reference point (else auto-picked). |
| `--gate-layer N` | Override the feasibility gate depth (else `max_layer - 1`). |
| `--max-layer N` | Override pipeline depth (else `max(infer_max_layer(f))`). |
| `--persist` | Write JSON to `sweeps/pareto_fronts/<scheme>_<kernel>_<mangled>.json`. |
| `--validate-with-cpp` | Re-run up to 10 front members at L8 via the shoccs binary. |
| `--verbose` | Forward to pymoo's `minimize(verbose=True)`. |

The driver prints a summary (front size, final hypervolume, top-5 members
ordered by each objective) after the run, validates (if requested), then
persists (if requested).  Validation runs *before* persistence so the JSON
captures `extras["cpp_validation"]` when both flags are set (plan 45.5a.1).

### 4.1 Example — classical-α E4, 2D Pareto front

```bash
SYMPY_CACHE_SIZE=50000 uv run python -m sweeps pareto \
    --scheme E4 --kernel classical \
    --objectives layer1.boundary_gv_err layer_bl42.max_spectral_abscissa \
    --bounds -2 2 0.05 2 \
    --pop-size 40 --n-gen 30 --seed 1 --persist
```

This is the 45.6a.1 calibration run.  Produces a ~19-member front at
`sweeps/pareto_fronts/E4_classical_layer1_boundary_gv_err__layer_bl42_max_spectral_abscissa.json`
straddling the 1e-10 BL42 threshold: at one end a
`(gv_err ≈ 0.047, BL42 ≈ 7e-15)` stability-focused closure near
`α ≈ (-1.54, 0.31)`; at the other end a `(gv_err ≈ 2.9e-3, BL42 ≈ 0.42)`
dispersion-focused closure near `α ≈ (-0.26, 0.05)`.  Wall time ~6 min on
the dev container.

### 4.2 Example — tension E4, 1D Pareto front

```bash
SYMPY_CACHE_SIZE=50000 uv run python -m sweeps pareto \
    --scheme E4 --kernel tension \
    --objectives layer1.boundary_gv_err layer_bl42.max_spectral_abscissa \
    --bounds 0.5 20 \
    --pop-size 20 --n-gen 20 --seed 1 --persist
```

The 45.6a.2 calibration run.  Both objectives are monotone non-decreasing
in σ over `[0.5, 20]`, so the Pareto front collapses to a single point at
the lower bound (σ = 0.5, `gv_err ≈ 2.2e-2`, `BL42 ≈ 0.65`).  This is the
correct mathematical answer, not an optimizer failure — tension exposes the
same qualitative story as classical (cleaner `gv_err`, worse BL42) but on a
1D parameter space the trade-off lives on a monotone curve and Pareto
dominance picks a single winner.

### 4.3 Example — classical-α E4, 3-objective run

```bash
SYMPY_CACHE_SIZE=50000 uv run python -m sweeps pareto \
    --scheme E4 --kernel classical \
    --objectives layer1.boundary_gv_err \
                 layer_bl42.max_spectral_abscissa \
                 layer6.transient_growth_bound \
    --bounds -2 2 0.05 2 \
    --pop-size 60 --n-gen 40 --seed 1 \
    --validate-with-cpp --persist
```

Exploratory 3D front.  `max_layer` auto-infers to 6 and `gate_layer` to 5,
so L1/L2/L3/L3r/L4/L5 must all pass before the point contributes a finite
triple.  Front size scales superlinearly with `n_obj`; pop_size=60 is the
recommended minimum for three objectives.

## 5. Persistence

Each `--persist`ed run writes a single JSON file under
`sweeps/pareto_fronts/`.  The filename is
`{scheme}_{kernel}_{mangled_objectives}.json`, where the mangler replaces
`.` with `_` inside each field and joins fields with `__`:

```
layer1.boundary_gv_err, layer_bl42.max_spectral_abscissa
  → layer1_boundary_gv_err__layer_bl42_max_spectral_abscissa
```

Top-level JSON keys are emitted in a fixed order (see
`sweeps/_pareto_io.py::_result_to_ordered`):

```
scheme, kernel, method, objective_fields, bounds,
pop_size, n_gen, n_evals, seed, compute_time,
ref_point, hv_trace, front, extras
```

Each `front[i]` carries `(x, params, objectives, report)`.  The `report`
dict is the same shape produced by `_report_to_dict` in
`stencil_gen/optimizer.py`; since plan 45.6a.1.1 it includes `layer_bl42`
alongside the existing `layer1`–`layer8` entries, so a persisted BL42
objective carries the full per-layer diagnostic context.

`sweeps/_pareto_io.py` exposes:

- `save_pareto_front(result, directory=PARETO_FRONTS_DIR) -> Path`
- `load_pareto_front(path) -> dict`
- `iter_pareto_fronts(directory=PARETO_FRONTS_DIR) -> Iterator[Path]`

The load helper returns a raw dict rather than a reconstructed
`ParetoResult`; the regression test in
`tests/test_phs.py::TestRegressionBrady2DPareto` reads that dict, rebuilds
`make_multi_objective(scheme, kernel, objective_fields)`, evaluates at
each stored `x`, and asserts recomputed objectives match stored values
within 1% relative tolerance (which holds bit-exact under the deterministic
ARPACK regime of plan 45.6b.1).

## 6. Reference-point selection

Hypervolume is undefined unless the reference point dominates every front
member.  `run_nsga2` auto-picks `ref_point` by:

1. Drawing 20 uniform-random samples inside `bounds` with a seeded
   `numpy.random.Generator`.
2. Dropping sentinel rows (`y[i] >= _PARETO_SENTINEL` for any `i`).
3. Taking `1.1 * max(finite_rows, axis=0)`.
4. Clipping the result to `[1.0, _PARETO_SENTINEL)` so pymoo's HV stays
   well-defined even for extremely small objective values (e.g.
   `layer_bl42.max_spectral_abscissa ~ 1e-15`).

If the probe phase finds no feasible sample, `ref_point` falls back to the
all-ones vector.  The resulting hypervolume will be zero, but `ParetoResult`
is still total — callers key off `len(front) > 0`, not HV.

Override with `--ref-point` or `run_nsga2(..., ref_point=...)` when you
want the indicator pinned to a known, problem-specific cap — e.g. when
comparing two runs that must share a normalization.

## 7. Cascade integration — `gate_layer` / `max_layer` auto-infer

Both `make_objective` (scalar, plan 45.0b) and `make_multi_objective`
(vector, plan 45.1b) share the same `gate_layer` convention:

- `max_layer` defaults to `max(_infer_max_layer(f) for f in fields)`.
- `gate_layer` defaults to `max(max_layer - 1, 0)`.

So an objective living "in" layer N gates at layer N-1.  For the vector
case this means a mixed objective list like
`[layer1.boundary_gv_err, layer_bl42.max_spectral_abscissa]` runs the
cascade to layer 3 (bl42 lives there), gating at layer 2 — precisely the
depth needed to populate both objectives while still rejecting
infeasible-at-layer-2 points early.

The `_PARETO_SENTINEL` replaces `+inf` on any gate trip: hypervolume and
NSGA-II's `ftol` termination both break on `inf`, but finite `1e12` keeps
the indicator well-defined and lets the evolutionary dynamics migrate
away from infeasible basins.

## 8. Relationship to `gv-stability-pareto`

Two CLI subcommands exist with "pareto" in the name — they are
complementary, not alternatives:

- `python -m sweeps pareto` (plan 45; this document): *optimizer*.
  NSGA-II evolves a population toward the Pareto front of a user-picked
  pair/triple of `StabilityReport` fields.  Output is a per-run JSON with
  `(x, params, objectives, report)` tuples.
- `python -m sweeps gv-stability-pareto` (plan 43 era): *read-only 1D scan
  with dominance filter*.  Evaluates a single kernel's 1D parameter sweep,
  applies a post-hoc non-domination filter over a fixed pair of
  scientifically interesting metrics, and prints a table.  No RNG, no
  population dynamics.

The 1D scan is preserved as a research and documentation aid; when you
need an *optimizer* (2D+ parameter space, or genuinely global exploration),
reach for the NSGA-II driver.

## 9. Known limitations

- **NSGA-III.** 4+ objectives are supported by
  `pymoo.algorithms.moo.nsga3.NSGA3(ref_dirs=...)`; this driver only ships
  NSGA-II.  Wrap the driver if needed — the `_StabilityProblem` inner class
  is algorithm-agnostic.
- **No weighted scalarization CLI.**  If you need a single weighted scalar
  (e.g. `--objective "0.5*a + 0.5*b"`), use the scalar `make_objective`
  with a lambda wrapper or promote it to plan 45.x.
- **No multi-fidelity Bayesian optimization.**  Deferred to a later plan.
- **Kernel scope matches plan 43.** `tension-penalty` and `mixed-ε` are
  out of scope for the same reason: `brady2d_stability_score` does not
  route those kernels through the full cascade.
- **Plots are not produced.** Fronts persist as JSON; visualization lives
  downstream (notebook consumers, dashboard scripts).

## 10. References

- Plan 41 — Brady-Livescu 2D analytical stability pipeline (L1–L7).
- Plan 42 — C++ bridge runtime-parameterized stencils (L8).
- Plan 43 — Stability optimization framework (scalar drivers).
- Plan 44 — L3r BL42 reflecting-BC layer (the objective the Pareto driver
  most often wants).
- Plan 45 — Multi-objective Pareto optimization (this layer).
- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). "A fast and
  elitist multiobjective genetic algorithm: NSGA-II." *IEEE Transactions on
  Evolutionary Computation*, 6(2), 182-197.
- pymoo 0.6 documentation:
  - https://pymoo.org/getting_started/part_2.html
  - https://pymoo.org/algorithms/moo/nsga2.html
  - https://pymoo.org/misc/indicators.html
