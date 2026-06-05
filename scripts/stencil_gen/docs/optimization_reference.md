# Stability Optimization Framework Reference

Companion to `brady2d_stability_reference.md`. Where that document describes
the *scoring* pipeline (L1–L8), this one describes the *optimizer* layer
built on top of it in plan 43 — how to drive `brady2d_stability_score` as an
objective function for `scipy.optimize` and turn parameter exploration into
actual optimization.

## Architecture

The optimizer layers two cascades on top of the analytical stack:

```
                fast (sub-ms)             slower (seconds)          minutes
  candidate -> L1 L2 L3 feasibility -> L4 L5 L6 L7 metrics -> top-k -> L8 C++ sim
               short-circuit on fail     re-evaluate survivors     validate winner
               cheap inner loop          staged outer loop         single final check
```

- **Inner stage (cheap):** `f(x) = brady2d_stability_score(..., gate_layer=3,
  max_layer=3).extract(field)` — returns `+inf` on gate failure, otherwise a
  finite scalar. Tens of milliseconds per evaluation; thousands of evaluations
  per optimizer run are comfortable.
- **Outer stage (expensive validator):** top-`k` inner survivors are
  re-evaluated at `max_layer=6` or `max_layer=7` (seconds each) and re-ranked
  against the user's real objective.
- **Final validation (optional):** L8 runs the compiled C++ solver on the
  final winner (minutes). The analytical verdict stands; L8 disagreement is
  diagnostic, not gating.

## Parameter spaces in scope

| Family | Scheme | Dim | `DEFAULT_BOUNDS` | Constraints |
|---|---|---|---|---|
| Tension | E2_1, E4_1 | 1 | σ ∈ [0.5, 20] | analytical stability only |
| Gaussian | E2_1, E4_1 | 1 | ε ∈ [0.1, 5] | analytical stability only |
| Multiquadric | E2_1, E4_1 | 1 | ε ∈ [0.1, 5] | analytical stability only |
| Classical-α | E4_1 | 2 | α₀ ∈ [-2, 2], α₁ ∈ [0.05, 2] | L8 cut-cell floor α₁ ≥ 197/288 reported as a diagnostic flag, not enforced (see plan 43.9a) |

Tension-penalty, mixed-ε, E2 classical-α, and E6/E8 families are explicitly
out of scope — see plan 43's "What this plan does NOT do" section.

## API reference

All exports live in `stencil_gen/optimizer.py`.

### `OptimizeResult`

Frozen dataclass returned by every driver.

| Field | Type | Description |
|-------|------|-------------|
| `best_params` | `dict` | Kernel-specific params (e.g. `{"sigma": 1.64}` or `{"alpha": [α₀, α₁]}`) |
| `best_x` | `np.ndarray` | Flat parameter vector at the optimum |
| `best_objective` | `float` | Value of the objective at `best_x`; `+inf` if infeasible |
| `best_report` | `dict` | Serialized `StabilityReport` at the optimum (empty when infeasible) |
| `method` | `str` | `"Nelder-Mead"`, `"COBYQA"`, `"SHGO"`, `"DE"`, `"multi-start"`, `"staged"` |
| `converged` | `bool` | Scipy `success` AND finite `best_objective` |
| `n_evals` | `int` | Total objective evaluations |
| `compute_time` | `float` | Wall-clock seconds |
| `history` | `list[(x, f)]` | Every evaluation (captured by a recorder wrapper, not scipy's per-iteration callback) |
| `extras` | `dict` | Driver-specific diagnostics (see per-driver sections) |

### `DEFAULT_BOUNDS`

Dict mapping `(scheme, kernel)` tuples to bound lists. E4 classical-α
bounds were widened in plan 43.9a to admit the Brady-Livescu analytical
feasible region (α₁ ≈ 0.162 sits below the C++ cut-cell floor 197/288).
The Python analytical pipeline L1–L7 does not enforce that cut-cell
constraint; plan 43.10 records a `cpp_cutcell_violates_197_288` flag at L8
diagnosis instead.

### Primitives

```python
params_from_vector(kernel: str, x: np.ndarray) -> dict
vector_from_params(kernel: str, params: dict) -> np.ndarray
```

Round-trip between flat optimizer vectors and the kernel-specific params
dicts `brady2d_stability_score` consumes.

- `"tension"` : `[σ]` ↔ `{"sigma": σ}`
- `"gaussian"` / `"multiquadric"` : `[ε]` ↔ `{"epsilon": ε}`
- `"classical"` : `[α₀, α₁]` ↔ `{"alpha": [α₀, α₁]}`

```python
extract_field(report: StabilityReport, dotted_path: str) -> float
```

Dotted-path lookup into a `StabilityReport`. First segment resolves via
`operator.attrgetter`; remaining segments walk dicts by key and dataclasses
by attribute. Returns `float("inf")` if any segment is missing — so a
layer that was not run never crashes the optimizer. Examples:
`"layer1.boundary_gv_err"`, `"layer3.max_stab_eig"`,
`"layer6.transient_growth_bound"`, `"kreiss.witness_sigma_min"`.

### `make_objective`

```python
def make_objective(
    scheme: str,
    kernel: str,
    report_field: str,
    *,
    gate_layer: int = 3,
    max_layer: int | None = None,
) -> Callable[[np.ndarray], float]
```

Builds the feasibility-gated objective: runs `brady2d_stability_score` in
short-circuit mode up to `max_layer`, returns `+inf` if any layer at or
before `gate_layer` failed, otherwise `extract_field(report,
report_field)`. Any exception (e.g. ill-conditioned RBF systems at
extreme parameters) is caught and collapsed to `+inf`.

`max_layer` defaults to the layer implied by `report_field` (`layer6.*` →
6, `kreiss.*` → 2). Raises `ValueError` if the resolved `max_layer` is
less than `gate_layer` — the optimizer cannot gate on layers it never
runs.

### Drivers

```python
run_scipy_local(f, x0, bounds, *, method="Nelder-Mead", max_evals=200, tol=1e-6)
```

Wraps `scipy.optimize.minimize`. Methods: `"Nelder-Mead"` (default,
adaptive simplex) and `"COBYQA"` (derivative-free trust region, scipy ≥
1.14 — an availability probe runs at import; if the probe fails, COBYQA
raises `RuntimeError` with the scipy version in the message).

```python
multi_start_optimize(f, bounds, n_restarts=10, *, method="Nelder-Mead",
                    seed=0, max_evals=200, tol=1e-6)
```

Sobol-seeded multi-start wrapper. Uses `scipy.stats.qmc.Sobol(d=len(bounds),
seed=seed)` to generate `n_restarts` starting points scaled to `bounds`
via `qmc.scale`, runs `run_scipy_local` from each, and returns the restart
with the smallest finite `best_objective`. If every restart is
infeasible, returns the last restart with `converged=False`. Extras
include `inner_method`, `n_restarts`, `seed`, `n_feasible_restarts`.

```python
run_scipy_shgo(f, bounds, *, n=100, iters=3)
```

Wraps `scipy.optimize.shgo`. Deterministic simplicial-homology global
optimization with a Nelder-Mead local polish. Extras:
`local_minima=[(x, f), ...]`, `n_local_minima`. Handles the
fully-infeasible case by returning a non-finite, non-converged record
with `best_x` set to the bound midpoint (so callers never see the scipy
`x=None` corner case).

```python
run_scipy_de(f, bounds, *, popsize=15, maxiter=100, seed=0, strategy="best1bin")
```

Wraps `scipy.optimize.differential_evolution` with `tol=1e-7`,
`init="sobol"`, `polish=True` (scipy's polish stage is L-BFGS-B for
unconstrained bounded problems). Extras: `popsize`, `maxiter`, `seed`,
`strategy`, `scipy_message`. Scipy DE's population-convergence tolerance
can leave `result.success=False` even after the polish pass has pinned
the minimum — callers should key off `np.isfinite(best_objective)`, not
`converged`.

```python
run_staged_optimize(
    scheme, kernel, report_field, bounds,
    *,
    inner_gate=3, inner_max_layer=3, validator_max_layer=6,
    top_k=5, method="Nelder-Mead", n_restarts=20,
    seed=0, max_evals=200, tol=1e-6,
)
```

The full cascade. Stage 1 runs a cheap inner multi-start at
`inner_max_layer`. Stage 2 takes the top `top_k` distinct feasible
candidates (dedup by rounding `x` to 6 decimals), re-evaluates each at
`validator_max_layer`, and re-ranks by `report_field`.

When `report_field` implies a layer deeper than `inner_max_layer` (e.g.
`layer6.transient_growth_bound` with an L3 inner), the inner stage falls
back to `layer3.max_stab_eig` — the canonical Brady-Livescu stability
short-circuit metric — and the original field is used only at the
validator. The inner field is recorded in `extras["inner_field"]` for
transparency.

Extras include:

- `stage` — `"validated"` when the validator re-ordered the winner, else
  `"inner"`.
- `validator_ranking` — `[(x, f_validator), ...]` sorted ascending.
- `inner_method`, `inner_n_restarts`, `inner_seed`,
  `inner_n_feasible_restarts`, `inner_field`, `inner_best_objective`,
  `inner_best_x`, `inner_max_layer`, `validator_max_layer` — inner-stage
  diagnostics preserved for debugging.
- `cpp_cutcell_violates_197_288` — informational flag for E4 classical-α
  only (plan 43.9b-r1); `True` when `best_x[1] < 197/288`. Absent for
  other kernels.

Raises `ValueError` on `inner_max_layer < inner_gate`,
`validator_max_layer < inner_max_layer`, or `top_k < 1`.

If every top-`k` candidate blows up at the validator depth, the driver
falls back to the inner result, wraps it as `method="staged"` with
`stage="inner"` and `converged=False`, and preserves the inner extras so
callers always see the pipeline marker.

## CLI

```bash
cd scripts/stencil_gen
uv run python -m sweeps optimize --help
```

The `optimize` subcommand supports every driver above and exposes method-
specific knobs: `--validator-max-layer`, `--top-k`, `--inner-method`
(staged); `--shgo-n`, `--shgo-iters` (SHGO); `--de-popsize`, `--de-maxiter`
(DE). Bounds default to `DEFAULT_BOUNDS[(scheme, kernel)]` when
`--bounds` is omitted.

### Example — tension E4 local refine

```bash
SYMPY_CACHE_SIZE=50000 uv run python -m sweeps optimize \
    --scheme E4 --kernel tension \
    --objective layer3.max_stab_eig \
    --gate-layer 3 --max-layer 3 --bounds 0.5 20 \
    --method Nelder-Mead --max-evals 40 --n-restarts 3
```

Converges at σ ≈ 1.644, `best_objective = -1.22e-4` in ~6 s.

### Example — classical-α E4 staged, with C++ validation

```bash
SYMPY_CACHE_SIZE=50000 uv run python -m sweeps optimize \
    --scheme E4 --kernel classical \
    --objective layer6.transient_growth_bound \
    --method staged --n-restarts 20 --validator-max-layer 6 \
    --validate-with-cpp --update-known-values
```

Runs the cascade end-to-end, validates the winner against the real shoccs
binary (L8), and persists to `known_values.json["brady2d_optima"]["E4"]["classical"]`.

## Persistence schema

Under `--update-known-values`, results land at
`known_values.json["brady2d_optima"][scheme][kernel][objective]` with these
fields (plan 43.8a/43.8c):

| Field | Description |
|-------|-------------|
| `best_x`, `best_params` | Optimum in both representations |
| `best_objective` | Value of `objective` at the optimum |
| `method` | Driver name (matches `OptimizeResult.method`) |
| `bounds` | Bounds used for the run |
| `gate_layer`, `max_layer` | Objective-evaluation config, so regression can rebuild `make_objective` deterministically |
| `validator_max_layer` | Present only for `method="staged"` (the validator's layer depth — where `best_objective` was actually computed) |
| `n_evals`, `compute_time`, `converged` | Run stats |
| `best_report` | Serialized `StabilityReport` at the optimum |
| `cpp_cutcell_violates_197_288` | Present only when `extras` carried it (E4 classical-α) |
| `cpp_validation` | Present only under `--validate-with-cpp`; `{stable, final_linf, wall_time_s}` from the L8 bridge |

`history` is never persisted.

`TestRegressionBrady2DOptima` in `tests/test_phs.py` reads each stored entry,
rebuilds `make_objective` using the persisted `gate_layer` / `max_layer`
(or `validator_max_layer` for staged entries — that's where `best_objective`
was computed), evaluates at `best_x`, and asserts the recomputed value
matches the stored `best_objective` within 1% relative tolerance.

## Alpha basin survey

`stencil_gen/benchmarks/alpha_basin_survey.py` runs `run_staged_optimize`
across multiple seeds and clusters winners into basins by rounding `best_x`
to `cluster_decimals` decimals. Analog of Brady-Livescu's Table 4
multi-modality study.

```bash
SYMPY_CACHE_SIZE=50000 uv run python -m stencil_gen.brady2d_cli \
    --alpha-basin-survey --n-seeds 20 --n-restarts 20
```

Flags: `--n-seeds`, `--base-seed`, `--n-restarts`, `--survey-max-evals`,
`--json-output`. Returns exit 0 iff at least one seed was feasible.

The returned dict carries a `basins` list (sorted ascending by
`best_objective`) with per-basin `alpha`, `best_objective`,
`n_seeds_in_basin`, `seeds`, and the propagated
`cpp_cutcell_violates_197_288` flag.

## Recipe: How to optimize a new family

1. Add the `(scheme, kernel)` entry to `DEFAULT_BOUNDS` in
   `stencil_gen/optimizer.py` with the parameter-space bounds.
2. Make sure `brady2d_stability_score` already routes the kernel. If it
   does not (as is the case for `tension-penalty` and `mixed-epsilon`),
   the optimizer cannot drive it — see plan 43.1d. Extending every layer
   helper is out of scope for plan 43.
3. Extend `params_from_vector` / `vector_from_params` with the new kernel
   branch and its shape. Add a round-trip test.
4. Optionally add the kernel to `_KERNEL_DIM` in `sweeps/optimize.py` so
   the CLI's dimension check catches bounds/kernel mismatches at parse
   time instead of hiding them behind a silent infeasible run.

## Recipe: How to add a new objective field

`extract_field` walks any dotted path that resolves on a `StabilityReport`,
so adding a new field is usually free:

1. Ensure the metric is populated under one of the `layerN` dicts on
   `StabilityReport` at or below `max_layer`.
2. Pass the dotted path directly as `--objective` or `report_field`. The
   layer prefix is matched by `_LAYER_PREFIX_RE`, so `max_layer` is
   inferred automatically.
3. If the field does not have a `layerN.` prefix, add an entry to
   `_FIELD_LAYER_ALIAS` in `optimizer.py` (e.g. `kreiss.*` → 2) or pass
   `--max-layer` explicitly.

## Known limitations

Documented here so future maintainers do not try to cover them in plan 43.
The plan's "What this plan does NOT do" section is the source of truth.

- **Multi-objective Pareto optimization.** Delivered in plan 45 via
  pymoo NSGA-II — see the "Multi-objective (plan 45)" section below and
  [`pareto_reference.md`](pareto_reference.md). The scalar drivers in this
  reference remain the per-point building block.
- **Multi-fidelity Bayesian optimization.** The staged cascade is the
  manual cheap-inner / expensive-validator pipeline. A principled
  BoTorch-backed multi-fidelity Bayesian optimizer (`python -m sweeps
  bo`) ships in plan 47 — see the "Multi-fidelity (plan 47)" section
  above and [`mfbo_reference.md`](mfbo_reference.md). Both drivers
  remain supported.
- **Brady-Livescu 1D Euler reproduction.** Their 2019 objective requires
  a full nonlinear 1D Euler RK4 solver that this repo does not have.
  Deferred to plan 48.
- **Classical-α E2_1** (fourth-dim space). Second-order stability was
  judged inconsequential; skipped in favor of E4_1.
- **E6 / E8 classical schemes.** No Python derivation pipeline exists.
- **NLopt.** `pip install nlopt` fails in the container; COBYQA provides
  the equivalent derivative-free trust-region method in the scipy stack.
- **Tension-penalty and mixed-ε.** `brady2d_stability_score` does not
  route those kernels; extending every layer helper is disproportionate
  to the optimizer's reach. Use the standalone
  `sweeps/tension_penalty_sweep` and `sweeps/mixed_epsilon_sweep` CLIs.

## Multi-objective (plan 45)

For conflicting-metric trade-offs (e.g. `layer1.boundary_gv_err` against
`layer_bl42.max_spectral_abscissa`) use `python -m sweeps pareto`, which
runs pymoo NSGA-II over a vector-valued objective and writes a Pareto
front as JSON. The scalar drivers documented above are the per-point
building block; the multi-objective factory `make_multi_objective` wraps
them into a length-`n_obj` closure and the driver `run_nsga2` evolves a
population toward the front. Fronts persist per-run under
`sweeps/pareto_fronts/<scheme>_<kernel>_<mangled_objectives>.json`. The
`gate_layer` / `max_layer` auto-infer from the API reference above
(`gate_layer = max(max_layer - 1, 0)` when omitted) applies identically
to `make_multi_objective`, with `max_layer` taken as the deepest layer
across the chosen objective fields. Full details, CLI examples, and the
schema live in [`pareto_reference.md`](pareto_reference.md).

## Multi-fidelity (plan 47)

When the cascade's heterogeneous costs (5+ orders of magnitude between
L1 and L7) make the staged cheap-inner / expensive-validator heuristic
above leave wall-time on the table, use `python -m sweeps bo`, which
fits a BoTorch ICM Gaussian-process surrogate jointly over `(x, m)` and
picks the next `(x, m)` via cost-aware qMFKG. Prefer MF-BO over the
scalar drivers when (a) the HF metric is genuinely expensive (L6+ on a
multi-modal landscape), (b) cheap layers carry usable signal about HF
behaviour but are not refinements of it (e.g. L3 → L3r tests different
physics), or (c) you want a principled cost/benefit tradeoff instead of
a hand-tuned `top_k`. Per-run JSONs persist under
`sweeps/bo_runs/<scheme>_<kernel>_<mangled>_<seed>.json`. Full details,
CLI examples, and the schema live in
[`mfbo_reference.md`](mfbo_reference.md).

## References

- Plan 41 — Brady-Livescu 2D analytical stability pipeline (L1–L7).
- Plan 42 — C++ bridge runtime-parameterized stencils (L8).
- Plan 43 — Stability optimization framework (this layer).
- Plan 45 — Multi-objective Pareto extension
  ([`pareto_reference.md`](pareto_reference.md)).
- Plan 47 — Multi-fidelity Bayesian optimization
  ([`mfbo_reference.md`](mfbo_reference.md)).
- Brady & Livescu 2019 — Table 4 reports 101 E4 schemes discovered from
  random restarts, motivating the multi-seed alpha-basin survey.
