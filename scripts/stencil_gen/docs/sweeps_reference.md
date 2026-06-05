# Sweeps Package Reference

## 1. Package Overview

The `sweeps` package provides standalone parameter-space exploration tools for
PHS/RBF-augmented finite difference stencils. These scripts sweep shape
parameters (epsilon, sigma), penalty weights (gamma), and stencil footprint
sizes (nextra) to discover configurations that yield stable differentiation
matrices.

**Key design principle:** Parameter exploration is separated from regression
testing. Sweeps are research tools that print results and optionally persist
optimal values; regression tests consume those persisted values.

### The Sweep Workflow

```
  1. Run a sweep        -->  explore parameter space, print tables
  2. --update-known-values  -->  write optimal values to known_values.json
  3. Regression tests    -->  tests/test_phs.py loads known_values.json
                              and asserts stability at recorded values
```

This separation means:

- Sweeps can be slow and exploratory (hundreds or thousands of eigenvalue
  evaluations) without slowing down the test suite.
- `known_values.json` acts as a contract between exploration and testing:
  sweeps produce it, tests consume it.
- When stencil derivation code changes, you re-run sweeps to check whether
  the optimal parameters have shifted, then update `known_values.json` if
  needed.

### Package Layout

```
sweeps/
  __init__.py                  # Package docstring
  __main__.py                  # CLI entry point (argparse dispatch)
  _common.py                   # Shared constants, types, helpers
  known_values.json            # Persisted optimal parameter values
  epsilon_sweep.py             # Uniform epsilon sweep (Gaussian/MQ)
  mixed_epsilon_sweep.py       # Per-row epsilon sweep
  tension_sweep.py             # Tension spline sigma sweep
  tension_penalty_sweep.py     # Joint (sigma, gamma) sweep
  footprint_sweep.py           # Stencil footprint (nextra) sweep
  comparison.py                # Multi-method comparison tables
  alpha_extraction.py          # Boundary alpha extraction at optimal epsilon
```


## 2. CLI Usage

All sweeps are invoked through the unified entry point:

```bash
uv run python -m sweeps <subcommand> [options]
```

Running without a subcommand prints help. Each subcommand can also be run as
a standalone module (e.g., `uv run python -m sweeps.epsilon_sweep --scheme E2`).

### 2.1 `epsilon` -- Uniform Epsilon Sweep (Gaussian/Multiquadric)

Sweeps the RBF shape parameter epsilon over a log-spaced range and reports
stability of the differentiation matrix at each grid size. Performs a
coarse sweep followed by a fine sweep around the best coarse value.

```
uv run python -m sweeps epsilon --help
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--scheme` | `E2` or `E4` | *required* | Stencil scheme |
| `--kernel` | `gaussian` or `multiquadric` | `gaussian` | RBF kernel type |
| `--n-values` | comma-separated ints | `20,40,80` | Grid sizes to evaluate |
| `--n-eps` | int | `60` | Number of epsilon sample points in coarse sweep |
| `--update-known-values` | flag | off | Write optimal epsilon to `known_values.json` |

**What it does:**

1. Coarse sweep: evaluates `n_eps` epsilon values log-spaced in [0.01, 10] at
   each grid size.
2. Fine sweep: zooms in +/-1 decade around the coarse-best epsilon at n=40
   with 200 points.
3. Cross-validation: checks the fine-sweep-best epsilon at grid sizes
   [20, 40, 80, 160].
4. Reports the best epsilon, stability eigenvalue, and which grid sizes are
   stable.

**Examples:**

```bash
# Full sweep for E2 Gaussian
uv run python -m sweeps epsilon --scheme E2 --kernel gaussian

# Quick smoke test with fewer samples
uv run python -m sweeps epsilon --scheme E4 --n-eps 10

# Multiquadric kernel, persist results
uv run python -m sweeps epsilon --scheme E2 --kernel multiquadric --update-known-values
```

**Updates to `known_values.json`:** Sets `<scheme_label>.<kernel>.epsilon` and
`<scheme_label>.<kernel>.stable_at`.


### 2.2 `tension` -- Tension Spline Sigma Sweep

Sweeps the tension parameter sigma, including sigma=0 (which recovers the
PHS k=2 limit). Uses a coarse-then-fine strategy like the epsilon sweep.

```
uv run python -m sweeps tension --help
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--scheme` | `E2` or `E4` | *required* | Stencil scheme |
| `--n-values` | comma-separated ints | `20,40,80` | Grid sizes to evaluate |
| `--n-sigma` | int | `61` | Number of sigma sample points in coarse sweep |
| `--sigma-max` | float | `20.0` | Upper bound for sigma range |
| `--update-known-values` | flag | off | Write optimal sigma to `known_values.json` |

**What it does:**

1. Coarse sweep: sigma=0 plus `n_sigma` values log-spaced in [0.01, sigma_max].
2. Fine sweep at n=40: zooms in around the best coarse sigma with 200 points.
3. Cross-validation at grid sizes [20, 40, 80, 160].

**Examples:**

```bash
uv run python -m sweeps tension --scheme E2
uv run python -m sweeps tension --scheme E4 --sigma-max 30.0 --update-known-values
uv run python -m sweeps tension --scheme E2 --n-sigma 10  # quick
```

**Updates to `known_values.json`:** Sets `<scheme_label>.tension.sigma` and
`<scheme_label>.tension.stable_at`.


### 2.3 `tension-penalty` -- Joint (sigma, gamma) Sweep

Sweeps both the tension parameter sigma and the conservation penalty weight
gamma to investigate the stability-conservation trade-off. Runs three phases:
coarse 2D sweep, penalty-effect analysis at the scheme's optimal sigma, and a
fine 2D sweep near the optimal region.

```
uv run python -m sweeps tension-penalty --help
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--scheme` | `E2` or `E4` | *required* | Stencil scheme |
| `--n-sigma` | int | `25` | Number of sigma sample points |
| `--n-gamma` | int | `25` | Number of gamma sample points |
| `--sigma-max` | float | `20.0` | Upper bound for sigma range |
| `--update-known-values` | flag | off | Write optimal (sigma, gamma) to `known_values.json` |

**Three phases:**

1. **Coarse 2D sweep:** sigma x gamma grid. Reports the most stable point,
   baseline (gamma=0), and the stable point with lowest conservation deficit.
2. **Penalty effect:** Fixes sigma at the scheme's known optimal (E2: 6.0,
   E4: 0.0) and sweeps gamma alone, reporting how far gamma can be pushed
   before stability breaks.
3. **Fine 2D sweep:** Narrows the sigma range (E2: [4, 8], E4: [0, 5]) and
   gamma range [0, 100] for higher resolution.

**Examples:**

```bash
uv run python -m sweeps tension-penalty --scheme E4
uv run python -m sweeps tension-penalty --scheme E2 --n-sigma 5 --n-gamma 5  # quick
```

**Updates to `known_values.json`:** Sets
`<scheme_label>.tension_penalty.{sigma, gamma, stable_at}`.


### 2.4 `mixed-epsilon` -- Per-Row Epsilon Sweep

Unlike the uniform epsilon sweep, this explores configurations where each
boundary row has its own shape parameter. Runs four strategies in sequence.

```
uv run python -m sweeps mixed-epsilon --help
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--scheme` | `E2` or `E4` | `E4` | Stencil scheme |
| `--kernel` | `gaussian` or `multiquadric` | `gaussian` | RBF kernel type |
| `--n-eps` | int | `20` | Number of epsilon sample points per dimension |
| `--update-known-values` | flag | off | Write per-row epsilons to `known_values.json` |

**Four strategies:**

1. **Single-epsilon baseline:** Sweep uniform epsilon to establish a reference
   stability eigenvalue.
2. **Two-group sweep:** Split boundary rows into outer (first half) and inner
   (second half), sweep two independent epsilon values on a 2D grid.
3. **Per-row coordinate descent:** Start from uniform epsilon, then optimize
   each row's epsilon independently for 3 passes.
4. **Conservation near-interior** (Gaussian only): Use a polynomial stencil
   (eps -> infinity limit) for the near-interior row (r-1) while sweeping
   epsilon for the remaining boundary rows. Also tries a 2D variant with a
   distinct epsilon for the last row.

**Examples:**

```bash
uv run python -m sweeps mixed-epsilon --scheme E4
uv run python -m sweeps mixed-epsilon --scheme E4 --kernel multiquadric --n-eps 10
```

**Updates to `known_values.json`:** Sets
`<scheme_label>.mixed_<kernel>.{per_row_epsilons, baseline_epsilon}`.


### 2.5 `footprint` -- Stencil Footprint (nextra) Sweep

Sweeps the `nextra` parameter (number of extra boundary rows beyond the
minimum) across sigma values to determine how stencil footprint affects
stability. This sweep is E4-only (hardcoded p=2, q=3, nu=1).

```
uv run python -m sweeps footprint --help
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--n-sigma` | int | `20` | Number of sigma sample points |
| `--n-gamma` | int | `20` | Number of gamma sample points (penalty phase) |
| `--sigma-max` | float | `50.0` | Upper bound for sigma range |
| `--nextra-values` | comma-separated ints | `0,1,2,3` | nextra values to test |
| `--update-known-values` | flag | off | Write footprint stability to `known_values.json` |

**Three phases:**

1. **nextra x sigma sweep:** For each nextra value, sweeps sigma and reports
   stability. Prints a combined table showing all nextra values side by side.
   Also prints a summary with stencil width (t), boundary rows (r), and
   extra DOF count.
2. **nextra x sigma x gamma penalty sweep:** Adds the gamma dimension to find
   whether conservation penalties help at each footprint size.
3. **Grid independence:** Checks sigma=0 (PHS k=2) at grid sizes [20, 40, 80,
   160] for each nextra value.

**Key relationships:**

- Stencil width: `t = p + q + 1 + nextra`
- Boundary rows per side: `r = q + 1 + nextra`
- Extra DOF (free parameters): `r * (p + nextra)`
- Minimum grid size: `n >= 2 * r`

**Examples:**

```bash
uv run python -m sweeps footprint
uv run python -m sweeps footprint --nextra-values 0,1,2 --n-sigma 10  # quick
uv run python -m sweeps footprint --update-known-values
```

**Updates to `known_values.json`:** Replaces the entire `footprint` key with
entries like `E4_nextra0_phs`, `E4_nextra1_phs`, and optionally
`E4_nextra0_tension_3` when a non-zero sigma yields stability.


### 2.6 `comparison` -- Multi-Method Comparison

Compares all RBF/tension methods side-by-side at each grid size.
For each method, reports: stability eigenvalue, spectral radius,
CFL number (RK4), and conservation deficit.

```
uv run python -m sweeps comparison --help
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--scheme` | `E2` or `E4` | both | Scheme to compare (omit for both) |
| `--n-values` | comma-separated ints | `20,40,80` | Grid sizes to evaluate |
| `--update-known-values` | flag | off | Write optimal parameters to `known_values.json` |

**Methods compared:**

1. PHS k=2 (sigma=0)
2. Gaussian at optimal epsilon
3. Multiquadric at optimal epsilon
4. Tension at optimal sigma (gamma=0)
5. Tension + conservation penalty at optimal (sigma, gamma)
6. Mixed-epsilon Gaussian (E4 only) -- per-row coordinate descent

**What it does:**

- Finds optimal parameters via coarse+fine sweeps at n=40.
- Evaluates all methods across all requested grid sizes.
- Prints formatted comparison tables and a stability summary.

**Examples:**

```bash
uv run python -m sweeps comparison --scheme E2
uv run python -m sweeps comparison                     # both E2 and E4
uv run python -m sweeps comparison --update-known-values
```

**Updates to `known_values.json`:** Sets
`<scheme_label>.gaussian.{epsilon, stable_at}` and
`<scheme_label>.multiquadric.{epsilon, stable_at}` for each scheme.


### 2.7 `alpha` -- Boundary Alpha Extraction

For a given scheme, finds the optimal Gaussian epsilon, extracts the implied
boundary stencil alpha parameters in the TEMO parameterization, and compares
them with production values from the C++ solver.

```
uv run python -m sweeps alpha --help
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--scheme` | `E2` or `E4` | *required* | Stencil scheme |

**What it does:**

1. Fine-sweeps Gaussian epsilon in [1.5, 3.5] with 200 points to find the
   most stable value.
2. Extracts the r x t boundary weight matrix at that epsilon.
3. Builds a linear system from the symbolic TEMO boundary matrix and solves
   (least-squares) for the alpha parameters.
4. Verifies that rows 0..r-2 of the TEMO boundary matrix match the RBF
   weights to machine precision.
5. Compares row r-1 (where TEMO enforces conservation but RBF does not).
6. Checks stability across grid sizes [20, 40, 80, 160] for both the direct
   RBF matrix and the TEMO-reconstructed matrix.
7. If production alphas are available (currently only E2), compares
   RBF-extracted, TEMO-reconstructed, and production stability.
8. Reports conservation deficit of the boundary block.

**Note:** This subcommand does not have an `--update-known-values` flag. It is
a diagnostic/analysis tool.

**Examples:**

```bash
uv run python -m sweeps alpha --scheme E2
uv run python -m sweeps alpha --scheme E4
```


### 2.8 `all` -- Run All Sweeps

Runs all sweeps sequentially with a single command. Useful for full
verification after code changes.

```
uv run python -m sweeps all --help
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--quick` | flag | off | Reduced resolution for fast verification |

**Sweeps executed (in order):**

1. Epsilon sweep E2 (Gaussian)
2. Epsilon sweep E4 (Gaussian)
3. Epsilon sweep E2 (Multiquadric)
4. Epsilon sweep E4 (Multiquadric)
5. Mixed epsilon sweep E4
6. Tension sweep E2
7. Tension sweep E4
8. Tension-penalty sweep E2
9. Tension-penalty sweep E4
10. Footprint sweep
11. Comparison (all schemes)
12. Alpha extraction E2

**`--quick` mode** reduces sample counts for fast smoke testing:

| Parameter | Default | `--quick` |
|---|---|---|
| `n_eps` | 60 | 10 |
| `n_sigma` | 61 | 10 |
| `n_gamma` | 25 | 5 |
| `n_values` | 20,40,80 | 20,40 |
| mixed `n_eps` | 20 | 5 |
| footprint `n_sigma` | 20 | 10 |
| footprint `n_gamma` | 20 | 10 |
| tension-penalty `n_sigma` | 25 | 5 |

Reports a summary of passed/failed sweeps at the end.

**Examples:**

```bash
# Full suite (slow, thorough)
uv run python -m sweeps all

# Quick verification (minutes instead of hours)
uv run python -m sweeps all --quick
```


### 2.9 Group velocity objectives

Sweeps use eigenvalue stability (`max Re λ(-D_bc) < STABILITY_TOL`) as the hard
feasibility gate. Group velocity (GV) error is a secondary objective: among the
stable set, lower GV error is better. Eigenvalue stability is never overridden
by GV — this is the *feasible-then-minimize* contract.

| Objective | Role | Hard gate? |
|---|---|---|
| Eigenvalue stability (`max Re λ(-D_bc)`) | Hard constraint | Yes (`< STABILITY_TOL`) |
| Boundary GV error (`boundary_gv_error_max`) | Secondary objective | No — minimize among feasible |
| Conservation deficit | Tertiary objective (`tension-penalty`) | No |
| GKS outgoing boundary modes | Advisory diagnostic | No |

The helpers in `sweeps/gv_objectives.py` wrap the `group_velocity` module:
`interior_gv_error_max`, `interior_cutoff_fraction`, `boundary_gv_error_max`,
`cutcell_gv_min_C`, `gv_score_from_matrix`, and `print_gks_advisory`. See
`docs/group_velocity_reference.md` for the underlying GV math.

#### `--include-gv` flag

Accepted by `tension`, `epsilon`, and `footprint`. The `tension-penalty` sweep
always computes GV (no flag needed) because `eval_point` already builds `D` once
and GV comes free from `gv_score_from_matrix(D)`.

When set, each sweep:

- Evaluates `boundary_gv_error_max` once per swept parameter value (GV is
  grid-independent, so no re-evaluation across grid sizes).
- Prints an added `gv_err` column in the per-grid results table.
- Reports a "Best feasible (by GV error)" line alongside the existing
  stability-optimum summary (widest stable range).
- When combined with `--update-known-values`, additively persists two new keys
  (see schema updates in section 3). The existing stability-optimum parameter
  and `stable_at` list are unchanged.

**Examples:**

```bash
uv run python -m sweeps tension --scheme E2 --include-gv
uv run python -m sweeps epsilon --scheme E2 --kernel gaussian --include-gv
uv run python -m sweeps footprint --include-gv
```

#### `--check-gks` flag

Accepted by `tension` and `epsilon`. After picking the stability-optimum, builds
`D` at the optimum and calls `gks_group_velocity_check(D, xi)`. Prints any
outgoing boundary modes as `WARNING: outgoing boundary mode at xi=…` advisory
lines, or a single `no outgoing boundary modes detected` line when the check is
clean.

**Important:** GKS is advisory only — necessary but not sufficient for
instability. It does NOT alter the optimum or `stable_at`, and is never promoted
to a feasibility gate.

```bash
uv run python -m sweeps tension --scheme E2 --check-gks
uv run python -m sweeps epsilon --scheme E2 --kernel gaussian --check-gks
```

#### `gv-stability-pareto` subcommand

Fine 1D scan that tabulates `(stab_eig, gv_error)` at every grid point for a
chosen scheme + parameter and extracts the Pareto-optimal subset (non-dominated
feasible points). Research/diagnostic only: no `--update-known-values` flag, no
regression test consumption.

```
uv run python -m sweeps gv-stability-pareto --help
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--scheme` | `E2` or `E4` | (required) | Stencil scheme |
| `--param` | `tension`, `gaussian`, or `multiquadric` | (required) | Kernel / swept parameter |
| `--n-points` | int | `61` | Number of parameter sample points |
| `--n` | int | `40` | Grid size to evaluate at (GV is grid-independent) |
| `--param-max` | float | `20.0` | Upper bound for the swept parameter |

Prints two markdown tables: the full sorted grid and the Pareto-optimal points.
A point is Pareto-optimal if no other feasible (`stab_eig < STABILITY_TOL`)
point dominates it under the partial order `(stab_eig ≤, gv_err ≤)` with at
least one strict inequality.

**Example:**

```bash
uv run python -m sweeps gv-stability-pareto --scheme E2 --param tension --n-points 11
```

#### Semantic contract on the additive `gv_error` field

When a sweep persists `gv_error` on its primary entry (e.g., `tension.gv_error`,
`gaussian.gv_error`, `E4_nextra{nx}_tension_{N}.gv_error`,
`tension_penalty.gv_error`), the field stores "the GV error you pay for that
stability choice" — that is, `boundary_gv_error_max` evaluated at the exact
rounded parameter that also sits on the same entry. The `(param, gv_error)`
pair is therefore bit-exact self-consistent at the persisted parameter, so a
regression test can rebuild `D` at `entry[param]` and compare against
`entry["gv_error"]` with a near-zero tolerance.

The parallel `*_gv` entry (`tension_gv`, `{kernel}_gv`, `tension_penalty_gv`,
`footprint.E4_nextra{nx}_tension_gv`) stores the GV-optimal feasible parameter
and its own `gv_error` — a different point from the primary entry whenever the
stability optimum and GV optimum diverge (which is the common case).


## 3. `known_values.json` Schema

This file is the contract between sweep scripts and regression tests. Sweeps
write to it (when `--update-known-values` is passed); tests in
`tests/test_phs.py` read from it.

### Top-Level Structure

```json
{
  "E2_1": { ... },       // Scheme-level entries keyed by label
  "E4_1": { ... },
  "footprint": { ... }   // Cross-scheme footprint data
}
```

### Scheme Entry (e.g., `"E2_1"`)

```json
{
  "params": {
    "p": 1,           // Interior half-width (scheme order = 2p)
    "q": 1,           // Minimum boundary rows minus 1
    "nextra": 1,      // Extra boundary rows beyond minimum
    "nu": 1           // Derivative order
  },
  "tension": {
    "sigma": 6.0,                 // Stability-optimal tension parameter (hard gate)
    "stable_at": [20, 40, 80],   // Grid sizes where stability holds
    "gv_error": 2.647e+00         // (optional) GV error at sigma above; populated by --include-gv
  },
  "tension_gv": {                 // (optional) GV-optimal feasible tension parameter
    "sigma": 20.0,                // GV-optimal sigma (may differ from tension.sigma)
    "gv_error": 2.075e+00,        // boundary_gv_error_max at this sigma
    "stable_at": [20, 40, 80, 160]
  },
  "gaussian": {
    "epsilon": 2.0,              // Stability-optimal Gaussian shape parameter
    "stable_at": [40],
    "gv_error": 8.09e+00          // (optional) GV error at epsilon above
  },
  "gaussian_gv": {                // (optional) GV-optimal feasible epsilon
    "epsilon": 1.778,
    "gv_error": 1.868,
    "stable_at": [20, 40, 80, 160]
  },
  "multiquadric": {
    "epsilon": 1.0,              // Stability-optimal MQ shape parameter
    "stable_at": [40],
    "gv_error": 2.14              // (optional) GV error at epsilon above
  },
  "multiquadric_gv": {            // (optional) GV-optimal feasible epsilon
    "epsilon": 10.0,
    "gv_error": 2.14,
    "stable_at": [20, 40, 80, 160]
  },
  "phs_k2": {
    "stable_at": [20, 40, 80, 160]  // PHS k=2 (no shape parameter)
  },
  "tension_penalty": {
    "sigma": 4.0,                 // Stability-optimal (sigma, gamma) from fine sweep
    "gamma": 0.1,
    "stable_at": [20, 40, 80],
    "gv_error": 2.579e+00         // (optional) GV error at (sigma, gamma) above
  },
  "tension_penalty_gv": {         // (optional) GV-optimal feasible (sigma, gamma)
    "sigma": 20.0,
    "gamma": 0.0,
    "gv_error": 2.075e+00,
    "stable_at": [20, 40, 80, 160]
  },
  "known_unstable": [            // Optional: explicitly unstable configs
    {"kernel": "gaussian", "epsilon": 0.1, "n": 20}
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|---|---|---|
| `params` | object | Scheme parameters: `p` (interior half-width), `q` (boundary parameter), `nextra` (extra rows), `nu` (derivative order) |
| `tension` | object | Best tension spline result. `sigma` is the optimal tension parameter; `stable_at` lists grid sizes with stability eigenvalue below `STABILITY_TOL` |
| `gaussian` | object | Best Gaussian RBF result. `epsilon` is the optimal shape parameter |
| `multiquadric` | object | Best Multiquadric RBF result. `epsilon` is the optimal shape parameter |
| `phs_k2` | object | Polyharmonic spline k=2 (parameter-free). Only has `stable_at` |
| `known_unstable` | array | Configurations explicitly verified as unstable. Used by negative tests |
| `mixed_<kernel>` | object | Per-row epsilon results from `mixed-epsilon` sweep. Contains `per_row_epsilons` (list of floats) and `baseline_epsilon` (float) |
| `tension_penalty` | object | Joint (sigma, gamma) result. Contains `sigma`, `gamma`, `stable_at`, and (when `--include-gv` was run) `gv_error` = GV error at that `(sigma, gamma)` |
| `tension.gv_error`, `gaussian.gv_error`, `multiquadric.gv_error`, `tension_penalty.gv_error` | float | (additive, optional) GV error at the stability-optimum parameter that lives on the same entry. Bit-exact self-consistent: rebuilding `D` at `entry[param]` returns this value. Populated by `--include-gv --update-known-values` |
| `tension_gv`, `gaussian_gv`, `multiquadric_gv` | object | (optional) GV-optimal feasible parameter for the named kernel. Contains the parameter (`sigma` or `epsilon`), `gv_error`, and a cross-grid `stable_at` list. The GV-optimal parameter may differ from the stability-optimal one; both are persisted separately |
| `tension_penalty_gv` | object | (optional) GV-optimal feasible `(sigma, gamma)`. Contains `sigma`, `gamma`, `gv_error`, and a cross-grid `stable_at` list |

### Footprint Entry

```json
{
  "footprint": {
    "E4_nextra0_phs": {
      "nextra": 0,
      "stable_at": [20, 40, 80, 160]
    },
    "E4_nextra0_tension_3": {
      "nextra": 0,
      "sigma": 3.0,
      "stable_at": [40],
      "gv_error": 4.75            // (optional) GV error at sigma above
    },
    "E4_nextra0_tension_gv": {    // (optional) GV-optimal feasible sigma at nextra=0
      "nextra": 0,
      "sigma": 50.0,
      "gv_error": 4.75,
      "stable_at": [20, 40, 80, 160]
    },
    "E4_nextra1_phs": {
      "nextra": 1,
      "stable_at": [40]
    }
  }
}
```

Keys follow the pattern `E4_nextra<N>_<method>[_<param>]`. Each entry records
`nextra`, optionally `sigma`, and the list of grid sizes where the
configuration is stable. When `footprint --include-gv --update-known-values` is
run, the primary `E4_nextra<N>_tension_<K>` entries gain an additive `gv_error`
field (bit-exact self-consistent with the stored `sigma`), and a parallel
`E4_nextra<N>_tension_gv` entry is written for the GV-optimal feasible sigma at
that nextra. The filter `best_stable_gv_sigma > 0` ensures that PHS-baseline
(sigma=0) points never land in a `_tension_gv` key — those are the
`E4_nextra<N>_phs` entries' concern.

### How Sweeps Update `known_values.json`

All sweep scripts follow the same pattern:

1. Call `load_known_values()` to read the current JSON.
2. Set/overwrite the relevant nested key (e.g., `kv["E4_1"]["tension"]`).
3. Call `save_known_values(kv)` to write back with `indent=2` formatting.

The update only happens when `--update-known-values` is explicitly passed.
Without it, sweeps are read-only and only print to stdout.


## 4. Common Helpers (`_common.py`)

### Constants

**`STABILITY_TOL = 1e-10`**

Eigenvalues of `-D_bc` (the boundary-conditioned differentiation matrix with
inflow row removed) below this threshold are considered numerically zero,
meaning the operator is stable. Floating-point eigenvalue solvers typically
return tiny positive real parts (~1e-14) for genuinely stable operators.

**`SCHEME_PARAMS`**

Dictionary mapping scheme names to their defining parameters:

```python
SCHEME_PARAMS = {
    "E2": {"p": 1, "q": 1, "nextra": 1, "nu": 1, "label": "E2_1"},
    "E4": {"p": 2, "q": 3, "nextra": 0, "nu": 1, "label": "E4_1"},
}
```

- `p`: interior stencil half-width (scheme order = 2p)
- `q`: controls the minimum number of boundary rows (r = q + 1 + nextra)
- `nextra`: extra boundary rows beyond the minimum
- `nu`: derivative order (always 1 for first derivative)
- `label`: key used in `known_values.json` (e.g., `"E2_1"`)

**`KNOWN_VALUES_PATH`**

`Path` object pointing to `known_values.json` in the same directory as
`_common.py`.

### Types

**`SweepResult`** (dataclass)

```python
@dataclass
class SweepResult:
    parameter: float       # The swept parameter value
    eigenvalue: float      # Stability eigenvalue at this point
    stable: bool           # Whether eigenvalue < STABILITY_TOL
    n: int | None = None   # Grid size (optional)
    label: str = ""        # Descriptive label (optional)
    extra: dict = field(default_factory=dict)  # Additional data
```

A structured container for a single sweep evaluation point. Not currently
used by all sweep scripts (some use raw tuples), but available for future
use.

### Functions

**`load_known_values() -> dict`**

Reads and returns the contents of `known_values.json`. Returns an empty dict
if the file does not exist.

**`save_known_values(data: dict) -> None`**

Writes `data` to `known_values.json` with `indent=2` formatting and a
trailing newline.

**`print_table(title, headers, rows, *, col_widths=None) -> None`**

Prints a formatted table to stdout with auto-calculated column widths.

- `title`: banner text printed above the table
- `headers`: list of column header strings
- `rows`: list of lists of cell value strings
- `col_widths`: optional manual column widths

**`print_sweep_table(label, results, *, param_label="param") -> None`**

Prints a formatted sweep results table grouped by grid size. Each entry shows
the parameter value, stability eigenvalue, and STABLE/unstable status.
After the per-grid-size tables, prints a summary of the best parameter value
for each grid size.

- `label`: banner title
- `results`: `dict[int, list[tuple[float, float]]]` mapping grid size `n` to
  list of `(parameter_value, stability_eigenvalue)` pairs
- `param_label`: column header for the swept parameter (e.g., `"epsilon"`,
  `"sigma"`)

**`report_stable_ranges(results, *, param_label="param") -> None`**

For each grid size, prints the count and range of stable parameter values.
If no stable values exist, prints the closest-to-stable point.

**`bisect_threshold(f, a, b, threshold, *, tol=1e-4, maxiter=60) -> float`**

Bisection search to find `x` where `f(x)` crosses `threshold` from above.
Assumes `f(a) > threshold` and `f(b) < threshold`. Returns the midpoint
when the interval width drops below `tol`.


## 5. Adding a New Sweep

### Step-by-Step

**1. Create the sweep module**

Create `sweeps/my_sweep.py` with this structure:

```python
"""Description of what this sweep explores.

Usage:
    uv run python -m sweeps.my_sweep --scheme E2
    uv run python -m sweeps my-sweep --scheme E2
"""
from __future__ import annotations
import argparse
import sys
import numpy as np

from stencil_gen.phs import stability_eigenvalue  # or other evaluators
from ._common import (
    SCHEME_PARAMS, STABILITY_TOL,
    load_known_values, save_known_values,
    print_sweep_table, report_stable_ranges,
)


def run_my_sweep(scheme: str, ...) -> dict:
    """Core sweep logic. Returns a summary dict."""
    params = SCHEME_PARAMS[scheme]
    p, q, nextra, nu = params["p"], params["q"], params["nextra"], params["nu"]
    label = params["label"]

    # ... sweep logic ...

    return {"best_param": ..., "stable_at": [...]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--scheme", choices=["E2", "E4"], required=True)
    # ... other arguments ...
    parser.add_argument("--update-known-values", action="store_true")

    args = parser.parse_args(argv)
    summary = run_my_sweep(args.scheme, ...)

    if args.update_known_values:
        kv = load_known_values()
        scheme_key = SCHEME_PARAMS[args.scheme]["label"]
        if scheme_key not in kv:
            kv[scheme_key] = {}
        kv[scheme_key]["my_key"] = {
            "param": summary["best_param"],
            "stable_at": summary["stable_at"],
        }
        save_known_values(kv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**2. Register the subcommand in `__main__.py`**

Add a subparser in the `main()` function:

```python
sub_mine = subparsers.add_parser("my-sweep", help="Description")
sub_mine.add_argument("--scheme", choices=["E2", "E4"], required=True)
# ... mirror the arguments from my_sweep.py ...
sub_mine.add_argument("--update-known-values", action="store_true")
```

Add a dispatch block:

```python
if args.command == "my-sweep":
    from .my_sweep import main as my_main
    return my_main([
        "--scheme", args.scheme,
        *(["--update-known-values"] if args.update_known_values else []),
    ])
```

Note: imports are lazy (inside the `if` block) to avoid loading numpy/sympy
at parse time.

**3. Add to `_run_all()` in `__main__.py`**

Add your sweep to the `sweeps` list in `_run_all()`:

```python
("My sweep E2", my_main, ["--scheme", "E2", ...]),
("My sweep E4", my_main, ["--scheme", "E4", ...]),
```

Include `--quick`-aware argument values using the `quick_*` variables.

**4. Update `known_values.json` schema (if needed)**

If your sweep introduces a new top-level or nested key, document it in this
reference and ensure regression tests in `tests/test_phs.py` know how to
consume it.

### Conventions

- The `main()` function accepts `argv: list[str] | None = None` so it can be
  called both from the CLI and from `_run_all()`.
- Separate the core logic (`run_my_sweep()`) from argument parsing (`main()`)
  so the sweep can be imported and called programmatically.
- Use `SCHEME_PARAMS` for parameter lookup; do not hardcode p/q/nextra/nu.
- Use `STABILITY_TOL` consistently for stable/unstable classification.
- Print results to stdout; only modify `known_values.json` when
  `--update-known-values` is explicitly passed.
- Return `0` on success, non-zero on failure.


## 6. Typical Workflows

### "I changed the stencil derivation, do the optimal parameters still hold?"

This is the most common workflow. After modifying code in `stencil_gen/phs.py`
or `stencil_gen/temo.py`:

```bash
cd scripts/stencil_gen

# Quick check: do the existing known values still produce stable operators?
SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_phs.py -x -q

# If tests pass: the change is compatible, no sweep needed.

# If tests fail: re-run sweeps to find new optimal values.
uv run python -m sweeps all --quick

# If --quick looks good, run the full suite:
uv run python -m sweeps all

# Persist new optimal values:
uv run python -m sweeps epsilon --scheme E2 --update-known-values
uv run python -m sweeps epsilon --scheme E4 --update-known-values
uv run python -m sweeps tension --scheme E2 --update-known-values
uv run python -m sweeps tension --scheme E4 --update-known-values
# ... etc. for each sweep that found different optimal values

# Verify tests pass with updated values:
SYMPY_CACHE_SIZE=50000 uv run pytest tests/test_phs.py -x -q
```

### "I want to explore a new kernel type"

For example, adding an inverse multiquadric kernel:

1. Implement the kernel in `stencil_gen/phs.py` (the `phi()` function and
   related builders).

2. Create `sweeps/imq_sweep.py` based on `epsilon_sweep.py`. The structure
   is almost identical -- just change the kernel name and possibly the
   epsilon range.

3. Register `imq` as a subcommand in `__main__.py`.

4. Run the sweep:
   ```bash
   uv run python -m sweeps imq --scheme E2
   uv run python -m sweeps imq --scheme E4
   ```

5. Add the kernel to `comparison.py` so it appears in side-by-side tables.

6. If the kernel is promising, add it to `known_values.json` and create
   regression tests.

### "I want to add E6 support"

Adding a new scheme requires changes at multiple levels:

1. **Add to `SCHEME_PARAMS` in `_common.py`:**
   ```python
   SCHEME_PARAMS = {
       "E2": {"p": 1, "q": 1, "nextra": 1, "nu": 1, "label": "E2_1"},
       "E4": {"p": 2, "q": 3, "nextra": 0, "nu": 1, "label": "E4_1"},
       "E6": {"p": 3, "q": 5, "nextra": 0, "nu": 1, "label": "E6_1"},
   }
   ```

2. **Update all `choices=["E2", "E4"]` in `__main__.py`** to include `"E6"`.

3. **Update scheme choices in each sweep module's `argparse` definitions**
   (epsilon_sweep.py, tension_sweep.py, etc.).

4. **Run sweeps to discover optimal parameters:**
   ```bash
   uv run python -m sweeps epsilon --scheme E6
   uv run python -m sweeps tension --scheme E6
   uv run python -m sweeps tension-penalty --scheme E6
   uv run python -m sweeps comparison --scheme E6
   ```

5. **Persist results:**
   ```bash
   uv run python -m sweeps epsilon --scheme E6 --update-known-values
   uv run python -m sweeps tension --scheme E6 --update-known-values
   ```

6. **Add E6 entries to `_run_all()`** in `__main__.py`.

7. **Add regression tests** in `tests/test_phs.py` that load the new E6
   entries from `known_values.json`.

Note: E6 stencils have wider boundaries (r = q + 1 + nextra) and require
larger minimum grid sizes (n >= 2r). The sweeps handle this automatically
via the `SCHEME_PARAMS` lookup, but you may need to adjust `--n-values` to
use larger grids.

### "I want to compare stability before and after a change"

Use the `comparison` sweep to get a comprehensive snapshot:

```bash
# Before the change:
uv run python -m sweeps comparison > before.txt

# Make your changes to stencil_gen/...

# After the change:
uv run python -m sweeps comparison > after.txt

# Diff the results:
diff before.txt after.txt
```

The comparison table includes stability eigenvalue, spectral radius, CFL
number, and conservation deficit for all methods, making it easy to spot
regressions or improvements.

### "I want to understand why a particular configuration is unstable"

Use targeted sweeps with narrow ranges:

```bash
# Check a specific epsilon at multiple grid sizes
uv run python -m sweeps epsilon --scheme E4 --kernel gaussian --n-values 10,20,30,40,50,60

# Check the alpha structure at the optimal point
uv run python -m sweeps alpha --scheme E4

# Explore the stability-conservation trade-off
uv run python -m sweeps tension-penalty --scheme E4 --n-sigma 50 --n-gamma 50
```
