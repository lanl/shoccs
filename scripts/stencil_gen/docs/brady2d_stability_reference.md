# Brady-Livescu 2D Analytical Stability Reference

> Any field documented here (e.g. `layer1.boundary_gv_err`,
> `layer_bl42.max_spectral_abscissa`, `layer6.transient_growth_bound`) is
> a valid element of `--objective` in `python -m sweeps optimize` and of
> `--objectives` in `python -m sweeps pareto` (multi-objective NSGA-II;
> see [`pareto_reference.md`](pareto_reference.md)). Layer depth and gate
> are auto-inferred from the field's layer prefix. The same fields
> double as the per-fidelity targets of `python -m sweeps bo`
> (multi-fidelity BoTorch ICM-GP; see
> [`mfbo_reference.md`](mfbo_reference.md)) — a layer index `m` is
> mapped to its dotted field via `--objective <field>` plus
> `--cheap-fidelities <m1> <m2> ...`.

## Problem statement

The benchmark is the two-dimensional varying-coefficient scalar advection
problem from Brady & Livescu 2019 &sect;4.3 (pp. 92&ndash;94):

```
u_t + grad(psi) . grad(u) = 0    on [0, sqrt(2)]^2,  t in [0, 1000]

psi(x, y) = sqrt((x + 0.25)^2 + (y + 0.25)^2)
c_x = d(psi)/dx = (x + 0.25) / psi
c_y = d(psi)/dy = (y + 0.25) / psi

u(x, y, 0)   = sin(2*pi*psi)
u(0, y, t)   = sin(2*pi*(psi(0, y) - t))    (inflow, Dirichlet)
u(x, 0, t)   = sin(2*pi*(psi(x, 0) - t))    (inflow, Dirichlet)

Exact: u(x, y, t) = sin(2*pi*(psi - t))
```

The radial velocity field `(c_x, c_y)` is unit-magnitude everywhere. Inflow
enters at `x = 0` and `y = 0`; outflow exits at `x = sqrt(2)` and
`y = sqrt(2)`.

## Layered pipeline

Each layer is strictly cheaper and more informative than running the full
C++ simulation. The pipeline short-circuits on the first failure, so
expensive layers only run for candidates that pass the cheap checks.

| Layer | Metric | What it catches | Approx. cost (E4, N=81) |
|-------|--------|-----------------|-------------------------|
| L1 | Interior + boundary group velocity error (1D) | Dispersion-quality mismatch at boundary | sub-ms |
| L2 | Rigorous GKS Kreiss determinant test | Boundary-closure instability (necessary **and** sufficient for the 1D reduction) | ~100 ms |
| L3 | 1D eigenvalue `max Re(lambda)` at multiple N | Semi-discrete asymptotic stability, constant coefficient | ~30 ms |
| L3r | BL §4.2 reflecting-hyperbolic `max Re(lambda)` | Boundary-closure instability on energy-conserving system (`div(c) = 0`, purely imaginary continuous spectrum) | ~30 ms |
| L4 | Per-point local GV error across the 2D grid | Local dispersion error from varying `(c_x, c_y)` | ~tens ms |
| L5 | 2D anisotropy `max angle_error` over propagation angles | Grid anisotropy interacting with the radial flow | ~100 ms |
| L6 | Non-normality on 1D operator: spectral + numerical abscissa, Henrici departure, Kreiss constant, transient-growth bound | Transient growth from non-normal 1D spatial operator | ~seconds |
| L7 | Sparse 2D Arnoldi `max Re(lambda)` on the full varying-coefficient operator | True 2D semi-discrete asymptotic stability | few seconds per N |
| L8 | (Plan 42) C++ simulation via Lua bridge | Actual long-time L-infinity bound | minutes |

## Failure thresholds

| Layer | Metric | Threshold constant | Value | Rationale |
|-------|--------|--------------------|-------|-----------|
| L1 | `boundary_gv_err` | `L1_TOL` | 0.05 (5%) | Boundary dispersion error > 5% indicates poor closure quality |
| L2 | `KreissResult.is_stable` | `sigma_tol` | 1e-8 | GKS determinant condition; `sigma_min(M(s)) < tol` indicates instability |
| L3 | `max_stab_eig` | `STABILITY_TOL` | 1e-10 | 1D eigenvalue in right half-plane → unstable semi-discrete scheme |
| L3r | `max_spectral_abscissa` | `BL42_TOL` | 1e-10 | BL §4.2 continuous spectrum is exactly imaginary; see [`bl42_reference.md`](bl42_reference.md) |
| L4 | `max_local_gv_error` | `L4_TOL` | 0.1 (10%) | Looser than L1 because varying-coefficient scaling amplifies baseline error |
| L5 | `max_aligned_error` | `L5_TOL` | 0.05 (5%) | Grid anisotropy projected onto the local propagation direction |
| L6 | `spectral_abscissa` | `STABILITY_TOL` | 1e-10 | 1D operator eigenvalue check |
| L6 | `transient_growth_bound` | `L6_TRANSIENT_GROWTH_TOL` | 50.0 | Kreiss constant bound `e*K > 50` indicates dangerous transient growth |
| L7 | `max_spectral_abscissa` | `L7_TOL` | 5e-3 | 2D varying-coefficient operator; stable schemes show max Re ~ O(1e-3), unstable ~ O(0.1) |
| L7+ | `transient_growth_bound` | `L7_TRANSIENT_GROWTH_TOL` | 50.0 | Same as L6 but on the 2D operator |

## API reference

### `brady2d_stability_score`

Main entry point in `stencil_gen/brady2d_stability.py`.

```python
def brady2d_stability_score(
    scheme: str,           # "E2" or "E4"
    kernel: str,           # "classical" | "tension" | "gaussian" | "multiquadric"
    params: dict,          # {"alpha": [...]}, {"sigma": float}, or {"epsilon": float}
    *,
    max_layer: int = 7,    # highest layer to run (1-8; L8 invokes the C++ bridge)
    short_circuit: bool = True,  # stop at first failure
    layer8_N: int = 31,          # grid resolution forwarded to L8
    layer8_t_final: float = 10.0,  # physical end time forwarded to L8
) -> StabilityReport
```

### `StabilityReport`

Dataclass returned by `brady2d_stability_score`.

| Field | Type | Description |
|-------|------|-------------|
| `layer1` .. `layer7` | `dict \| None` | Per-layer metrics (populated when that layer runs) |
| `layer_bl42` | `dict \| None` | L3r BL §4.2 reflecting-hyperbolic eigenvalue result |
| `layer8` | `dict \| None` | L8 C++ simulation result (populated when `max_layer >= 8`) |
| `kreiss` | `KreissResult \| None` | Alias for `layer2` (the Kreiss determinant result) |
| `non_normality` | `NonNormalityReport \| None` | Populated by L6 (1D) or L7 (2D) |
| `overall_verdict` | `"pass" \| "fail" \| "unknown"` | Final verdict |
| `failed_layer` | `int \| None` | First layer that failed |
| `failed_reason` | `str` | Human-readable failure description |
| `compute_time` | `float` | Total wall-clock seconds |

The `__str__` method produces a compact per-layer summary table.

### `kreiss_stability_check`

Rigorous GKS determinant test in `stencil_gen/gks_kreiss.py`. Implements
Trefethen 1983 (pp. 206-207).

```python
def kreiss_stability_check(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    *,
    s_grid_params: dict | None = None,
    sigma_tol: float = 1e-8,
    refine: bool = True,
    min_s_magnitude: float = 0.1,
) -> KreissResult
```

`KreissResult` fields: `is_stable`, `witness_s`, `witness_sigma_min`,
`imaginary_axis_perturbation_verdict`, `defective_kappa_detected`,
`s_grid_shape`, `compute_time`, `sigma_min_field`, `s_grid`,
`n_admissible_roots`.

### `compute_non_normality`

Non-normality diagnostics in `stencil_gen/non_normality.py`. Calibration
bands from Trefethen & Embree 2005, ch. 14.

```python
def compute_non_normality(
    L,                           # sparse or dense matrix
    *,
    small_dense_threshold: int = 900,
    epsilon_values: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
    s_grid_params: dict | None = None,
) -> NonNormalityReport
```

`NonNormalityReport` fields: `spectral_abscissa`, `numerical_abscissa`,
`henrici_departure`, `eigenvector_condition`, `pseudospectral_abscissae`
(dict mapping epsilon to abscissa), `kreiss_constant`,
`transient_growth_bound` (`= e * kreiss_constant`), `n`, `compute_time`,
`notes`.

## Layer 8 — C++ simulation

L8 closes the loop between the analytical stack (L1&ndash;L7) and the
compiled C++ solver. It is the only layer that runs an actual
time-stepping simulation, so it is gated behind `max_layer >= 8` and
typically only invoked for candidates that have already passed the cheap
analytical layers.

### Architecture

Python builds a Lua config string from the `lua-configs/brady_livescu_4_3.lua`
template, writes it to a per-call `tempfile.TemporaryDirectory`, and invokes
`build/src/app/shoccs` as a subprocess. The simulation writes
`logs/system.csv` under the tempdir (so concurrent invocations do not race),
Python parses the final `Linf` column, and a `BridgeResult` is returned.
The mechanics are described in detail in
`docs/brady2d_cpp_bridge_reference.md`.

### Dispatch

`layer8_cpp_simulation(scheme, kernel, params, *, N, t_final)` maps
`(scheme, kernel)` to the Lua `scheme.type` string expected by `stencil::from_lua`:

| `(scheme, kernel)` | Lua `scheme.type` | Passed param |
|--------------------|-------------------|--------------|
| `("E4", "classical")` | `"E4u"` | `alpha = {...}` (2-vector) |
| `("E4", "tension")` | `"tension_E4u"` | `sigma` |
| `("E4", "gaussian")` | `"gaussian_E4u"` | `epsilon` |
| `("E4", "multiquadric")` | `"multiquadric_E4u"` | `epsilon` |

E2 spline variants and cut-cell (`E4_1`, not `E4u_1`) variants are
deferred &mdash; see `plans/42-cpp-bridge-runtime-parameterized-stencils.md`
items 42.10a / 42.10b.

### Failure thresholds

| Metric | Threshold constant | Value | Rationale |
|--------|--------------------|-------|-----------|
| `stable` | `BridgeResult.stable` | `final_linf < 10.0` and finite | Simulation did not blow up or timeout |
| `final_linf` | `L8_FINAL_LINF_TOL` | 1.0 | Schemes that survive L1&ndash;L7 but diverge under the full 2D varying-coefficient flow still trip this at `t_final = 10` |

A layer-8 failure sets `failed_layer = 8` and `failed_reason` to one of
`"stable=False"` or `"final_linf=<v> > L8_FINAL_LINF_TOL=1.0"`.

### `BridgeResult`

Returned inside `report.layer8["bridge_result"]`:

| Field | Type | Description |
|-------|------|-------------|
| `final_linf` | `float` | L&infin; at `t_final`; `nan` on failure |
| `linf_trace` | `np.ndarray` | Per-step L&infin; history |
| `t_trace` | `np.ndarray` | Per-step physical time |
| `stable` | `bool` | `isfinite(final_linf) and final_linf < 10.0` |
| `wall_time_s` | `float` | Subprocess wall time |
| `exit_code` | `int` | `shoccs` exit status |
| `stderr` | `str` | Captured stderr (diagnostic on failure) |

### Cost and runtime knobs

Default `N = 31, t_final = 10.0` runs in ~1&nbsp;second per call at E4.
For fast integration testing, pass `layer8_N=21, layer8_t_final=1.0`
&mdash; the full L1&ndash;L8 pipeline then completes in ~20&nbsp;seconds
(most of it L7's 2D Arnoldi, not the C++ run itself).

### Sweep integration

`sweeps/brady2d_sweep.py` (subcommand `brady2d`) sweeps a parameter range
at `max_layer <= 7`, ranks survivors, and then re-runs the top-3 at
`max_layer = 8` when invoked with `--validate-with-cpp`:

```bash
cd scripts/stencil_gen
uv run python -m sweeps brady2d --scheme E4 --kernel tension \
    --param-range 2 6 20 --max-layer 6 --validate-with-cpp
```

Ranking uses `layer6.transient_growth_bound` ascending when L6 is
present, falling back to `layer3.max_stab_eig` otherwise. The C++ build
is compiled once up front; Lua configs are the only thing that changes
per sweep point.

## Optimization

The layered pipeline also backs a scipy-based optimizer that turns the
brute-force sweep into actual optimization (single-objective +
feasibility cliff, random-restart multi-start, SHGO and DE for global
coverage, and a staged cheap-inner + expensive-validator cascade). See
[`optimization_reference.md`](optimization_reference.md) for the full
API and CLI surface.

Entry points:

- `stencil_gen/optimizer.py` — `make_objective`, `run_scipy_local`,
  `multi_start_optimize`, `run_scipy_shgo`, `run_scipy_de`,
  `run_staged_optimize`, `OptimizeResult`, `DEFAULT_BOUNDS`.
- `python -m sweeps optimize ...` — CLI driver that persists winners
  under `known_values.json["brady2d_optima"]` and can round-trip an L8
  verdict when invoked with `--validate-with-cpp`.
- `stencil_gen/benchmarks/alpha_basin_survey.py` — multi-seed diversity
  study for the 2D classical-alpha landscape (Brady-Livescu Table 4
  analog).

The optimizer reuses the `StabilityReport` schema documented above:
objectives are dotted paths into the report
(`layer1.boundary_gv_err`, `layer3.max_stab_eig`,
`layer6.transient_growth_bound`, ...), and the feasibility cliff fires
when `failed_layer <= gate_layer`.

## CLI usage

```bash
# Score a single scheme at max_layer=6
cd scripts/stencil_gen
uv run python -m stencil_gen.brady2d --scheme E4 --kernel tension --sigma 3.0 --max-layer 6

# Run calibration across all families
uv run python -m stencil_gen.brady2d --run-calibration --max-layer 6

# Run calibration and persist to known_values.json
uv run python -m stencil_gen.brady2d --run-calibration --max-layer 7 --update-known-values
```

L8 has no dedicated CLI entry point. Drive it via the Python API
(`brady2d_stability_score(..., max_layer=8, layer8_N=..., layer8_t_final=...)`)
or through the sweep subcommand (`python -m sweeps brady2d
--validate-with-cpp`).

## Calibration results

Results stored in `sweeps/known_values.json` under the `"brady2d_calibration"` key.
Regression tests in `tests/test_phs.py::TestRegressionBrady2DCalibration` verify
that re-running at `max_layer=3` reproduces the stored verdicts.

At `max_layer=6` (2026-04-14, with L3r/BL42 enabled):
- **Pass all layers (2/9):** E4_classical, E2_phs_k2
- **Fail at L3r/BL42 (4/9):** E4_phs_k2, E4_tension_3, E4_gaussian_09, E4_multiquadric_1
- **Fail at L1 (3/9):** E2_tension_6, E2_gaussian_2, E2_multiquadric_1

The L3r (BL42) layer is a strict discriminator: schemes that pass L3
(constant-coefficient advection eigenvalue) can still fail on the
reflecting-BC hyperbolic system. See [`bl42_reference.md`](bl42_reference.md)
for the full calibration table and API.

## References

- Brady & Livescu 2019: "High-order multiblock/multiresolution finite
  difference methods for the compressible Navier-Stokes equations", &sect;4.3 pp. 92-94
- Trefethen 1983: "Group velocity in finite difference schemes", pp. 204-210
  (GKS determinant condition)
- Trefethen & Embree 2005: *Spectra and Pseudospectra*, ch. 14 (Kreiss constant
  calibration bands)
