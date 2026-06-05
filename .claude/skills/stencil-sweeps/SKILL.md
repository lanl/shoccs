---
name: stencil-sweeps
description: Guide to parameter sweep scripts for stencil stability analysis. Use when running epsilon sweeps, tension sweeps, footprint analysis, alpha extraction, or exploring optimal parameters for PHS/RBF stencils.
---

# Parameter Sweeps

The `sweeps/` package provides standalone scripts for parameter space exploration, separated from the test suite. Sweeps discover optimal parameters; regression tests verify them.

## CLI Quick Reference

```bash
cd scripts/stencil_gen

# Epsilon (Gaussian/MQ kernel parameter) sweep
uv run python -m sweeps epsilon --scheme E2 --kernel gaussian --n-eps 60

# Tension (sigma parameter) sweep
uv run python -m sweeps tension --scheme E4 --n-sigma 61

# 2D tension + conservation penalty sweep
uv run python -m sweeps tension-penalty --scheme E4 --n-sigma 25 --n-gamma 25

# Mixed (per-row) epsilon sweep
uv run python -m sweeps mixed-epsilon --scheme E4

# Boundary footprint (nextra) sweep
uv run python -m sweeps footprint --n-sigma 50

# Multi-method comparison table
uv run python -m sweeps comparison --scheme E2

# Alpha extraction from RBF weights
uv run python -m sweeps alpha --scheme E2

# Group velocity vs stability Pareto front (research / docs only)
uv run python -m sweeps gv-stability-pareto --scheme E2 --param tension --n-points 61

# Brady-Livescu 2D stability scoring (layered analytical pipeline)
uv run python -m stencil_gen.brady2d_cli --scheme E4 --kernel tension --sigma 3.0 --max-layer 6

# Brady-Livescu 2D sweep (L1-L7 analytical; kernel in {classical, tension, gaussian, multiquadric})
uv run python -m sweeps brady2d --scheme E4 --kernel tension --param-range 2 4 3 --max-layer 3

# Same, but re-run top-3 survivors against the compiled C++ solver (L8 end-to-end validation)
uv run python -m sweeps brady2d --scheme E4 --kernel tension --param-range 2 4 11 --max-layer 3 --validate-with-cpp

# Optimize boundary-closure parameters (Nelder-Mead, COBYQA, SHGO, DE, or staged cascade)
uv run python -m sweeps optimize --scheme E4 --kernel tension --objective layer1.boundary_gv_err --bounds 0.5 20 --method staged --n-restarts 10

# Optimize against the BL §4.2 neutrally-stable hyperbolic eigenvalue (L3r; strictest discriminator)
uv run python -m sweeps optimize --scheme E4 --kernel tension --objective layer_bl42.max_spectral_abscissa --bounds 0.5 20 --method Nelder-Mead --max-evals 40

# NSGA-II multi-objective Pareto front over 2-3 stability metrics (plan 45); persists to sweeps/pareto_fronts/
uv run python -m sweeps pareto --scheme E4 --kernel classical --objectives layer1.boundary_gv_err layer_bl42.max_spectral_abscissa --bounds -2 2 0.05 2 --pop-size 40 --n-gen 30 --seed 1 --persist

# Multi-fidelity Bayesian optimization over the cascade (plan 47, BoTorch qMFKG); persists to sweeps/bo_runs/
uv run python -m sweeps bo --scheme E4 --kernel classical --objective layer7.max_spectral_abscissa --cheap-fidelities 1 3 5 6 --bounds -2 2 0.05 2 --budget-evals 60 --seed 1 --persist

# Run all sweeps (quick mode for smoke testing)
uv run python -m sweeps all --quick

# Update known_values.json with discovered optima
uv run python -m sweeps epsilon --scheme E2 --update-known-values
```

## Group Velocity Objectives

- `--include-gv` (tension/epsilon/tension-penalty/footprint): adds a `gv_err` column and a "Best feasible by GV error" line; persists additive `*.gv_error` and parallel `*_gv` keys to `known_values.json` when paired with `--update-known-values`. Stability stays the only hard feasibility gate; GV is a soft secondary objective minimized over the feasible set.
- `--check-gks` (tension/epsilon): advisory only — runs `gks_group_velocity_check` at the stability optimum and prints `WARNING: outgoing boundary mode at xi=…` lines if any are detected. Necessary-not-sufficient for instability; never alters the optimum.

## Workflow

1. **Run sweep** → discovers optimal parameters, prints tables
2. **Update JSON** → `--update-known-values` writes to `sweeps/known_values.json`
3. **Regression tests** → `TestRegression*` classes in `test_phs.py` verify against JSON

## Key Files

| File | Purpose |
|------|---------|
| `sweeps/known_values.json` | Single source of truth for optimal parameters |
| `sweeps/_common.py` | `STABILITY_TOL`, `SCHEME_PARAMS`, shared helpers |
| `sweeps/gv_objectives.py` | Group-velocity scalar wrappers + GKS advisory printer |
| `sweeps/epsilon_sweep.py` | Gaussian/MQ epsilon sweeps |
| `sweeps/mixed_epsilon_sweep.py` | Per-row epsilon optimization |
| `sweeps/tension_sweep.py` | Tension sigma sweeps |
| `sweeps/tension_penalty_sweep.py` | 2D (sigma, gamma) joint sweeps |
| `sweeps/footprint_sweep.py` | Boundary footprint (nextra) sweeps |
| `sweeps/gv_stability_pareto.py` | 1D parametric scan with dominance filter — read-only research/docs |
| `sweeps/pareto.py` | NSGA-II multi-objective Pareto driver (plan 45); produces full fronts |
| `sweeps/_pareto_io.py` | JSON serialization for `ParetoResult` → `sweeps/pareto_fronts/<scheme>_<kernel>_<mangled>.json` |
| `sweeps/comparison.py` | Multi-method comparison tables |
| `sweeps/alpha_extraction.py` | Extract stencil alphas from RBF weights |
| `sweeps/brady2d_sweep.py` | Brady-Livescu §4.3 2D sweep; `--validate-with-cpp` re-runs top-3 survivors through the compiled C++ solver (L8) |
| `sweeps/bo.py` | Multi-fidelity Bayesian optimization driver (plan 47); BoTorch qMFKG over the cascade with cost-aware acquisition and ICM kernel |
| `sweeps/_bo_io.py` | JSON serialization for `BOResult` → `sweeps/bo_runs/<scheme>_<kernel>_<mangled>_<seed>.json` |

## When to Use

- After changing stencil derivation code, verify optimal parameters still hold
- Exploring a new kernel type or scheme order
- Producing comparison tables for documentation/papers
- Diagnosing why a scheme is unstable at certain parameters
- Optimizing boundary-closure parameters against a `StabilityReport` field (single-objective + feasibility cliff; staged cheap-inner + expensive-validator pipeline); persists to `known_values.json["brady2d_optima"]`. See `scripts/stencil_gen/docs/optimization_reference.md`.
- Multi-objective Pareto exploration when two metrics genuinely conflict (e.g., `layer1.boundary_gv_err` vs `layer_bl42.max_spectral_abscissa`); use `sweeps pareto` (NSGA-II via pymoo). See `scripts/stencil_gen/docs/pareto_reference.md`.
- Multi-fidelity Bayesian optimization when expensive-validator wall-time dominates and you want a principled cost-aware HF/cheap split (instead of `optimize --method staged`'s hand-coded top-K threshold); use `sweeps bo` (BoTorch qMFKG with discrete-fidelity ICM kernel). See `scripts/stencil_gen/docs/mfbo_reference.md`.

## Detailed Reference

For complete CLI documentation, JSON schema, and adding new sweeps, see:
- `scripts/stencil_gen/docs/sweeps_reference.md`
- `scripts/stencil_gen/docs/pareto_reference.md` (plan 45 NSGA-II multi-objective driver)
- `scripts/stencil_gen/docs/mfbo_reference.md` (plan 47 multi-fidelity Bayesian optimization driver)
