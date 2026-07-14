---
name: group-velocity-analysis
description: Guide to group velocity analysis for finite difference stencils. Use when discussing wave propagation, dispersion, modified wavenumber, parasitic modes, GKS stability, cut-cell group velocity, or 2D anisotropy.
---

# Group Velocity Analysis

Group velocity `C(ξ) = dω/dξ` governs how wave packets propagate through finite difference discretizations. Unlike phase velocity, group velocity determines energy transport and controls boundary stability (GKS theory).

## Quick Reference

| Function | Purpose |
|----------|---------|
| `interior_group_velocity(p, nu, xi)` | GV profile for E2-E8 interior stencils |
| `boundary_group_velocity(p, q, nextra, nu, sigma, kernel, xi)` | GV for RBF/tension boundary rows |
| `boundary_group_velocity_classical(boundary_rows, alpha_values, order, xi)` | GV for symbolic boundary stencils |
| `cut_cell_group_velocity(result, psi_sym, psi_val, alpha, xi)` | GV for TEMO cut-cell stencils — **superseded for stability** by the plan-48 normal-mode tier below; retained as a dispersion diagnostic |
| `psi_sweep_group_velocity(scheme, psi_values, alpha, xi)` | Sweep GV across ψ ∈ [0,1] — **superseded for stability** (and derives per call; the plan-48 tier uses the cached evaluator); retained as a dispersion diagnostic |
| `group_velocity_2d(kappa_x_star, kappa_y_star, xi, eta, a, b)` | 2D GV vector from pre-computed modified wavenumber arrays |
| `anisotropy_profile(p, nu, theta, xi_mag)` | Grid anisotropy vs propagation angle |
| `gks_group_velocity_check(D, xi, side=)` | GKS eigenmode boundary stability diagnostic (heuristic) |
| `kreiss_stability_check(interior, offsets, brows, row_indices=)` | Rigorous GKS Kreiss determinant test (Trefethen 1983); augmented m > p closures via `row_indices` (plan 48) |
| `make_cutcell_evaluator(scheme, bc=)` | Cached lambdified cut-cell Dirichlet closure evaluator (plan 48; E4_1 derives once, then ~µs per ψ) |
| `kreiss_cutcell_check(scheme, alpha, psi)` | ψ-parameterized rigorous GKS check on the wall-eliminated Dirichlet block; `CutCellInfeasible` on denominator vanishing |
| `kreiss_margin_over_psi(scheme, alpha, psi_grid=)` | Worst-case-over-ψ Kreiss margin (default 20-point grid clustered near ψ=0) |
| `mode_split(iw, io, s)` / `reflection_matrix(...)` / `max_reflection(...)` | Trefethen-1984 reflection coefficients R(s) and axis margin ‖R‖ |
| `mirror_closure(brows, interior, from_physical_right=)` | Right-end closure mirror (pure index reversal; production entry `False` = codegen revneg symmetry) |
| `reflection_over_psi(scheme, alpha, psi_grid=, side=)` | ψ-swept reflection margin |
| `build_interval_operator(scheme, alpha, psi_L, psi_R, N)` | Two-ended cut-cell interval operator, `derivative.cpp` layout, wall columns dropped |
| `interval_stability_scan(scheme, alpha, psi_pairs=, N_values=)` | Dense abscissa + Kreiss-constant resolvent scan over (ψ_L, ψ_R, N) — catches finite-interval modes (e.g. the E2_1 outflow law `(1−ψ_R)/(1+ψ_R)`) |
| `round_trip_check(...)` / `cutcell_round_trip(scheme, alpha, psi_L, psi_R)` | Trefethen-1985 two-boundary round trip: ρ(E_Δj(s)) radius + decoupled reflection-product screen |
| `cutcell_stability_report(scheme, alpha, psi_grid=, psi_pairs=)` | Opt-in `layer_cutcell` report: worst_kreiss_sigma, max_reflection, reflection_product, round_trip_radius, interval_worst_abscissa (never populated by the L1-L8 cascade; optimizer wiring is plan 49) |
| `local_group_velocity(weights_func, x, xi)` | Local GV for varying coefficients (1D) |
| `local_group_velocity_2d_varying(stencil_x, stencil_y, c_x, c_y, xi)` | Per-point local GV error on a 2D varying-coefficient grid |
| `anisotropy_over_coefficient_field(scheme, c_x, c_y, grid, theta, xi_mag)` | 2D anisotropy projected onto a varying flow field |
| `ray_trace_group_velocity(C_field, x, xi, ...)` | Ray tracing dx/dt = C, dξ/dt = -∂C/∂x |
| `brady2d_stability_score(scheme, kernel, params, max_layer=)` | Layered stability scoring for Brady-Livescu 2D benchmark |
| `build_bl42_operator(D)` | 2×2 block operator for BL §4.2 reflecting hyperbolic system |
| `layer_bl42_reflecting_hyperbolic(scheme, kernel, params, n_values=)` | L3r eigenvalue analysis on BL §4.2 (purely imaginary continuous spectrum) |
| `make_multi_objective(scheme, kernel, fields, *, gate_layer=None, max_layer=None)` | Vector-valued objective factory for NSGA-II (plan 45); finite sentinel `1e12` on infeasibility |
| `run_nsga2(scheme, kernel, fields, bounds, *, pop_size, n_gen, seed)` | Multi-objective Pareto driver via pymoo NSGA-II; returns `ParetoResult` with HV trace |

## Key Facts

- **Sign convention:** `C(ξ) = d(Im(κ*))/dξ` where `κ*(ξ)` is the modified wavenumber
- **Error amplification:** Group velocity error is `(2p+1)×` the phase velocity error for a 2p-order scheme
- **Cutoff wavenumber:** Where `C(ξ) = 0` — above this, energy propagates backwards (parasitic)
- **GKS connection:** Boundary instability ⟺ outgoing modes (`C > 0`) spontaneously generated at boundary
- **Scaling advantage:** GV analysis is O(1) per stencil; eigenvalue analysis is O(N³) per grid

## When to Use

- Analyzing dispersion behavior of new stencil schemes
- Diagnosing boundary instabilities via GKS group velocity check (heuristic) or rigorous Kreiss determinant test
- Evaluating cut-cell stencils across the ψ range — for **stability**, use the plan-48 normal-mode tier (`kreiss_margin_over_psi`, `reflection_over_psi`, `interval_stability_scan`, `cutcell_round_trip`, aggregated by `cutcell_stability_report`); the GV-based `cut_cell_group_velocity` / `psi_sweep_group_velocity` remain useful only as dispersion diagnostics
- Checking 2D grid anisotropy for tensor-product operators
- Comparing GV analysis cost vs eigenvalue analysis for large problems
- Scoring boundary closures against the Brady-Livescu 2D varying-coefficient benchmark (L1-L7 layered pipeline)
- Layer 8 closed-loop validation: re-running top survivors through the compiled C++ solver (`run_cpp_brady2d` / `sweeps brady2d --validate-with-cpp`) to confirm analytical L1-L7 verdicts empirically
- Non-normality diagnostics: spectral/numerical abscissa, Kreiss constant, transient growth bound
- Optimizing boundary-closure parameters against any `StabilityReport` field using `stencil_gen.optimizer` (Nelder-Mead, COBYQA, SHGO, DE, staged cascade, multi-start); the scoring pipeline is the objective function
- Testing boundary closures against the BL §4.2 neutrally-stable hyperbolic system — the strictest `div(c) = 0` discriminator, purely imaginary continuous spectrum, energy-conserving reflecting BCs (available as `layer_bl42.*` fields in `StabilityReport`)
- Multi-objective Pareto exploration when stability metrics conflict (e.g., tension closures have low `layer1.boundary_gv_err` but high `layer_bl42.max_spectral_abscissa`; classical closures have the opposite trade-off) — use `python -m sweeps pareto` for an NSGA-II driver that exposes the full front (plan 45). The single-objective optimizer collapses these onto a scalar; Pareto exposes the trade-off explicitly.

## Detailed Reference

For complete API documentation, usage examples, and mathematical derivations, see:
- `scripts/stencil_gen/docs/group_velocity_reference.md`
- `scripts/stencil_gen/docs/brady2d_stability_reference.md` (layered stability pipeline for the Brady-Livescu 2D benchmark)
- `docs/brady2d_cpp_bridge_reference.md` (L8 C++ bridge architecture: Lua template, subprocess runner, runtime-parameterized spline families)
- `scripts/stencil_gen/docs/optimization_reference.md` (plan-43 optimizer: objective factory, local/global/staged drivers, multi-start, classical-α basin survey)
- `scripts/stencil_gen/docs/bl42_reference.md` (plan-44 BL §4.2 reflecting-hyperbolic layer: problem statement, 2×2 block operator construction, calibration results)
- `scripts/stencil_gen/docs/pareto_reference.md` (plan-45 NSGA-II multi-objective Pareto driver: `make_multi_objective`, `run_nsga2`, hypervolume tracking, per-run JSON persistence under `sweeps/pareto_fronts/`)
- `scripts/stencil_gen/docs/cutcell_normal_mode_reference.md` (plan-48 cut-cell normal-mode tier: augmented Kreiss determinant, reflection coefficients, two-boundary round trip, interval resolvent scan, `layer_cutcell` report; limits — 1D, semi-discrete, straight cuts)
