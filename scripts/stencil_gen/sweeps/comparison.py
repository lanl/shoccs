"""Multi-method comparison tables for RBF-augmented stencils.

Extracted from TestCorrectedComparison, TestComparisonTable,
TestTensionComparison, TestTensionOptimalSigma in test_phs.py.

Compares PHS k=2, Gaussian RBF, Tension spline, and Tension+penalty
methods side-by-side with metrics: stability eigenvalue, spectral radius,
CFL number (RK4), and conservation deficit.

Usage:
    uv run python -m sweeps.comparison --scheme E2
    uv run python -m sweeps.comparison --scheme E4
    uv run python -m sweeps.comparison              # both schemes
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from stencil_gen.phs import (
    build_diff_matrix_mixed_epsilon,
    build_diff_matrix_rbf,
    build_diff_matrix_rbf_penalty,
    stability_eigenvalue,
    stability_eigenvalue_from_matrix,
    uniform_boundary_weights,
)

from ._common import SCHEME_PARAMS, STABILITY_TOL, load_known_values, save_known_values

# Floating-point eigenvalue solvers return tiny positive real parts (~1e-14)
# RK4 stability limit along the imaginary axis
RK4_IMAG_LIMIT = 2.828


# ---------------------------------------------------------------------- helpers


def _metrics(D):
    """Compute (stab_eig, spectral_radius, cfl_rk4, conservation_deficit).

    stab_eig = max Re(eig(-D_bc)) where D_bc = D[1:, 1:] (inflow removed).
    Spectral radius and CFL use the full D matrix.
    """
    stab_eig = stability_eigenvalue_from_matrix(D)
    eigvals = np.linalg.eigvals(D)
    spec_rad = float(np.max(np.abs(eigvals)))
    cfl = RK4_IMAG_LIMIT / spec_rad if spec_rad > 0 else float("inf")
    deficit = float(np.max(np.abs(np.sum(D, axis=0))))
    return stab_eig, spec_rad, cfl, deficit


def _find_best_epsilon(n, p, q, nu, nextra, kernel="gaussian"):
    """Coarse + fine sweep for best epsilon using corrected stability."""
    eps_coarse = np.logspace(np.log10(0.01), np.log10(15), 80)
    best_eps, best_se = None, float("inf")
    for e in eps_coarse:
        se = stability_eigenvalue(n, p, q, e, kernel, nu, nextra)
        if se < best_se:
            best_se = se
            best_eps = e

    lo = max(0.001, best_eps / 3)
    hi = min(50.0, best_eps * 3)
    for e in np.linspace(lo, hi, 200):
        se = stability_eigenvalue(n, p, q, e, kernel, nu, nextra)
        if se < best_se:
            best_se = se
            best_eps = e

    return best_eps, best_se


def _find_best_sigma(n, p, q, nu, nextra):
    """Coarse + fine sweep for best tension sigma using corrected stability."""
    sigmas_coarse = np.concatenate(
        [[0.0], np.logspace(np.log10(0.01), np.log10(20), 100)]
    )
    best_sigma, best_se = None, float("inf")
    for s in sigmas_coarse:
        se = stability_eigenvalue(n, p, q, s, "tension", nu, nextra)
        if se < best_se:
            best_se = se
            best_sigma = s

    if best_sigma > 0.5:
        lo = max(0.0, best_sigma * 0.5)
        hi = min(30.0, best_sigma * 2.0)
    else:
        lo = 0.0
        hi = 2.0
    for s in np.linspace(lo, hi, 200):
        se = stability_eigenvalue(n, p, q, s, "tension", nu, nextra)
        if se < best_se:
            best_se = se
            best_sigma = s

    return best_sigma, best_se


def _find_best_sigma_gamma(n, p, q, nu, nextra, sigma_hint):
    """2D sweep over (sigma, gamma) with corrected stability metric."""
    sigmas = np.concatenate(
        [[0.0], np.linspace(max(0.01, sigma_hint - 5), sigma_hint + 10, 25)]
    )
    gammas = np.concatenate([[0.0], np.logspace(-2, 2, 25)])

    best_sigma, best_gamma, best_se = None, None, float("inf")
    for s in sigmas:
        for g in gammas:
            D = build_diff_matrix_rbf_penalty(
                n, p, q, s, "tension", nu, nextra, gamma=g,
            )
            se = stability_eigenvalue_from_matrix(D)
            if se < best_se:
                best_se = se
                best_sigma = s
                best_gamma = g

    return best_sigma, best_gamma, best_se


def _find_best_mixed_epsilon(p, q, nextra, nu, r, kernel, n=40):
    """Coordinate descent to find per-row optimal epsilon (returns list)."""
    best_single, _ = _find_best_epsilon(n, p, q, nu, nextra, kernel)
    current = [best_single] * r
    eps_vals = np.logspace(np.log10(0.3), np.log10(10.0), 40)

    def _stab_eig(eps_list):
        D = build_diff_matrix_mixed_epsilon(
            n, p, q, eps_list, kernel, nu, nextra
        )
        return stability_eigenvalue_from_matrix(D)

    current_se = _stab_eig(current)
    for _ in range(3):
        for row in range(r):
            for eps in eps_vals:
                trial = list(current)
                trial[row] = eps
                se = _stab_eig(trial)
                if se < current_se:
                    current_se = se
                    current[row] = eps

    return current, current_se


def _print_comparison_table(title, results):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    hdr = (f"  {'Method':>30s}  {'stab_eig':>14s}  {'|lambda|_max':>14s}"
           f"  {'CFL(RK4)':>10s}  {'cons deficit':>14s}  {'status':>10s}")
    print(hdr)
    print(f"  {'-' * 30}  {'-' * 14}  {'-' * 14}"
          f"  {'-' * 10}  {'-' * 14}  {'-' * 10}")
    for name, se, sr, cfl, cd in results:
        status = "STABLE" if se < STABILITY_TOL else "unstable"
        print(f"  {name:>30s}  {se:14.6e}  {sr:14.6e}"
              f"  {cfl:10.4f}  {cd:14.6e}  {status:>10s}")


# ------------------------------------------------------------------ main logic


def run_comparison(scheme: str, n_values: list[int]) -> dict:
    """Run multi-method comparison for a scheme.

    Finds optimal parameters at n=40, then evaluates across grid sizes.
    Returns a summary dict with best parameters and stable grid sizes.
    """
    params = SCHEME_PARAMS[scheme]
    p, q, nextra, nu = params["p"], params["q"], params["nextra"], params["nu"]
    label = params["label"]
    r = q + 1 + nextra  # number of boundary rows per side

    n_opt = 40 if 40 in n_values else n_values[0]

    # Find optimal parameters at n_opt
    print(f"\n  Finding optimal parameters for {label} at n={n_opt}...")
    eps_g, _ = _find_best_epsilon(n_opt, p, q, nu, nextra, "gaussian")
    eps_m, _ = _find_best_epsilon(n_opt, p, q, nu, nextra, "multiquadric")
    sigma_t, _ = _find_best_sigma(n_opt, p, q, nu, nextra)
    sg, gg, _ = _find_best_sigma_gamma(n_opt, p, q, nu, nextra, sigma_t)

    print(f"  Optimal params (found at n={n_opt}):")
    print(f"    Gaussian epsilon*={eps_g:.4f}")
    print(f"    Multiquadric epsilon*={eps_m:.4f}")
    print(f"    Tension sigma*={sigma_t:.4f}")
    print(f"    Tension+penalty sigma*={sg:.4f}, gamma*={gg:.4f}")

    # For E4, also find mixed-epsilon
    mixed_eps = None
    if scheme == "E4":
        mixed_eps, _ = _find_best_mixed_epsilon(
            p, q, nextra, nu, r, "gaussian", n_opt
        )
        eps_str = ",".join(f"{e:.2f}" for e in mixed_eps)
        print(f"    Mixed Gaussian eps=[{eps_str}]")

    # Evaluate across grid sizes
    all_stable = {
        "phs_k2": [], "gaussian": [], "multiquadric": [],
        "tension": [], "tension_penalty": [],
    }
    if mixed_eps is not None:
        all_stable["mixed_gaussian"] = []

    for n in n_values:
        results = []

        # 1. PHS k=2 (sigma -> 0)
        D_phs = build_diff_matrix_rbf(n, p, q, 1e-15, "tension", nu, nextra)
        m = _metrics(D_phs)
        results.append(("PHS k=2 (sigma=0)", *m))
        if m[0] < STABILITY_TOL:
            all_stable["phs_k2"].append(n)

        # 2. Gaussian epsilon*
        D_gauss = build_diff_matrix_rbf(n, p, q, eps_g, "gaussian", nu, nextra)
        m = _metrics(D_gauss)
        results.append((f"Gaussian eps*={eps_g:.3f}", *m))
        if m[0] < STABILITY_TOL:
            all_stable["gaussian"].append(n)

        # 3. Multiquadric epsilon*
        D_mq = build_diff_matrix_rbf(n, p, q, eps_m, "multiquadric", nu, nextra)
        m = _metrics(D_mq)
        results.append((f"MQ eps*={eps_m:.3f}", *m))
        if m[0] < STABILITY_TOL:
            all_stable["multiquadric"].append(n)

        # 4. Tension sigma* (gamma=0)
        D_tension = build_diff_matrix_rbf(n, p, q, sigma_t, "tension", nu, nextra)
        m = _metrics(D_tension)
        results.append((f"Tension sigma*={sigma_t:.2f}", *m))
        if m[0] < STABILITY_TOL:
            all_stable["tension"].append(n)

        # 5. Tension + conservation penalty (sigma*, gamma*)
        D_pen = build_diff_matrix_rbf_penalty(
            n, p, q, sg, "tension", nu, nextra, gamma=gg,
        )
        m = _metrics(D_pen)
        results.append((f"Tension s={sg:.2f} g={gg:.2f}", *m))
        if m[0] < STABILITY_TOL:
            all_stable["tension_penalty"].append(n)

        # 6. Mixed-epsilon Gaussian (E4 only)
        if mixed_eps is not None:
            D_mixed = build_diff_matrix_mixed_epsilon(
                n, p, q, mixed_eps, "gaussian", nu, nextra
            )
            m = _metrics(D_mixed)
            eps_str = ",".join(f"{e:.1f}" for e in mixed_eps)
            results.append((f"Mixed Gauss [{eps_str}]", *m))
            if m[0] < STABILITY_TOL:
                all_stable["mixed_gaussian"].append(n)

        _print_comparison_table(
            f"{label} Comparison (n={n}, p={p}, q={q}, nextra={nextra})", results,
        )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  {label} Stability Summary")
    print(f"{'=' * 60}")
    for method, stable_ns in all_stable.items():
        if stable_ns:
            print(f"  {method:>20s}: stable at n={stable_ns}")
        else:
            print(f"  {method:>20s}: no stable grid sizes found")

    return {
        "gaussian_epsilon": round(eps_g, 4),
        "multiquadric_epsilon": round(eps_m, 4),
        "tension_sigma": round(sigma_t, 4),
        "penalty_sigma": round(sg, 4),
        "penalty_gamma": round(gg, 4),
        "mixed_epsilon": [round(e, 4) for e in mixed_eps] if mixed_eps else None,
        "stable_at": all_stable,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.comparison",
        description="Multi-method comparison tables for RBF-augmented stencils",
    )
    parser.add_argument(
        "--scheme", choices=["E2", "E4"], default=None,
        help="Scheme to compare (default: both E2 and E4)",
    )
    parser.add_argument(
        "--n-values", default="20,40,80",
        help="Comma-separated grid sizes (default: 20,40,80)",
    )
    parser.add_argument(
        "--update-known-values", action="store_true",
        help="Update known_values.json with discovered optimal parameters",
    )

    args = parser.parse_args(argv)
    n_values = [int(x) for x in args.n_values.split(",")]

    schemes = [args.scheme] if args.scheme else ["E2", "E4"]
    summaries = {}

    for scheme in schemes:
        summaries[scheme] = run_comparison(scheme, n_values)

    if args.update_known_values:
        kv = load_known_values()
        for scheme, summary in summaries.items():
            scheme_key = SCHEME_PARAMS[scheme]["label"]
            if scheme_key not in kv:
                kv[scheme_key] = {}
            kv[scheme_key]["gaussian"] = {
                "epsilon": summary["gaussian_epsilon"],
                "stable_at": summary["stable_at"]["gaussian"],
            }
            kv[scheme_key]["multiquadric"] = {
                "epsilon": summary["multiquadric_epsilon"],
                "stable_at": summary["stable_at"]["multiquadric"],
            }
        save_known_values(kv)
        print(f"\n  Updated known_values.json with comparison results")

    return 0


if __name__ == "__main__":
    sys.exit(main())
