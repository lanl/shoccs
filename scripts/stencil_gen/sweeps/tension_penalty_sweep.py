"""Joint (sigma, gamma) tension + conservation penalty sweep.

Extracted from TestCorrectedTensionPenaltyE4, TestTensionConservationE2,
TestTensionConservationE4 in test_phs.py.

Sweeps both the tension parameter sigma and the conservation penalty gamma
to investigate the stability-conservation trade-off.  For each (sigma, gamma)
pair, builds the penalty-augmented differentiation matrix and evaluates:
  - stability eigenvalue (max Re(eig(-D_bc)))
  - conservation deficit (max |column sum of D|)

Usage:
    uv run python -m sweeps.tension_penalty_sweep --scheme E2
    uv run python -m sweeps.tension_penalty_sweep --scheme E4
    uv run python -m sweeps.tension_penalty_sweep --scheme E4 --n-sigma 5 --n-gamma 5
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from stencil_gen.phs import (
    build_diff_matrix_rbf_penalty,
    stability_eigenvalue_from_matrix,
)

from ._common import SCHEME_PARAMS, STABILITY_TOL, load_known_values, save_known_values
from .gv_objectives import gv_score_from_matrix

# Floating-point eigenvalue solvers return tiny positive real parts (~1e-14)
# for genuinely stable operators.  Use this threshold to distinguish true
# instability from numerical noise.

def eval_point(
    n: int,
    sigma: float,
    gamma: float,
    *,
    p: int,
    q: int,
    nextra: int,
    nu: int,
) -> tuple[float, float, float]:
    """Evaluate a (sigma, gamma) point.

    Returns (stab_eig, deficit, gv_error) where:
      stab_eig = max Re(eig(-D_bc))  (stable means < STABILITY_TOL)
      deficit  = max |column sum of D|  (conservation measure)
      gv_error = max boundary-row group velocity error from the same D matrix
    """
    D = build_diff_matrix_rbf_penalty(
        n, p, q, sigma, "tension", nu, nextra,
        gamma=gamma,
    )
    se = stability_eigenvalue_from_matrix(D)
    deficit = float(np.max(np.abs(np.sum(D, axis=0))))
    gv_error = float(gv_score_from_matrix(D)["max_gv_error"])
    return se, deficit, gv_error


def run_joint_sweep_coarse(
    scheme: str,
    n: int,
    n_sigma: int,
    n_gamma: int,
    sigma_max: float,
) -> dict:
    """Coarse 2D sweep over sigma x gamma.

    Returns a summary dict with best (sigma, gamma) and landscape statistics.
    """
    params = SCHEME_PARAMS[scheme]
    p, q, nextra, nu = params["p"], params["q"], params["nextra"], params["nu"]
    label = params["label"]

    if scheme == "E4":
        # E4: log-spaced sigma [0, sigma_max] matching Phase 32.3b
        sigmas = np.concatenate(
            [[0.0], np.logspace(np.log10(0.01), np.log10(sigma_max), n_sigma)]
        )
    else:
        # E2: linear sigma [0, sigma_max]
        sigmas = np.linspace(0.0, sigma_max, n_sigma)

    gammas = np.concatenate([[0.0], np.logspace(-1, 2, n_gamma)])  # 0 + log[0.1..100]

    best_se = float("inf")
    best_sigma = None
    best_gamma = None
    best_deficit = None

    # Track gamma=0 baseline
    baseline_se = float("inf")
    baseline_deficit = None

    # Track best stable deficit (for E2: at gamma > 0)
    best_stable_deficit = float("inf")
    best_stable_sigma = None
    best_stable_gamma = None

    # Track best stable GV error (feasible-then-minimize, parallel to deficit)
    best_stable_gv = float("inf")
    best_stable_gv_sigma = None
    best_stable_gv_gamma = None

    n_stable = 0
    total = len(sigmas) * len(gammas)

    for sigma in sigmas:
        for gamma in gammas:
            se, deficit, gv = eval_point(
                n, sigma, gamma,
                p=p, q=q, nextra=nextra, nu=nu,
            )

            if se < best_se:
                best_se = se
                best_sigma = sigma
                best_gamma = gamma
                best_deficit = deficit

            if gamma == 0.0 and se < baseline_se:
                baseline_se = se
                baseline_deficit = deficit

            if se < STABILITY_TOL:
                n_stable += 1
                if deficit < best_stable_deficit:
                    best_stable_deficit = deficit
                    best_stable_sigma = sigma
                    best_stable_gamma = gamma
                if gv < best_stable_gv:
                    best_stable_gv = gv
                    best_stable_gv_sigma = sigma
                    best_stable_gv_gamma = gamma

    print(f"\n{'='*72}")
    print(f"  {label} Joint (sigma, gamma) Sweep (n={n})")
    print(f"{'='*72}")
    print(f"  Grid: {len(sigmas)} sigma x {len(gammas)} gamma = {total} points")
    print(f"  Stable points: {n_stable}/{total}")

    print(f"\n  Best (sigma, gamma) — most negative stab_eig:")
    print(f"    sigma*={best_sigma:.4f}, gamma*={best_gamma:.4f}")
    print(f"    stab_eig={best_se:.6e}")
    print(f"    deficit={best_deficit:.6e}")

    print(f"\n  Baseline (gamma=0) best:")
    print(f"    stab_eig={baseline_se:.6e}")
    if baseline_deficit is not None:
        print(f"    deficit={baseline_deficit:.6e}")

    if best_stable_sigma is not None:
        print(f"\n  Best stable point (lowest deficit):")
        print(f"    sigma={best_stable_sigma:.4f}, gamma={best_stable_gamma:.4f}")
        print(f"    deficit={best_stable_deficit:.6e}")

    if best_stable_gv_sigma is not None:
        print(f"\n  Best stable point (lowest GV error):")
        print(f"    sigma={best_stable_gv_sigma:.4f}, gamma={best_stable_gv_gamma:.4f}")
        print(f"    gv_error={best_stable_gv:.6e}")

    return {
        "best_sigma": best_sigma,
        "best_gamma": best_gamma,
        "best_se": best_se,
        "best_deficit": best_deficit,
        "baseline_se": baseline_se,
        "baseline_deficit": baseline_deficit,
        "n_stable": n_stable,
        "total": total,
        "best_stable_sigma": best_stable_sigma,
        "best_stable_gamma": best_stable_gamma,
        "best_stable_deficit": best_stable_deficit,
        "best_stable_gv_sigma": best_stable_gv_sigma,
        "best_stable_gv_gamma": best_stable_gv_gamma,
        "best_stable_gv": best_stable_gv if best_stable_gv_sigma is not None else None,
    }


def run_penalty_effect(
    scheme: str,
    n: int,
    n_gamma: int,
) -> dict:
    """Sweep gamma at the scheme's optimal sigma to check penalty effect.

    E2: sigma*=6.0 (tension optimal).  E4: sigma*=0.0 (PHS k=2 optimal).
    """
    params = SCHEME_PARAMS[scheme]
    p, q, nextra, nu = params["p"], params["q"], params["nextra"], params["nu"]
    label = params["label"]

    sigma_star = 6.0 if scheme == "E2" else 0.0
    gammas = np.concatenate([[0.0], np.logspace(-2, 2, n_gamma)])

    print(f"\n{'='*72}")
    print(f"  {label} Penalty Effect at sigma*={sigma_star} (n={n})")
    print(f"{'='*72}")
    print(f"  {'gamma':>10s}  {'stab_eig':>14s}  {'deficit':>14s}  {'status':>10s}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*10}")

    baseline_se = None
    baseline_deficit = None
    best_se = float("inf")
    best_gamma = None
    max_stable_gamma = -1.0
    deficit_at_max_stable = None

    for gamma in gammas:
        se, deficit, _gv = eval_point(
            n, sigma_star, gamma,
            p=p, q=q, nextra=nextra, nu=nu,
        )
        status = "STABLE" if se < STABILITY_TOL else "unstable"
        # Print a representative subset
        if (gamma == 0.0 or gamma < 0.02
                or abs(gamma - 0.1) < 0.02 or abs(gamma - 1.0) < 0.3
                or abs(gamma - 10.0) < 2.0 or abs(gamma - 100.0) < 20.0):
            print(f"  {gamma:10.4f}  {se:14.6e}  {deficit:14.6e}  {status:>10s}")

        if gamma == 0.0:
            baseline_se = se
            baseline_deficit = deficit
        if se < best_se:
            best_se = se
            best_gamma = gamma
        if se < STABILITY_TOL:
            max_stable_gamma = gamma
            deficit_at_max_stable = deficit

    print(f"\n  Baseline (gamma=0): stab_eig={baseline_se:.6e}")
    if baseline_deficit is not None:
        print(f"    deficit={baseline_deficit:.6e}")
    print(f"  Best gamma={best_gamma:.4f}: stab_eig={best_se:.6e}")
    print(f"  Max gamma with stability: {max_stable_gamma:.4f}")
    if deficit_at_max_stable is not None and baseline_deficit is not None:
        print(f"    deficit at max stable gamma: {deficit_at_max_stable:.6e}")
        if baseline_deficit > 0:
            improvement = 1.0 - deficit_at_max_stable / baseline_deficit
            print(f"    deficit improvement: {improvement:.1%}")

    return {
        "sigma_star": sigma_star,
        "baseline_se": baseline_se,
        "best_gamma": best_gamma,
        "best_se": best_se,
        "max_stable_gamma": max_stable_gamma,
    }


def run_fine_sweep(
    scheme: str,
    n: int,
    n_sigma: int,
    n_gamma: int,
) -> dict:
    """Fine 2D sweep near the scheme's optimal region.

    E2: sigma in [4, 8].  E4: sigma in [0, 5].
    """
    params = SCHEME_PARAMS[scheme]
    p, q, nextra, nu = params["p"], params["q"], params["nextra"], params["nu"]
    label = params["label"]

    if scheme == "E2":
        sigmas = np.linspace(4.0, 8.0, n_sigma)
    else:
        sigmas = np.concatenate([[0.0], np.linspace(0.1, 5.0, n_sigma - 1)])

    gammas = np.concatenate([[0.0], np.logspace(-2 if scheme == "E4" else -1, 2, n_gamma)])

    best_se = float("inf")
    best_sigma = None
    best_gamma = None
    best_deficit = None

    best_se_baseline = float("inf")

    for sigma in sigmas:
        for gamma in gammas:
            se, deficit, _gv = eval_point(
                n, sigma, gamma,
                p=p, q=q, nextra=nextra, nu=nu,
            )
            if se < best_se:
                best_se = se
                best_sigma = sigma
                best_gamma = gamma
                best_deficit = deficit
            if gamma == 0.0 and se < best_se_baseline:
                best_se_baseline = se

    print(f"\n{'='*72}")
    sigma_lo = 4.0 if scheme == "E2" else 0.0
    sigma_hi = 8.0 if scheme == "E2" else 5.0
    print(f"  {label} Fine Joint Sweep: sigma in [{sigma_lo}, {sigma_hi}], gamma in [0, 100]")
    print(f"{'='*72}")
    print(f"  Grid: {len(sigmas)} x {len(gammas)} = {len(sigmas) * len(gammas)} points")

    print(f"\n  Best overall (sigma, gamma):")
    print(f"    sigma*={best_sigma:.4f}, gamma*={best_gamma:.4f}")
    print(f"    stab_eig={best_se:.6e}")
    print(f"    deficit={best_deficit:.6e}")

    print(f"\n  Best gamma=0 baseline: stab_eig={best_se_baseline:.6e}")

    # Grid independence check
    print(f"\n  Grid independence at (sigma*={best_sigma:.4f}, gamma*={best_gamma:.4f}):")
    stable_at = []
    for nn in [20, 40, 80]:
        se, deficit, _gv = eval_point(
            nn, best_sigma, best_gamma,
            p=p, q=q, nextra=nextra, nu=nu,
        )
        status = "STABLE" if se < STABILITY_TOL else "unstable"
        print(f"    n={nn:4d}: stab_eig={se:.6e}, deficit={deficit:.6e} [{status}]")
        if se < STABILITY_TOL:
            stable_at.append(nn)

    return {
        "best_sigma": best_sigma,
        "best_gamma": best_gamma,
        "best_se": best_se,
        "best_deficit": best_deficit,
        "baseline_se": best_se_baseline,
        "stable_at": stable_at,
    }


def run_tension_penalty_sweep(
    scheme: str,
    n_sigma: int,
    n_gamma: int,
    sigma_max: float = 20.0,
) -> dict:
    """Run all three phases: coarse sweep, penalty effect, fine sweep.

    Returns a summary dict for known_values.json updates.
    """
    n = 40  # primary grid size, matching test classes

    params = SCHEME_PARAMS[scheme]
    p, q, nextra, nu = params["p"], params["q"], params["nextra"], params["nu"]

    coarse = run_joint_sweep_coarse(scheme, n, n_sigma, n_gamma, sigma_max)

    # Cross-grid stability re-check at the GV-optimal (sigma, gamma) pair.
    # The coarse sweep is single-grid (n=40 only); without this loop the
    # persisted tension_penalty_gv entry would have no stable_at field.
    gv_sigma = coarse["best_stable_gv_sigma"]
    gv_gamma = coarse["best_stable_gv_gamma"]
    gv_stable_at: list[int] | None = None
    if gv_sigma is not None:
        gv_stable_at = []
        print(
            f"\n  Checking GV-optimal (sigma={gv_sigma:.6f}, gamma={gv_gamma:.6f}) "
            f"across grid sizes:"
        )
        for nn in sorted({20, 40, 80, 160}):
            se, deficit, _gv = eval_point(
                nn, gv_sigma, gv_gamma,
                p=p, q=q, nextra=nextra, nu=nu,
            )
            status = "STABLE" if se < STABILITY_TOL else "unstable"
            print(f"    n={nn:4d}: stab_eig={se:.6e}, deficit={deficit:.6e} [{status}]")
            if se < STABILITY_TOL:
                gv_stable_at.append(nn)

    penalty = run_penalty_effect(scheme, n, n_gamma)
    fine = run_fine_sweep(scheme, n, n_sigma, n_gamma)

    # 40.8g: the persisted (param, gv_error) pair must be bit-exact self-
    # consistent at the *rounded* parameter.  Mirror 40.8c (semantic fix) +
    # 40.8d (bit-level fix) for the other three sweeps: round sigma/gamma
    # first and re-evaluate GV at the rounded values for both the primary
    # (stability-optimum) and secondary (GV-optimum) entries.  eval_point
    # already returns (stab_eig, deficit, gv_error) per 40.4a.
    best_sigma_rounded = round(float(fine["best_sigma"]), 6)
    best_gamma_rounded = round(float(fine["best_gamma"]), 6)
    _se, _def, gv_at_stability = eval_point(
        n, best_sigma_rounded, best_gamma_rounded,
        p=p, q=q, nextra=nextra, nu=nu,
    )

    if gv_sigma is not None:
        gv_sigma_rounded = round(float(gv_sigma), 6)
        gv_gamma_rounded = round(float(gv_gamma), 6)
        _se, _def, gv_at_gv = eval_point(
            n, gv_sigma_rounded, gv_gamma_rounded,
            p=p, q=q, nextra=nextra, nu=nu,
        )
    else:
        gv_sigma_rounded = None
        gv_gamma_rounded = None
        gv_at_gv = None

    return {
        "best_sigma": best_sigma_rounded,
        "best_gamma": best_gamma_rounded,
        "stable_at": fine["stable_at"],
        "baseline_stable": coarse["baseline_se"] < STABILITY_TOL,
        "max_stable_gamma": penalty["max_stable_gamma"],
        "gv_sigma": gv_sigma_rounded,
        "gv_gamma": gv_gamma_rounded,
        "gv_error_at_stability": gv_at_stability,
        "gv_error_at_gv": gv_at_gv,
        "gv_stable_at": gv_stable_at,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.tension_penalty_sweep",
        description="Joint (sigma, gamma) tension + conservation penalty sweep",
    )
    parser.add_argument("--scheme", choices=["E2", "E4"], required=True)
    parser.add_argument(
        "--n-sigma", type=int, default=25,
        help="Number of sigma sample points (default: 25)",
    )
    parser.add_argument(
        "--n-gamma", type=int, default=25,
        help="Number of gamma sample points (default: 25)",
    )
    parser.add_argument(
        "--sigma-max", type=float, default=20.0,
        help="Maximum sigma value (default: 20.0)",
    )
    parser.add_argument(
        "--update-known-values", action="store_true",
        help="Update known_values.json with discovered optimal (sigma, gamma)",
    )

    args = parser.parse_args(argv)

    summary = run_tension_penalty_sweep(
        args.scheme, args.n_sigma, args.n_gamma, args.sigma_max,
    )

    if args.update_known_values:
        kv = load_known_values()
        scheme_key = SCHEME_PARAMS[args.scheme]["label"]
        if scheme_key not in kv:
            kv[scheme_key] = {}
        # Merge into the existing tension_penalty entry so that keys written
        # by an earlier or future invocation survive this write (mirrors the
        # 40.2d/40.3c additive-merge pattern).
        tp_entry = dict(kv[scheme_key].get("tension_penalty", {}))
        tp_entry["sigma"] = summary["best_sigma"]
        tp_entry["gamma"] = summary["best_gamma"]
        tp_entry["stable_at"] = summary["stable_at"]
        if summary["gv_error_at_stability"] is not None:
            tp_entry["gv_error"] = summary["gv_error_at_stability"]
        kv[scheme_key]["tension_penalty"] = tp_entry
        updated_keys = [f"{scheme_key}.tension_penalty"]
        if summary["gv_sigma"] is not None:
            tp_gv_entry = dict(kv[scheme_key].get("tension_penalty_gv", {}))
            tp_gv_entry["sigma"] = summary["gv_sigma"]
            tp_gv_entry["gamma"] = summary["gv_gamma"]
            tp_gv_entry["gv_error"] = summary["gv_error_at_gv"]
            tp_gv_entry["stable_at"] = summary["gv_stable_at"]
            kv[scheme_key]["tension_penalty_gv"] = tp_gv_entry
            updated_keys.append(f"{scheme_key}.tension_penalty_gv")
        save_known_values(kv)
        print(f"\n  Updated known_values.json: {', '.join(updated_keys)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
