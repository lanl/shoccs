"""Tension spline sigma parameter sweep for RBF-augmented stencils.

Extracted from TestCorrectedTensionE2, TestCorrectedTensionE4,
TestTensionSweepE2, TestTensionSweepE4 in test_phs.py.

Sweeps the tension parameter sigma over a range including sigma=0
(PHS k=2 limit) and reports stability of the resulting differentiation
matrix at each grid size. The persisted ``tension`` regression entry is
restricted to ``sigma >= --sigma-floor`` (default 1.0) so it stays
strictly above the PHS k=2 limit at ``sigma=0`` — at floors below 1.0
the constrained E4 fine-sweep optimum lands exactly on the floor
(``sigma_floor=0.1`` and ``0.5`` both produce floor-snap), which
collapses the ``E4_1.tension`` regression entry into a near-duplicate
of ``E4_1.phs_k2`` (see plan 46.3a.2). Pass ``--sigma-floor 0.0`` to
permit the PHS k=2 limit as the optimum.

Usage:
    uv run python -m sweeps.tension_sweep --scheme E2
    uv run python -m sweeps.tension_sweep --scheme E4
    uv run python -m sweeps.tension_sweep --scheme E2 --n-sigma 10  # quick smoke test
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from stencil_gen.phs import build_diff_matrix_rbf, stability_eigenvalue

from ._common import (
    SCHEME_PARAMS,
    STABILITY_TOL,
    load_known_values,
    print_sweep_table,
    report_stable_ranges,
    save_known_values,
)
from .gv_objectives import boundary_gv_error_max, print_gks_advisory

# Floating-point eigenvalue solvers return tiny positive real parts (~1e-14)
# for genuinely stable operators.  Use this threshold to distinguish true
# instability from numerical noise.

# CLI default for --sigma-floor. Exposed as a module-level constant so the
# regression test ``test_e4_tension_sigma_strictly_above_floor`` can assert
# that the persisted tension entry stays strictly above the floor (plan
# 46.3a.2). Empirically the E4 constrained fine-sweep optimum snaps to the
# floor for floors below 1.0; only at floor=1.0 does it land strictly above.
CLI_DEFAULT_SIGMA_FLOOR = 1.0

def sweep_stability(
    n_values: list[int],
    sigmas: np.ndarray,
    *,
    p: int,
    q: int,
    nextra: int,
    nu: int,
    include_gv: bool = False,
) -> tuple[
    dict[int, list[tuple[float, float]]],
    dict[float, float] | None,
]:
    """Run sigma sweep using stability_eigenvalue with tension kernel.

    Returns ``(stab_results, gv_by_sigma)`` where ``stab_results`` maps
    ``n -> list of (sigma, stab_eig)`` and ``gv_by_sigma`` maps
    ``sigma -> boundary_gv_error_max`` (independent of ``n``) when
    ``include_gv`` is set, else ``None``.
    """
    results: dict[int, list[tuple[float, float]]] = {}
    gv_by_sigma: dict[float, float] | None = {} if include_gv else None
    if include_gv:
        for sigma in sigmas:
            gv_by_sigma[float(sigma)] = boundary_gv_error_max(
                p=p, q=q, nextra=nextra, nu=nu,
                sigma=float(sigma), kernel="tension",
            )
    for n in n_values:
        rows = []
        for sigma in sigmas:
            se = stability_eigenvalue(
                n, p=p, q=q, epsilon=sigma,
                kernel="tension", nu=nu, nextra=nextra,
            )
            rows.append((float(sigma), se))
        results[n] = rows
    return results, gv_by_sigma


def fine_sweep(
    n: int,
    sigmas_coarse: np.ndarray,
    *,
    p: int,
    q: int,
    nextra: int,
    nu: int,
    n_fine: int = 200,
    sigma_floor: float = 0.0,
) -> tuple[float, float, float, float]:
    """Coarse-then-fine sweep at a single grid size.

    ``sigma_floor`` restricts both the coarse and fine searches to
    ``sigma >= sigma_floor``. This is used to prevent the persisted
    ``tension.sigma`` from collapsing to the PHS k=2 limit (sigma=0),
    which would make the ``tension`` regression entry structurally
    identical to ``phs_k2`` (see plan 46.3a.1). Default 0.0 preserves
    the historical library behavior; the CLI sets a non-zero default.

    When ``sigma_floor > 0`` and the constrained fine-sweep optimum
    lands *exactly* on the floor (within a few × ``np.spacing(sigma_floor)``),
    the function emits a ``UserWarning``: this signals that the persisted
    optimum is determined by the floor rather than by the underlying
    objective and the test/regression entry is structurally similar to
    the PHS k=2 limit (see plan 46.3a.2). Callers passing
    ``sigma_floor=0.0`` deliberately permit ``sigma=0`` and do not get
    the warning.

    Returns (best_coarse_sigma, best_coarse_se, best_fine_sigma, best_fine_se).
    """
    coarse = []
    for sigma in sigmas_coarse:
        se = stability_eigenvalue(
            n, p=p, q=q, epsilon=sigma,
            kernel="tension", nu=nu, nextra=nextra,
        )
        coarse.append((float(sigma), se))

    coarse_search = [r for r in coarse if r[0] >= sigma_floor] or coarse
    best_coarse = min(coarse_search, key=lambda r: r[1])
    sigma_best = best_coarse[0]

    # Fine sweep: ±factor around best (or [sigma_floor, 2.0] if best is near floor)
    if sigma_best < max(sigma_floor, 0.1):
        lo, hi = sigma_floor, 2.0
    else:
        lo = max(sigma_floor, sigma_best / 5)
        hi = sigma_best * 5
    sigmas_fine = np.linspace(lo, hi, n_fine)
    fine = []
    for sigma in sigmas_fine:
        se = stability_eigenvalue(
            n, p=p, q=q, epsilon=sigma,
            kernel="tension", nu=nu, nextra=nextra,
        )
        fine.append((float(sigma), se))

    fine_search = [r for r in fine if r[0] >= sigma_floor] or fine
    best_fine = min(fine_search, key=lambda r: r[1])

    if sigma_floor > 0.0:
        snap_tol = max(4 * np.spacing(sigma_floor), 1e-12)
        if abs(best_fine[0] - sigma_floor) <= snap_tol:
            import warnings
            warnings.warn(
                f"fine_sweep: constrained optimum sigma={best_fine[0]:.6g} "
                f"snapped to sigma_floor={sigma_floor:.6g}; the persisted "
                f"tension.sigma is determined by the floor rather than the "
                f"objective. Consider raising --sigma-floor (plan 46.3a.2).",
                UserWarning,
                stacklevel=2,
            )

    return best_coarse[0], best_coarse[1], best_fine[0], best_fine[1]


def run_tension_sweep(
    scheme: str,
    n_values: list[int],
    n_sigma: int,
    sigma_max: float = 20.0,
    *,
    include_gv: bool = False,
    check_gks: bool = False,
    sigma_floor: float = 0.0,
) -> dict:
    """Run a full tension sigma sweep for a scheme.

    ``sigma_floor`` is forwarded to :func:`fine_sweep` to keep the
    persisted ``tension.sigma`` strictly above the PHS k=2 limit; see
    that function for details.

    Returns a summary dict with best sigma and stable grid sizes.
    """
    params = SCHEME_PARAMS[scheme]
    p, q, nextra, nu = params["p"], params["q"], params["nextra"], params["nu"]
    label = params["label"]

    # Include sigma=0 (PHS k=2 limit) plus logarithmic spacing for sigma > 0
    sigmas = np.concatenate(
        [[0.0], np.logspace(np.log10(0.01), np.log10(sigma_max), n_sigma)]
    )

    # Main sweep
    results, gv_by_sigma = sweep_stability(
        n_values, sigmas,
        p=p, q=q, nextra=nextra, nu=nu,
        include_gv=include_gv,
    )
    print_sweep_table(
        f"{label} Tension Spline — Stability Sweep (p={p}, q={q}, nextra={nextra})",
        results,
        param_label="sigma",
        gv_by_param=gv_by_sigma,
    )
    print()
    report_stable_ranges(results, param_label="sigma")

    gv_best_sigma: float | None = None
    gv_best_error: float | None = None
    if gv_by_sigma is not None:
        # Among feasible (stable at every grid size in the sweep) sigmas,
        # report the smallest GV error.  This does not alter the stability
        # optimum reported below — it is purely advisory secondary output.
        feasible_sigmas: set[float] = set(gv_by_sigma)
        for n, rows in results.items():
            stable_for_n = {
                float(s) for s, se in rows if se < STABILITY_TOL
            }
            feasible_sigmas &= stable_for_n
        if feasible_sigmas:
            gv_best_sigma = min(feasible_sigmas, key=lambda s: gv_by_sigma[s])
            gv_best_error = gv_by_sigma[gv_best_sigma]
            print(
                f"  Best feasible by GV error: sigma={gv_best_sigma:.6f}, "
                f"gv_err={gv_best_error:.6e}"
            )
        else:
            print("  Best feasible by GV error: (no sigma stable at every grid size)")

    # Fine sweep at n=40 (or largest available)
    n_fine_grid = 40 if 40 in n_values else max(n_values)
    coarse_sigma, coarse_se, fine_sigma, fine_se = fine_sweep(
        n_fine_grid, sigmas,
        p=p, q=q, nextra=nextra, nu=nu,
        sigma_floor=sigma_floor,
    )
    stable = fine_se < STABILITY_TOL
    print(f"\n  Fine sweep (n={n_fine_grid}):")
    print(f"  Coarse best: sigma={coarse_sigma:.6f}, stab_eig={coarse_se:.6e}")
    print(f"  Fine best:   sigma={fine_sigma:.6f}, stab_eig={fine_se:.6e}")
    print(f"  Stable: {stable}")

    # Check fine-sweep best across all grid sizes
    sigma_star = fine_sigma
    stable_at = []
    print(f"\n  Checking sigma*={sigma_star:.6f} across grid sizes:")
    for nn in sorted(set(n_values + [20, 40, 80, 160])):
        se = stability_eigenvalue(
            nn, p=p, q=q, epsilon=sigma_star,
            kernel="tension", nu=nu, nextra=nextra,
        )
        status = "STABLE" if se < STABILITY_TOL else "unstable"
        print(f"    n={nn:4d}: stab_eig={se:.6e} [{status}]")
        if se < STABILITY_TOL:
            stable_at.append(nn)

    # GV error at the stability-optimum sigma (sigma_star). This is what
    # the additive ``tension.gv_error`` field must hold so that the
    # ``(sigma, gv_error)`` pair describes a single point. Evaluate at the
    # rounded sigma so the persisted pair is exactly self-consistent with
    # the rounded ``tension.sigma`` field.
    sigma_star_rounded = round(float(sigma_star), 6)
    gv_at_sigma_star: float | None = None
    if include_gv:
        gv_at_sigma_star = boundary_gv_error_max(
            p=p, q=q, nextra=nextra, nu=nu,
            sigma=sigma_star_rounded, kernel="tension",
        )

    # Cross-check GV-optimal sigma at the same grid sizes as sigma_star.
    gv_stable_at: list[int] | None = None
    if gv_best_sigma is not None:
        gv_stable_at = []
        print(f"\n  Checking GV-optimal sigma={gv_best_sigma:.6f} across grid sizes:")
        for nn in sorted(set(n_values + [20, 40, 80, 160])):
            se = stability_eigenvalue(
                nn, p=p, q=q, epsilon=gv_best_sigma,
                kernel="tension", nu=nu, nextra=nextra,
            )
            status = "STABLE" if se < STABILITY_TOL else "unstable"
            print(f"    n={nn:4d}: stab_eig={se:.6e} [{status}]")
            if se < STABILITY_TOL:
                gv_stable_at.append(nn)

    if check_gks:
        D_star = build_diff_matrix_rbf(
            n_fine_grid, p, q, sigma_star, "tension", nu, nextra,
        )
        print_gks_advisory(
            D_star, label=f"n={n_fine_grid}, sigma*={sigma_star:.6f}",
        )

    gv_sigma_rounded = (
        round(float(gv_best_sigma), 6) if gv_best_sigma is not None else None
    )
    gv_error_at_rounded: float | None = None
    if include_gv and gv_sigma_rounded is not None:
        gv_error_at_rounded = boundary_gv_error_max(
            p=p, q=q, nextra=nextra, nu=nu,
            sigma=gv_sigma_rounded, kernel="tension",
        )

    return {
        "sigma": sigma_star_rounded,
        "stable_at": stable_at,
        "fine_stab_eig": fine_se,
        "gv_by_sigma": gv_by_sigma,
        "gv_at_sigma_star": gv_at_sigma_star,
        "gv_sigma": gv_sigma_rounded,
        "gv_error": gv_error_at_rounded if gv_error_at_rounded is not None else gv_best_error,
        "gv_stable_at": gv_stable_at,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.tension_sweep",
        description="Tension spline sigma parameter sweep for RBF-augmented stencils",
    )
    parser.add_argument("--scheme", choices=["E2", "E4"], required=True)
    parser.add_argument(
        "--n-values", default="20,40,80",
        help="Comma-separated grid sizes (default: 20,40,80)",
    )
    parser.add_argument(
        "--n-sigma", type=int, default=61,
        help="Number of sigma sample points in coarse sweep (default: 61)",
    )
    parser.add_argument(
        "--sigma-max", type=float, default=20.0,
        help="Maximum sigma value in coarse sweep (default: 20.0)",
    )
    parser.add_argument(
        "--update-known-values", action="store_true",
        help="Update known_values.json with discovered optimal sigma",
    )
    parser.add_argument(
        "--include-gv", action="store_true",
        help="Also compute boundary group-velocity error at each sigma "
             "(advisory secondary objective; does not alter the stability optimum)",
    )
    parser.add_argument(
        "--check-gks", action="store_true",
        help="After picking the stability optimum, run gks_group_velocity_check "
             "on D at sigma* and print any outgoing boundary modes as WARNINGs "
             "(advisory only; necessary-not-sufficient for instability)",
    )
    parser.add_argument(
        "--sigma-floor", type=float, default=CLI_DEFAULT_SIGMA_FLOOR,
        help=f"Restrict the fine-sweep search to sigma >= sigma_floor "
             f"(default {CLI_DEFAULT_SIGMA_FLOOR}) so the persisted tension "
             f"entry stays strictly above the PHS k=2 limit (sigma=0). "
             f"Empirically the E4 constrained optimum lands exactly on the "
             f"floor for floors below 1.0; only at floor=1.0 does it land "
             f"strictly above (see plan 46.3a.2). Pass 0.0 to allow sigma=0.",
    )

    args = parser.parse_args(argv)
    n_values = [int(x) for x in args.n_values.split(",")]

    summary = run_tension_sweep(
        args.scheme, n_values, args.n_sigma, args.sigma_max,
        include_gv=args.include_gv,
        check_gks=args.check_gks,
        sigma_floor=args.sigma_floor,
    )

    if args.update_known_values:
        kv = load_known_values()
        scheme_key = SCHEME_PARAMS[args.scheme]["label"]
        if scheme_key not in kv:
            kv[scheme_key] = {}
        # Merge into the existing tension entry so that keys written by an
        # earlier --include-gv run survive a subsequent non-GV invocation.
        tension_entry = dict(kv[scheme_key].get("tension", {}))
        tension_entry["sigma"] = summary["sigma"]
        tension_entry["stable_at"] = summary["stable_at"]
        # The additive ``tension.gv_error`` must represent the GV at
        # ``tension.sigma`` (the stability optimum), not the GV-optimum
        # sigma — otherwise ``(sigma, gv_error)`` describe two different
        # points and the ``test_scheme_primary_gv_error_match`` regression
        # test fails. The GV at the GV-optimum sigma is persisted on the
        # secondary ``tension_gv`` entry below.
        if args.include_gv and summary["gv_at_sigma_star"] is not None:
            tension_entry["gv_error"] = summary["gv_at_sigma_star"]
        kv[scheme_key]["tension"] = tension_entry
        updated_keys = [f"{scheme_key}.tension"]
        if args.include_gv and summary["gv_sigma"] is not None:
            tension_gv_entry = dict(kv[scheme_key].get("tension_gv", {}))
            tension_gv_entry["sigma"] = summary["gv_sigma"]
            tension_gv_entry["gv_error"] = summary["gv_error"]
            tension_gv_entry["stable_at"] = summary["gv_stable_at"]
            kv[scheme_key]["tension_gv"] = tension_gv_entry
            updated_keys.append(f"{scheme_key}.tension_gv")
        save_known_values(kv)
        print(f"\n  Updated known_values.json: {', '.join(updated_keys)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
