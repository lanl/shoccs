"""Epsilon (Gaussian/Multiquadric) parameter sweep for RBF-augmented stencils.

Extracted from TestCorrectedSweepE2, TestCorrectedSweepE4, TestEpsilonSweepE2,
TestEpsilonSweepE4 in test_phs.py.

Sweeps the shape parameter epsilon over a log-spaced range and reports
stability of the resulting differentiation matrix at each grid size.

The persisted ``gaussian`` regression entry is restricted to
``epsilon >= --eps-floor`` (default 1.5) so it stays strictly above the
``eps -> 0`` polynomial-reproduction (degenerate-kernel) limit and the
narrow lower stable basins exposed by under-sampled coarse grids.
Empirically (plan 46.3b.1a) the gaussian ``stab_eig(eps)`` landscape
contains multiple disjoint stable basins; floor=1.5 places the
constrained fine-sweep optimum strictly inside the upper basin for both
E2 (``eps≈3.55``) and E4 (``eps≈2.10``). Pass ``--eps-floor 0.0`` to
permit the unconstrained grid-min (which may snap to a degenerate-kernel
limit). The floor is gaussian-specific in motivation but applies to all
kernels for consistency; multiquadric and other kernels are not known
to suffer from boundary-snap with floor=1.5 in the standard
``[0.01, 10]`` range.

Usage:
    uv run python -m sweeps.epsilon_sweep --scheme E2 --kernel gaussian
    uv run python -m sweeps.epsilon_sweep --scheme E4 --kernel multiquadric
    uv run python -m sweeps.epsilon_sweep --scheme E2 --n-eps 10  # quick smoke test
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from stencil_gen.phs import (
    build_diff_matrix_rbf,
    stability_eigenvalue,
    stability_eigenvalue_from_matrix,
)

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

# CLI default for --eps-floor. Exposed as a module-level constant so the
# regression test ``test_e{2,4}_gaussian_epsilon_strictly_above_floor`` can
# assert the persisted gaussian entry stays strictly above the floor (plan
# 46.3b.1.2). Empirically (plan 46.3b.1a) the gaussian stab_eig(eps)
# landscape has multiple disjoint stable basins; floor=1.5 yields
# strictly-interior optima for both E2 (~3.55) and E4 (~2.10).
CLI_DEFAULT_EPS_FLOOR = 1.5


def sweep_stability(
    kernel: str,
    n_values: list[int],
    epsilons: np.ndarray,
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
    """Run epsilon sweep using stability_eigenvalue.

    Returns ``(stab_results, gv_by_eps)`` where ``stab_results`` maps
    ``n -> list of (eps, stab_eig)`` and ``gv_by_eps`` maps
    ``eps -> boundary_gv_error_max`` (independent of ``n``) when
    ``include_gv`` is set, else ``None``.
    """
    results: dict[int, list[tuple[float, float]]] = {}
    gv_by_eps: dict[float, float] | None = {} if include_gv else None
    if include_gv:
        for eps in epsilons:
            gv_by_eps[float(eps)] = boundary_gv_error_max(
                p=p, q=q, nextra=nextra, nu=nu,
                sigma=float(eps), kernel=kernel,
            )
    for n in n_values:
        rows = []
        for eps in epsilons:
            se = stability_eigenvalue(
                n, p=p, q=q, epsilon=eps,
                kernel=kernel, nu=nu, nextra=nextra,
            )
            rows.append((float(eps), se))
        results[n] = rows
    return results, gv_by_eps


def fine_sweep(
    n: int,
    kernel: str,
    epsilons_coarse: np.ndarray,
    *,
    p: int,
    q: int,
    nextra: int,
    nu: int,
    n_fine: int = 200,
    eps_floor: float = 0.0,
) -> tuple[float, float, float, float]:
    """Coarse-then-fine sweep at a single grid size.

    ``eps_floor`` restricts both the coarse and fine searches to
    ``epsilon >= eps_floor``. This is used to prevent the persisted
    ``gaussian.epsilon`` from collapsing to the ``eps -> 0``
    polynomial-reproduction (degenerate-kernel) limit or to a narrow
    lower stable basin produced by an under-sampled coarse grid (see
    plan 46.3b.1a). Default 0.0 preserves the historical library
    behavior; the CLI sets a non-zero default.

    When ``eps_floor > 0`` and the constrained fine-sweep optimum
    lands *exactly* on the floor (within a few × ``np.spacing(eps_floor)``),
    the function emits a ``UserWarning``: this signals that the persisted
    optimum is determined by the floor rather than by the underlying
    objective and the regression entry may be structurally degenerate
    (see plan 46.3b.1.2). Callers passing ``eps_floor=0.0`` deliberately
    permit the unconstrained grid-min and do not get the warning.

    Returns (best_coarse_eps, best_coarse_se, best_fine_eps, best_fine_se).
    """
    coarse = []
    for eps in epsilons_coarse:
        se = stability_eigenvalue(
            n, p=p, q=q, epsilon=eps,
            kernel=kernel, nu=nu, nextra=nextra,
        )
        coarse.append((float(eps), se))

    coarse_search = [r for r in coarse if r[0] >= eps_floor] or coarse
    best_coarse = min(coarse_search, key=lambda r: r[1])
    eps_best = best_coarse[0]

    # Fine sweep: ±1 decade around best, clamped to [eps_floor, 100]
    lo = max(eps_floor, 0.001, eps_best / 10)
    hi = min(100, eps_best * 10)
    epsilons_fine = np.linspace(lo, hi, n_fine)
    fine = []
    for eps in epsilons_fine:
        se = stability_eigenvalue(
            n, p=p, q=q, epsilon=eps,
            kernel=kernel, nu=nu, nextra=nextra,
        )
        fine.append((float(eps), se))

    fine_search = [r for r in fine if r[0] >= eps_floor] or fine
    best_fine = min(fine_search, key=lambda r: r[1])

    if eps_floor > 0.0:
        snap_tol = max(4 * np.spacing(eps_floor), 1e-12)
        if abs(best_fine[0] - eps_floor) <= snap_tol:
            import warnings
            warnings.warn(
                f"fine_sweep: constrained optimum epsilon={best_fine[0]:.6g} "
                f"snapped to eps_floor={eps_floor:.6g}; the persisted "
                f"{kernel}.epsilon is determined by the floor rather than "
                f"the objective. Consider raising --eps-floor (plan 46.3b.1.2).",
                UserWarning,
                stacklevel=2,
            )

    return best_coarse[0], best_coarse[1], best_fine[0], best_fine[1]


def run_epsilon_sweep(
    scheme: str,
    kernel: str,
    n_values: list[int],
    n_eps: int,
    *,
    include_gv: bool = False,
    check_gks: bool = False,
    eps_floor: float = 0.0,
) -> dict:
    """Run a full epsilon sweep for a scheme/kernel combination.

    ``eps_floor`` is forwarded to :func:`fine_sweep` to keep the
    persisted ``{kernel}.epsilon`` strictly above the eps -> 0
    degenerate-kernel limit; see that function for details.

    Returns a summary dict with best epsilon and stable grid sizes.
    """
    params = SCHEME_PARAMS[scheme]
    p, q, nextra, nu = params["p"], params["q"], params["nextra"], params["nu"]
    label = params["label"]

    epsilons = np.logspace(np.log10(0.01), np.log10(10), n_eps)

    # Main sweep
    results, gv_by_eps = sweep_stability(
        kernel, n_values, epsilons,
        p=p, q=q, nextra=nextra, nu=nu,
        include_gv=include_gv,
    )
    print_sweep_table(
        f"{label} {kernel.capitalize()} — Stability Sweep (p={p}, q={q}, nextra={nextra})",
        results,
        param_label="epsilon",
        gv_by_param=gv_by_eps,
    )
    print()
    report_stable_ranges(results, param_label="epsilon")

    gv_best_eps: float | None = None
    gv_best_error: float | None = None
    if gv_by_eps is not None:
        # Among feasible (stable at every grid size in the sweep) epsilons,
        # report the smallest GV error.  This does not alter the stability
        # optimum reported below — it is purely advisory secondary output.
        feasible_eps: set[float] = set(gv_by_eps)
        for n, rows in results.items():
            stable_for_n = {
                float(e) for e, se in rows if se < STABILITY_TOL
            }
            feasible_eps &= stable_for_n
        if feasible_eps:
            gv_best_eps = min(feasible_eps, key=lambda e: gv_by_eps[e])
            gv_best_error = gv_by_eps[gv_best_eps]
            print(
                f"  Best feasible by GV error: eps={gv_best_eps:.6f}, "
                f"gv_err={gv_best_error:.6e}"
            )
        else:
            print("  Best feasible by GV error: (no eps stable at every grid size)")

    # Fine sweep at n=40 (or largest available)
    n_fine_grid = 40 if 40 in n_values else max(n_values)
    coarse_eps, coarse_se, fine_eps, fine_se = fine_sweep(
        n_fine_grid, kernel, epsilons,
        p=p, q=q, nextra=nextra, nu=nu,
        eps_floor=eps_floor,
    )
    stable = fine_se < STABILITY_TOL
    print(f"\n  Fine sweep (n={n_fine_grid}):")
    print(f"  Coarse best: eps={coarse_eps:.6f}, stab_eig={coarse_se:.6e}")
    print(f"  Fine best:   eps={fine_eps:.6f}, stab_eig={fine_se:.6e}")
    print(f"  Stable: {stable}")

    # Check fine-sweep best across all grid sizes
    eps_star = fine_eps
    stable_at = []
    print(f"\n  Checking eps*={eps_star:.6f} across grid sizes:")
    for nn in sorted(set(n_values + [20, 40, 80, 160])):
        se = stability_eigenvalue(
            nn, p=p, q=q, epsilon=eps_star,
            kernel=kernel, nu=nu, nextra=nextra,
        )
        status = "STABLE" if se < STABILITY_TOL else "unstable"
        print(f"    n={nn:4d}: stab_eig={se:.6e} [{status}]")
        if se < STABILITY_TOL:
            stable_at.append(nn)

    # GV error at the stability-optimum epsilon (eps_star). This is what
    # the additive ``{kernel}.gv_error`` field must hold so that the
    # ``(epsilon, gv_error)`` pair describes a single point. Evaluate at
    # the rounded epsilon so the persisted pair is exactly self-consistent
    # with the rounded ``{kernel}.epsilon`` field.
    eps_star_rounded = round(float(eps_star), 6)
    gv_at_eps_star: float | None = None
    if include_gv:
        gv_at_eps_star = boundary_gv_error_max(
            p=p, q=q, nextra=nextra, nu=nu,
            sigma=eps_star_rounded, kernel=kernel,
        )

    # Cross-check GV-optimal epsilon at the same grid sizes as eps_star.
    gv_stable_at: list[int] | None = None
    if gv_best_eps is not None:
        gv_stable_at = []
        print(f"\n  Checking GV-optimal eps={gv_best_eps:.6f} across grid sizes:")
        for nn in sorted(set(n_values + [20, 40, 80, 160])):
            se = stability_eigenvalue(
                nn, p=p, q=q, epsilon=gv_best_eps,
                kernel=kernel, nu=nu, nextra=nextra,
            )
            status = "STABLE" if se < STABILITY_TOL else "unstable"
            print(f"    n={nn:4d}: stab_eig={se:.6e} [{status}]")
            if se < STABILITY_TOL:
                gv_stable_at.append(nn)

    if check_gks:
        D_star = build_diff_matrix_rbf(
            n_fine_grid, p, q, eps_star, kernel, nu, nextra,
        )
        print_gks_advisory(
            D_star, label=f"n={n_fine_grid}, eps*={eps_star:.6f}, kernel={kernel}",
        )

    gv_eps_rounded = (
        round(float(gv_best_eps), 6) if gv_best_eps is not None else None
    )
    gv_error_at_rounded: float | None = None
    if include_gv and gv_eps_rounded is not None:
        gv_error_at_rounded = boundary_gv_error_max(
            p=p, q=q, nextra=nextra, nu=nu,
            sigma=gv_eps_rounded, kernel=kernel,
        )

    return {
        "epsilon": eps_star_rounded,
        "stable_at": stable_at,
        "fine_stab_eig": fine_se,
        "gv_by_eps": gv_by_eps,
        "gv_at_eps_star": gv_at_eps_star,
        "gv_epsilon": gv_eps_rounded,
        "gv_error": gv_error_at_rounded if gv_error_at_rounded is not None else gv_best_error,
        "gv_stable_at": gv_stable_at,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.epsilon_sweep",
        description="Epsilon (Gaussian/MQ) parameter sweep for RBF-augmented stencils",
    )
    parser.add_argument("--scheme", choices=["E2", "E4"], required=True)
    parser.add_argument(
        "--kernel", choices=["gaussian", "multiquadric"], default="gaussian",
    )
    parser.add_argument(
        "--n-values", default="20,40,80",
        help="Comma-separated grid sizes (default: 20,40,80)",
    )
    parser.add_argument(
        "--n-eps", type=int, default=60,
        help="Number of epsilon sample points in coarse sweep (default: 60)",
    )
    parser.add_argument(
        "--update-known-values", action="store_true",
        help="Update known_values.json with discovered optimal epsilon",
    )
    parser.add_argument(
        "--include-gv", action="store_true",
        help="Also compute boundary group-velocity error at each epsilon "
             "(advisory secondary objective; does not alter the stability optimum)",
    )
    parser.add_argument(
        "--check-gks", action="store_true",
        help="After picking the stability optimum, run gks_group_velocity_check "
             "on D at eps* and print any outgoing boundary modes as WARNINGs "
             "(advisory only; necessary-not-sufficient for instability)",
    )
    parser.add_argument(
        "--eps-floor", type=float, default=CLI_DEFAULT_EPS_FLOOR,
        help=f"Restrict the fine-sweep search to epsilon >= eps_floor "
             f"(default {CLI_DEFAULT_EPS_FLOOR}) so the persisted gaussian "
             f"entry stays strictly above the eps -> 0 polynomial-reproduction "
             f"limit. Empirically (plan 46.3b.1a) gaussian stab_eig(eps) has "
             f"multiple disjoint stable basins; floor=1.5 yields strictly "
             f"interior optima for both E2 (~3.55) and E4 (~2.10). Pass 0.0 "
             f"to allow the unconstrained grid-min.",
    )

    args = parser.parse_args(argv)
    n_values = [int(x) for x in args.n_values.split(",")]

    summary = run_epsilon_sweep(
        args.scheme, args.kernel, n_values, args.n_eps,
        include_gv=args.include_gv,
        check_gks=args.check_gks,
        eps_floor=args.eps_floor,
    )

    if args.update_known_values:
        kv = load_known_values()
        scheme_key = SCHEME_PARAMS[args.scheme]["label"]
        if scheme_key not in kv:
            kv[scheme_key] = {}
        # Merge into the existing kernel entry so that keys written by an
        # earlier --include-gv run survive a subsequent non-GV invocation.
        kernel_entry = dict(kv[scheme_key].get(args.kernel, {}))
        kernel_entry["epsilon"] = summary["epsilon"]
        kernel_entry["stable_at"] = summary["stable_at"]
        # The additive ``{kernel}.gv_error`` must represent the GV at
        # ``{kernel}.epsilon`` (the stability optimum), not the GV-optimum
        # epsilon — otherwise ``(epsilon, gv_error)`` describe two different
        # points and the ``test_scheme_primary_gv_error_match`` regression
        # test fails. The GV at the GV-optimum epsilon is persisted on the
        # secondary ``{kernel}_gv`` entry below.
        if args.include_gv and summary["gv_at_eps_star"] is not None:
            kernel_entry["gv_error"] = summary["gv_at_eps_star"]
        kv[scheme_key][args.kernel] = kernel_entry
        updated_keys = [f"{scheme_key}.{args.kernel}"]
        if args.include_gv and summary["gv_epsilon"] is not None:
            gv_key = f"{args.kernel}_gv"
            gv_entry = dict(kv[scheme_key].get(gv_key, {}))
            gv_entry["epsilon"] = summary["gv_epsilon"]
            gv_entry["gv_error"] = summary["gv_error"]
            gv_entry["stable_at"] = summary["gv_stable_at"]
            kv[scheme_key][gv_key] = gv_entry
            updated_keys.append(f"{scheme_key}.{gv_key}")
        save_known_values(kv)
        print(f"\n  Updated known_values.json: {', '.join(updated_keys)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
