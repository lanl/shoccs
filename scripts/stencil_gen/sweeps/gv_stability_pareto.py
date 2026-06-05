"""Group-velocity vs stability Pareto sweep (research / documentation aid).

For a chosen scheme and RBF kernel, this sweep evaluates
``(stab_eig, gv_error)`` on a fine 1D parameter grid and prints two tables:

1. A full grid table sorted by parameter value.
2. A Pareto-optimal subset — stable points that are not dominated by any
   other stable point under the partial order ``(stab_eig, gv_error)`` (lower
   is better on both axes).

Eigenvalue stability remains the hard feasibility gate; GV error is the
secondary objective.  No ``--update-known-values`` path: this sweep is for
research and docs only, mirroring the read-only ``comparison`` sweep.

Usage:
    uv run python -m sweeps gv-stability-pareto --scheme E2 --param tension
    uv run python -m sweeps gv-stability-pareto --scheme E4 --param gaussian --n-points 21
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from stencil_gen.phs import stability_eigenvalue

from ._common import SCHEME_PARAMS, STABILITY_TOL
from .gv_objectives import boundary_gv_error_max

_PARAM_LABEL = {
    "tension": "sigma",
    "gaussian": "epsilon",
    "multiquadric": "epsilon",
}


def _default_grid(param: str, param_max: float, n_points: int) -> np.ndarray:
    """Build a 1D parameter grid appropriate for the chosen kernel.

    Tension includes the ``sigma=0`` PHS k=2 limit; Gaussian/multiquadric use
    a pure log-spaced positive range (epsilon=0 is not meaningful).
    """
    if param == "tension":
        if n_points <= 1:
            return np.array([0.0])
        return np.concatenate(
            [[0.0], np.logspace(np.log10(0.01), np.log10(param_max), n_points - 1)]
        )
    return np.logspace(np.log10(0.01), np.log10(param_max), n_points)


def evaluate_grid(
    scheme: str,
    param: str,
    values: np.ndarray,
    *,
    n: int,
) -> list[tuple[float, float, float]]:
    """Return ``[(val, stab_eig, gv_err), ...]`` sorted by ``val``."""
    sp = SCHEME_PARAMS[scheme]
    p, q, nextra, nu = sp["p"], sp["q"], sp["nextra"], sp["nu"]
    out: list[tuple[float, float, float]] = []
    for val in values:
        v = float(val)
        se = stability_eigenvalue(
            n, p=p, q=q, epsilon=v,
            kernel=param, nu=nu, nextra=nextra,
        )
        gv = boundary_gv_error_max(
            p=p, q=q, nextra=nextra, nu=nu, sigma=v, kernel=param,
        )
        out.append((v, float(se), float(gv)))
    out.sort(key=lambda r: r[0])
    return out


def pareto_front(
    rows: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Non-dominated subset of stable rows under ``(stab_eig, gv_err)``.

    A stable row ``a`` dominates ``b`` iff ``a.stab_eig <= b.stab_eig`` and
    ``a.gv_err <= b.gv_err`` with at least one inequality strict.  Only
    feasible (``stab_eig < STABILITY_TOL``) rows are considered.
    """
    stable = [r for r in rows if r[1] < STABILITY_TOL]
    front: list[tuple[float, float, float]] = []
    for i, a in enumerate(stable):
        dominated = False
        for j, b in enumerate(stable):
            if i == j:
                continue
            if (
                b[1] <= a[1]
                and b[2] <= a[2]
                and (b[1] < a[1] or b[2] < a[2])
            ):
                dominated = True
                break
        if not dominated:
            front.append(a)
    front.sort(key=lambda r: r[0])
    return front


def print_markdown_table(
    rows: list[tuple[float, float, float]],
    *,
    param_label: str,
    title: str,
) -> None:
    """Print a sorted markdown table of ``(param, stab_eig, gv_err, status)``."""
    print(f"\n### {title}\n")
    print(f"| {param_label} | stab_eig | gv_err | status |")
    print("|---|---|---|---|")
    for val, se, gv in rows:
        status = "STABLE" if se < STABILITY_TOL else "unstable"
        print(f"| {val:.6f} | {se:.6e} | {gv:.6e} | {status} |")


def run_gv_stability_pareto(
    scheme: str,
    param: str,
    n_points: int,
    *,
    n: int,
    param_max: float,
) -> dict:
    """Run the Pareto sweep and print both tables.  Returns a summary dict."""
    values = _default_grid(param, param_max, n_points)
    rows = evaluate_grid(scheme, param, values, n=n)
    label = SCHEME_PARAMS[scheme]["label"]
    param_label = _PARAM_LABEL[param]

    print(
        f"\nGV-stability Pareto sweep: {label} / {param} "
        f"(n={n}, {n_points} points)"
    )
    print_markdown_table(
        rows,
        param_label=param_label,
        title=f"{label} — {param} sweep (sorted by {param_label})",
    )

    front = pareto_front(rows)
    if front:
        print_markdown_table(
            front,
            param_label=param_label,
            title=f"{label} — Pareto-optimal points (stable, non-dominated)",
        )
    else:
        print("\n### Pareto-optimal points (stable, non-dominated)\n")
        print("*(no stable points on this grid)*")

    return {"rows": rows, "pareto": front}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.gv_stability_pareto",
        description="GV-vs-stability Pareto sweep (research / documentation aid)",
    )
    parser.add_argument("--scheme", choices=["E2", "E4"], required=True)
    parser.add_argument(
        "--param", choices=["tension", "gaussian", "multiquadric"], required=True,
        help="Kernel to sweep (parameter is sigma for tension, epsilon otherwise)",
    )
    parser.add_argument(
        "--n-points", type=int, default=61,
        help="Number of sample points on the parameter grid (default: 61)",
    )
    parser.add_argument(
        "--n", type=int, default=40,
        help="Grid size for the stability eigenvalue (default: 40)",
    )
    parser.add_argument(
        "--param-max", type=float, default=20.0,
        help="Upper end of the parameter grid (default: 20.0)",
    )

    args = parser.parse_args(argv)
    run_gv_stability_pareto(
        args.scheme, args.param, args.n_points,
        n=args.n, param_max=args.param_max,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
