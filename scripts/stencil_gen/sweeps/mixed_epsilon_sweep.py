"""Mixed (per-row) epsilon sweep for RBF-augmented stencils.

Extracted from TestMixedEpsilon in test_phs.py.

Unlike the uniform epsilon sweep, this explores configurations where each
boundary row has its own shape parameter.  Strategies include:
  - Single uniform epsilon baseline
  - Two-group sweep (outer rows vs inner rows)
  - Coordinate descent over all R rows independently
  - Conservation variant (polynomial stencil for near-interior row)

Usage:
    uv run python -m sweeps.mixed_epsilon_sweep --scheme E4
    uv run python -m sweeps.mixed_epsilon_sweep --scheme E4 --n-eps 10  # quick
    uv run python -m sweeps.mixed_epsilon_sweep --scheme E4 --kernel multiquadric
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from stencil_gen.phs import (
    build_diff_matrix_mixed_epsilon,
    stability_eigenvalue_from_matrix,
    uniform_boundary_weights_rbf,
)

from ._common import SCHEME_PARAMS, STABILITY_TOL, load_known_values, save_known_values

# Floating-point eigenvalue solvers return tiny positive real parts (~1e-14)
# for genuinely stable operators.

def _stab_eig_mixed(
    n: int,
    epsilons: list[float],
    *,
    p: int,
    q: int,
    nu: int,
    nextra: int,
    kernel: str = "gaussian",
) -> float:
    """Compute stability eigenvalue for a mixed-epsilon configuration."""
    D = build_diff_matrix_mixed_epsilon(
        n, p=p, q=q, epsilons=list(epsilons),
        kernel=kernel, nu=nu, nextra=nextra,
    )
    return stability_eigenvalue_from_matrix(D)


def _status(se: float) -> str:
    return "STABLE" if se < STABILITY_TOL else "unstable"


def single_epsilon_baseline(
    n: int,
    r: int,
    n_eps: int,
    *,
    p: int,
    q: int,
    nu: int,
    nextra: int,
    kernel: str = "gaussian",
) -> tuple[float, float]:
    """Sweep uniform epsilon to establish baseline.

    Returns (best_eps, best_stab_eig).
    """
    epsilons_sweep = np.logspace(np.log10(0.1), np.log10(10), n_eps)
    best_eps, best_re = None, np.inf
    for eps in epsilons_sweep:
        mr = _stab_eig_mixed(
            n, [float(eps)] * r,
            p=p, q=q, nu=nu, nextra=nextra, kernel=kernel,
        )
        if mr < best_re:
            best_re = mr
            best_eps = float(eps)

    print(f"\n  Single-epsilon baseline (n={n}, kernel={kernel}):")
    print(f"  Best eps={best_eps:.4f}, stab_eig={best_re:.6e} [{_status(best_re)}]")
    return best_eps, best_re


def two_group_sweep(
    n: int,
    r: int,
    n_eps: int,
    *,
    p: int,
    q: int,
    nu: int,
    nextra: int,
    kernel: str = "gaussian",
) -> tuple[tuple[float, float], float]:
    """2D sweep: eps_outer (first half of rows) vs eps_inner (second half).

    Returns ((best_outer, best_inner), best_stab_eig).
    """
    eps_range = np.logspace(np.log10(0.3), np.log10(8.0), n_eps)
    r_outer = r // 2
    r_inner = r - r_outer

    best_combo = None
    best_re = np.inf

    for eps_outer in eps_range:
        for eps_inner in eps_range:
            epsilons = [float(eps_outer)] * r_outer + [float(eps_inner)] * r_inner
            mr = _stab_eig_mixed(
                n, epsilons,
                p=p, q=q, nu=nu, nextra=nextra, kernel=kernel,
            )
            if mr < best_re:
                best_re = mr
                best_combo = (float(eps_outer), float(eps_inner))

    print(f"\n  Two-group sweep (n={n}, kernel={kernel}):")
    print(f"  Rows 0..{r_outer-1} (outer): eps={best_combo[0]:.4f}")
    print(f"  Rows {r_outer}..{r-1} (inner): eps={best_combo[1]:.4f}")
    print(f"  stab_eig={best_re:.6e} [{_status(best_re)}]")

    # Verify at multiple grid sizes
    print(f"\n  Checking best combo across grid sizes:")
    for nn in [20, 40, 80]:
        epsilons = [best_combo[0]] * r_outer + [best_combo[1]] * r_inner
        mr = _stab_eig_mixed(
            nn, epsilons,
            p=p, q=q, nu=nu, nextra=nextra, kernel=kernel,
        )
        print(f"    n={nn:3d}: stab_eig={mr:.6e} [{_status(mr)}]")

    return best_combo, best_re


def per_row_coordinate_descent(
    n: int,
    r: int,
    n_eps: int,
    *,
    p: int,
    q: int,
    nu: int,
    nextra: int,
    kernel: str = "gaussian",
    start_eps: float | None = None,
    n_passes: int = 3,
) -> tuple[list[float], float]:
    """Coordinate descent over R independent per-row epsilon values.

    Returns (optimal_epsilons, best_stab_eig).
    """
    eps_vals = np.logspace(np.log10(0.3), np.log10(10.0), n_eps)

    if start_eps is None:
        start_eps = 1.7 if kernel == "gaussian" else 5.0
    current = [start_eps] * r
    current_re = _stab_eig_mixed(
        n, current,
        p=p, q=q, nu=nu, nextra=nextra, kernel=kernel,
    )

    for _iteration in range(n_passes):
        for row in range(r):
            best_eps_row = current[row]
            best_re_row = current_re
            for eps in eps_vals:
                trial = list(current)
                trial[row] = float(eps)
                mr = _stab_eig_mixed(
                    n, trial,
                    p=p, q=q, nu=nu, nextra=nextra, kernel=kernel,
                )
                if mr < best_re_row:
                    best_re_row = mr
                    best_eps_row = float(eps)
            current[row] = best_eps_row
            current_re = best_re_row

    print(f"\n  Per-row coordinate descent (n={n}, kernel={kernel}):")
    print(f"  Optimal epsilons: [{', '.join(f'{e:.4f}' for e in current)}]")
    print(f"  stab_eig={current_re:.6e} [{_status(current_re)}]")

    # Compare with uniform best
    uniform_eps = np.logspace(np.log10(0.5), np.log10(5.0), n_eps)
    uniform_mr = min(
        _stab_eig_mixed(
            n, [float(eps)] * r,
            p=p, q=q, nu=nu, nextra=nextra, kernel=kernel,
        )
        for eps in uniform_eps
    )
    print(f"\n  Uniform best: stab_eig={uniform_mr:.6e}")
    print(f"  Mixed best:   stab_eig={current_re:.6e}")

    # Grid convergence
    check_sizes = [20, 40, 80, 160] if kernel == "gaussian" else [20, 40, 80]
    print(f"\n  Grid convergence with optimal epsilons:")
    for nn in check_sizes:
        mr = _stab_eig_mixed(
            nn, current,
            p=p, q=q, nu=nu, nextra=nextra, kernel=kernel,
        )
        print(f"    n={nn:3d}: stab_eig={mr:.6e} [{_status(mr)}]")

    return current, current_re


def conservation_near_interior(
    n: int,
    r: int,
    n_eps: int,
    *,
    p: int,
    q: int,
    nu: int,
    nextra: int,
) -> tuple[float, float]:
    """Hybrid: RBF for rows 0..r-2, polynomial stencil for row r-1.

    The near-interior row (r-1) uses a very large epsilon (polynomial limit),
    while the outer rows sweep over epsilon.

    Returns (best_eps, best_stab_eig).
    """
    from stencil_gen.interior import derive_interior, full_gamma_array

    t = p + q + 1 + nextra

    # Interior stencil for circulant rows
    interior_coeffs = derive_interior(0, p, nu)
    interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

    # Polynomial stencil for row r-1 (eps=100 approximates polynomial limit)
    poly_w_last = uniform_boundary_weights_rbf(
        r - 1, t, nu, q, epsilon=100.0, kernel="gaussian"
    )

    eps_range = np.logspace(np.log10(0.3), np.log10(8.0), n_eps)
    best_eps, best_re = None, np.inf

    for eps in eps_range:
        D = np.zeros((n, n))

        # Left boundary rows 0..r-2: RBF
        for i in range(r - 1):
            w = uniform_boundary_weights_rbf(i, t, nu, q, float(eps))
            for j in range(t):
                D[i, j] = w[j]

        # Left boundary row r-1: polynomial stencil
        for j in range(t):
            D[r - 1, j] = poly_w_last[j]

        # Interior rows
        for i in range(r, n - r):
            for k_idx, jj in enumerate(range(i - p, i + p + 1)):
                D[i, jj] = interior_w[k_idx]

        # Right boundary: reflected
        sign = (-1.0) ** nu
        for i in range(r - 1):
            w = uniform_boundary_weights_rbf(i, t, nu, q, float(eps))
            row = n - 1 - i
            for j in range(t):
                D[row, n - 1 - j] = sign * w[j]
        # Right row r-1: reflected polynomial
        row = n - 1 - (r - 1)
        for j in range(t):
            D[row, n - 1 - j] = sign * poly_w_last[j]

        mr = stability_eigenvalue_from_matrix(D)
        if mr < best_re:
            best_re = mr
            best_eps = float(eps)

    print(f"\n  Conservation near-interior (n={n}):")
    print(f"  Row {r-1} uses polynomial stencil (eps->inf limit)")
    print(f"  Rows 0..{r-2} use Gaussian with swept eps")
    print(f"  Best eps={best_eps:.4f}, stab_eig={best_re:.6e} [{_status(best_re)}]")

    # Also try: different eps for row r-1 (2D sweep)
    n_coarse = max(n_eps // 2, 10)
    eps_coarse = np.logspace(np.log10(0.3), np.log10(8.0), n_coarse)
    best2_main, best2_last, best2_re = None, None, np.inf

    for eps_main in eps_coarse:
        for eps_last in eps_coarse:
            epsilons = [float(eps_main)] * (r - 1) + [float(eps_last)]
            mr = _stab_eig_mixed(
                n, epsilons,
                p=p, q=q, nu=nu, nextra=nextra,
            )
            if mr < best2_re:
                best2_re = mr
                best2_main = float(eps_main)
                best2_last = float(eps_last)

    print(f"\n  Variant: different eps for row {r-1} (2D sweep):")
    print(f"  Best: eps_main={best2_main:.4f}, eps_last={best2_last:.4f}")
    print(f"  stab_eig={best2_re:.6e} [{_status(best2_re)}]")

    return best_eps, best_re


def run_mixed_epsilon_sweep(
    scheme: str,
    kernel: str,
    n_eps: int,
) -> dict:
    """Run the full mixed epsilon sweep suite for a scheme/kernel.

    Returns summary dict with per-row optimal epsilons and stability info.
    """
    params = SCHEME_PARAMS[scheme]
    p, q, nextra, nu = params["p"], params["q"], params["nextra"], params["nu"]
    label = params["label"]
    r = q + 1 + nextra

    n = 40
    common = dict(p=p, q=q, nu=nu, nextra=nextra)

    print(f"\n{'='*72}")
    print(f"  {label} Mixed Epsilon Sweep (p={p}, q={q}, nextra={nextra}, r={r})")
    print(f"{'='*72}")

    # 1. Single epsilon baseline
    baseline_eps, baseline_re = single_epsilon_baseline(
        n, r, max(n_eps, 40), kernel=kernel, **common,
    )

    # 2. Two-group sweep
    two_group_combo, two_group_re = two_group_sweep(
        n, r, n_eps, kernel=kernel, **common,
    )

    # 3. Per-row coordinate descent
    opt_eps, opt_re = per_row_coordinate_descent(
        n, r, n_eps, kernel=kernel, **common,
    )

    # 4. Conservation near-interior (Gaussian only)
    if kernel == "gaussian":
        conservation_near_interior(n, r, n_eps, **common)

    return {
        "baseline_epsilon": round(baseline_eps, 6),
        "baseline_stab_eig": baseline_re,
        "per_row_epsilons": [round(e, 6) for e in opt_eps],
        "per_row_stab_eig": opt_re,
        "two_group": {
            "outer": round(two_group_combo[0], 6),
            "inner": round(two_group_combo[1], 6),
            "stab_eig": two_group_re,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.mixed_epsilon_sweep",
        description="Mixed (per-row) epsilon sweep for RBF-augmented stencils",
    )
    parser.add_argument("--scheme", choices=["E2", "E4"], default="E4")
    parser.add_argument(
        "--kernel", choices=["gaussian", "multiquadric"], default="gaussian",
    )
    parser.add_argument(
        "--n-eps", type=int, default=20,
        help="Number of epsilon sample points per dimension (default: 20)",
    )
    parser.add_argument(
        "--update-known-values", action="store_true",
        help="Update known_values.json with discovered optimal epsilons",
    )

    args = parser.parse_args(argv)

    summary = run_mixed_epsilon_sweep(args.scheme, args.kernel, args.n_eps)

    if args.update_known_values:
        kv = load_known_values()
        scheme_key = SCHEME_PARAMS[args.scheme]["label"]
        if scheme_key not in kv:
            kv[scheme_key] = {}
        mixed_key = f"mixed_{args.kernel}"
        kv[scheme_key][mixed_key] = {
            "per_row_epsilons": summary["per_row_epsilons"],
            "baseline_epsilon": summary["baseline_epsilon"],
        }
        save_known_values(kv)
        print(f"\n  Updated known_values.json: {scheme_key}.{mixed_key}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
