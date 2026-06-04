"""Boundary alpha extraction at optimal Gaussian epsilon.

Extracted from TestStableEpsilonAlphas in test_phs.py.

For a given stencil scheme (E2/E4), finds the optimal Gaussian epsilon,
extracts the implied boundary stencil alpha parameters, and compares them
with production values from the C++ solver.

Steps:
1. Fine-sweep epsilon to find the most stable value (most negative max Re(lambda))
2. Extract RBF boundary weights at that epsilon
3. Map weights to alpha values in the TEMO parameterization
4. Verify eigenvalue stability with the extracted alphas
5. Compare with optimizer-derived production alphas
6. Report conservation deficit

Usage:
    uv run python -m sweeps.alpha_extraction --scheme E2
    uv run python -m sweeps.alpha_extraction --scheme E4
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from stencil_gen.interior import derive_interior, full_gamma_array
from stencil_gen.phs import (
    build_diff_matrix_rbf,
    stability_eigenvalue,
    stability_eigenvalue_from_matrix,
    uniform_boundary_weights_rbf,
)

from ._common import SCHEME_PARAMS, print_table

# Production alphas from src/operators/gradient.t.cpp
PRODUCTION_ALPHAS = {
    "E2": [
        -1.47956280234494,
        0.261900367793859,
        -0.145072532538541,
        -0.224665713988644,
    ],
}


def find_best_epsilon(
    p: int, q: int, nextra: int, nu: int,
    n: int = 40, eps_min: float = 1.5, eps_max: float = 3.5, n_eps: int = 200,
) -> tuple[float, float]:
    """Find the Gaussian epsilon that minimises max Re(lambda)."""
    epsilons = np.linspace(eps_min, eps_max, n_eps)
    best_eps, best_se = None, np.inf
    for eps in epsilons:
        se = stability_eigenvalue(
            n, p=p, q=q, epsilon=eps,
            kernel="gaussian", nu=nu, nextra=nextra,
        )
        if se < best_se:
            best_se = se
            best_eps = eps
    return best_eps, best_se


def extract_boundary_weights(
    p: int, q: int, nextra: int, nu: int, epsilon: float,
) -> tuple[np.ndarray, int, int]:
    """Extract the r x t boundary weight matrix at a given epsilon."""
    t = p + q + 1 + nextra
    r = q + 1 + nextra
    rows = []
    for i in range(r):
        w = uniform_boundary_weights_rbf(
            i, t, nu, q, epsilon, kernel="gaussian"
        )
        rows.append(w)
    return np.array(rows), r, t


def extract_alphas(
    scheme: str, epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Extract alpha values from RBF boundary weights at given epsilon.

    Uses the symbolic TEMO boundary matrix to set up a linear system:
    for each entry (i, j) in rows 0..r-2, the equation
    B_sym[i,j](alphas) = B_num[i,j] is linear in the alphas.  We collect
    all such equations and solve the overdetermined system via least squares.

    Returns (rbf_alphas, B_num, B_temo, ur).
    """
    import sympy

    from stencil_gen.temo import E2_1, E4_1, derive_uniform_boundary_for_temo

    sp = SCHEME_PARAMS[scheme]
    p, q, nextra, nu = sp["p"], sp["q"], sp["nextra"], sp["nu"]

    B_num, r, t = extract_boundary_weights(p, q, nextra, nu, epsilon)

    temo_scheme = {"E2": E2_1, "E4": E4_1}[scheme]
    ur = derive_uniform_boundary_for_temo(temo_scheme)
    B_sym = ur.B_u
    alphas = ur.alpha_symbols
    n_alpha = len(alphas)

    # Build linear system from non-conserved rows (0..r-2).
    # B_sym[i,j] = c_ij + sum_k a_ijk * alpha_k
    # => sum_k a_ijk * alpha_k = B_num[i,j] - c_ij
    r_eff = r - 1
    equations = []
    for i in range(r_eff):
        for j in range(t):
            equations.append(B_sym[i, j] - B_num[i, j])

    A_sym, b_sym = sympy.linear_eq_to_matrix(equations, list(alphas))
    A_np = np.array(A_sym, dtype=float)
    b_np = np.array(b_sym, dtype=float).ravel()

    # Solve (may be overdetermined)
    rbf_alphas, residuals, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)

    # Build the TEMO boundary block with these alphas
    B_temo = np.zeros((r, t))
    for i_row in range(r):
        for j_col in range(t):
            expr = B_sym[i_row, j_col]
            B_temo[i_row, j_col] = float(
                expr.subs({a: v for a, v in zip(alphas, rbf_alphas)})
            )

    return rbf_alphas, B_num, B_temo, ur


def build_D_from_boundary(
    n: int, B_boundary: np.ndarray, p: int, nu: int,
) -> np.ndarray:
    """Build n x n differentiation matrix from explicit boundary weights."""
    r, t = B_boundary.shape
    interior_coeffs = derive_interior(0, p, nu)
    interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

    D = np.zeros((n, n))
    # Left boundary
    for i in range(r):
        for j in range(t):
            D[i, j] = B_boundary[i, j]
    # Interior
    for i in range(r, n - r):
        for k_idx, j in enumerate(range(i - p, i + p + 1)):
            D[i, j] = interior_w[k_idx]
    # Right boundary (antisymmetric for nu=1)
    for i in range(r):
        row = n - 1 - i
        for j in range(t):
            D[row, n - 1 - j] = -B_boundary[i, j]
    return D


def run_alpha_extraction(scheme: str) -> int:
    """Run the full alpha extraction pipeline for a scheme."""
    sp = SCHEME_PARAMS[scheme]
    p, q, nextra, nu = sp["p"], sp["q"], sp["nextra"], sp["nu"]
    label = sp["label"]

    # 1. Find best epsilon
    print(f"\n{'='*70}")
    print(f"  Alpha extraction for {label}")
    print(f"{'='*70}")

    eps_star, best_se = find_best_epsilon(p, q, nextra, nu)
    print(f"\n  Best Gaussian epsilon: eps* = {eps_star:.6f}")
    print(f"  max Re(lambda) at eps*: {best_se:.6e}")

    # 2. Extract alphas
    rbf_alphas, B_num, B_temo, ur = extract_alphas(scheme, eps_star)
    alphas = ur.alpha_symbols
    r, t = B_num.shape

    # 3. Report boundary weights
    print(f"\n  RBF boundary weights at eps*={eps_star:.6f}:")
    for i in range(r):
        w_str = ", ".join(f"{B_num[i, j]:12.8f}" for j in range(t))
        print(f"    row {i}: [{w_str}]")

    # 4. Report extracted alphas
    print(f"\n  Extracted alphas (from free columns of rows 0..{r-2}):")
    for k, (a, v) in enumerate(zip(alphas, rbf_alphas)):
        print(f"    {a} = {v:.12f}")

    # 5. Verify rows 0..r-2 match exactly
    for row in range(r - 1):
        resid = np.max(np.abs(B_temo[row] - B_num[row]))
        print(f"    Row {row} residual: {resid:.6e}")
        if resid > 1e-12:
            print(f"    WARNING: Row {row} residual too large!")

    # 6. Row r-1: TEMO (conservation-enforced) vs RBF (not conserved)
    row_last = r - 1
    row_diff = B_temo[row_last] - B_num[row_last]
    print(f"\n  Row {row_last} comparison (TEMO conservation vs RBF):")
    print(f"    TEMO: [{', '.join(f'{v:12.8f}' for v in B_temo[row_last])}]")
    print(f"    RBF:  [{', '.join(f'{v:12.8f}' for v in B_num[row_last])}]")
    print(f"    Diff: [{', '.join(f'{v:12.8f}' for v in row_diff)}]")
    print(f"    Max diff: {np.max(np.abs(row_diff)):.6e}")

    # 7. Stability comparison across grid sizes
    headers = ["n", "RBF direct", "TEMO+RBF alpha"]
    rows = []
    for n in [20, 40, 80, 160]:
        D_rbf = build_diff_matrix_rbf(
            n, p=p, q=q, epsilon=eps_star,
            kernel="gaussian", nu=nu, nextra=nextra,
        )
        se_rbf = stability_eigenvalue_from_matrix(D_rbf)

        D_temo = build_D_from_boundary(n, B_temo, p, nu)
        se_temo = stability_eigenvalue_from_matrix(D_temo)

        rows.append([str(n), f"{se_rbf:.6e}", f"{se_temo:.6e}"])
    print_table(f"Stability eigenvalue comparison (eps*={eps_star:.4f})", headers, rows)

    # 8. Compare with production alphas if available
    if scheme in PRODUCTION_ALPHAS:
        prod_alphas = np.array(PRODUCTION_ALPHAS[scheme])
        B_sym = ur.B_u

        print(f"\n  Alpha comparison ({label}, eps*={eps_star:.4f}):")
        alpha_headers = ["Symbol", "RBF-extracted", "Production", "Diff"]
        alpha_rows = []
        for k, a in enumerate(alphas):
            diff = rbf_alphas[k] - prod_alphas[k]
            alpha_rows.append([
                str(a),
                f"{rbf_alphas[k]:.10f}",
                f"{prod_alphas[k]:.10f}",
                f"{diff:.6e}",
            ])
        print_table(f"Alpha values ({label})", alpha_headers, alpha_rows)

        # Build boundary block with production alphas
        B_prod = np.zeros((r, t))
        for i_row in range(r):
            for j_col in range(t):
                expr = B_sym[i_row, j_col]
                B_prod[i_row, j_col] = float(
                    expr.subs({a: v for a, v in zip(alphas, prod_alphas)})
                )

        # Stability comparison: RBF vs TEMO+RBF vs Production
        n = 40
        D_rbf = build_diff_matrix_rbf(
            n, p=p, q=q, epsilon=eps_star,
            kernel="gaussian", nu=nu, nextra=nextra,
        )
        se_rbf = stability_eigenvalue_from_matrix(D_rbf)
        sr_rbf = float(np.max(np.abs(np.linalg.eigvals(D_rbf))))

        D_temo_rbf = build_D_from_boundary(n, B_temo, p, nu)
        se_temo_rbf = stability_eigenvalue_from_matrix(D_temo_rbf)
        sr_temo_rbf = float(np.max(np.abs(np.linalg.eigvals(D_temo_rbf))))

        D_prod = build_D_from_boundary(n, B_prod, p, nu)
        se_prod = stability_eigenvalue_from_matrix(D_prod)
        sr_prod = float(np.max(np.abs(np.linalg.eigvals(D_prod))))

        method_headers = ["Method", "stab eig", "spec radius"]
        method_rows = [
            ["RBF direct (eps*)", f"{se_rbf:.6e}", f"{sr_rbf:.6e}"],
            ["TEMO + RBF alphas", f"{se_temo_rbf:.6e}", f"{sr_temo_rbf:.6e}"],
            ["Production alphas", f"{se_prod:.6e}", f"{sr_prod:.6e}"],
        ]
        print_table(f"Stability at n={n}", method_headers, method_rows)

    # 9. Conservation deficit
    _report_conservation_deficit(B_num, p, q, nextra, nu, eps_star)

    return 0


def _report_conservation_deficit(
    B_num: np.ndarray, p: int, q: int, nextra: int, nu: int, eps_star: float,
) -> None:
    """Report conservation deficit of the RBF boundary stencil."""
    r, t = B_num.shape

    print(f"\n  Conservation analysis (eps*={eps_star:.4f}):")
    print(f"  r={r}, t={t}, p={p}")

    col_headers = ["Column", "Boundary sum"]
    col_rows = []
    for j in range(t):
        col_sum = sum(B_num[i, j] for i in range(r))
        col_rows.append([str(j), f"{col_sum:.8f}"])
    print_table("Column sums of boundary block B_u", col_headers, col_rows)

    print(f"  Conservation deficit (boundary col sum for overlap cols j >= {p+1}):")
    for j in range(p + 1, t):
        col_sum = sum(B_num[i, j] for i in range(r))
        print(f"    col {j}: boundary sum = {col_sum:.8f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract boundary alpha values at optimal Gaussian epsilon",
    )
    parser.add_argument("--scheme", choices=["E2", "E4"], required=True)
    args = parser.parse_args(argv)

    return run_alpha_extraction(args.scheme)


if __name__ == "__main__":
    sys.exit(main())
