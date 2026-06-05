"""Boundary stencil derivation.

Provides:
- solve_boundary_row: solves the underdetermined Taylor system for a single
  boundary row, expressing coefficients as symbolic functions of free alpha
  parameters.
- derive_boundary: orchestrates the full boundary derivation for a given
  scheme, producing all boundary rows with symbolic alpha parameters.
"""

from dataclasses import dataclass

from sympy import Expr, Matrix, Rational, S, Symbol, cancel, symbols

from stencil_gen.taylor_system import build_taylor_system


@dataclass
class BoundaryRow:
    """Result of solving one boundary row's Taylor system."""

    row_index: int  # i (0..r-1)
    coefficients: list[Expr]  # length t, each is Expr in alpha symbols
    free_params: list[Symbol]  # the alpha symbols for this row


def solve_boundary_row(
    i: int,
    t: int,
    q: int,
    nu: int,
    free_symbols: list,
) -> BoundaryRow:
    """Solve the Taylor system for boundary row i.

    Parameters
    ----------
    i : int
        Row index (0..r-1).
    t : int
        Boundary stencil width.
    q : int
        Polynomial order of boundary scheme.
    nu : int
        Derivative order.
    free_symbols : list
        Pre-created alpha symbols (or S.Zero) for the free columns.
        Length must equal t - (q + 1).

    Returns
    -------
    BoundaryRow with coefficients as symbolic expressions of the free params.
    """
    V, rhs = build_taylor_system(i, t, q, nu)

    n_det = q + 1
    n_free = t - n_det
    assert len(free_symbols) == n_free

    # Partition: first n_det columns are determined, last n_free are free
    V_det = V[:, :n_det]
    V_free = V[:, n_det:]

    alpha_vec = Matrix(free_symbols)
    rhs_adjusted = rhs - V_free * alpha_vec

    gamma_det = V_det.solve(rhs_adjusted)

    coefficients = [cancel(gamma_det[k]) for k in range(n_det)] + list(free_symbols)

    return BoundaryRow(
        row_index=i,
        coefficients=coefficients,
        free_params=[s for s in free_symbols if isinstance(s, Symbol)],
    )


@dataclass
class BoundaryResult:
    """Result of full boundary derivation for a scheme."""

    r: int  # number of boundary rows
    t: int  # boundary stencil width
    rows: list[BoundaryRow]  # r BoundaryRow objects
    interior_coeffs: list[Rational]  # 2p+1 interior coefficients
    all_free_params: list[Symbol]  # all alpha symbols, globally ordered


def derive_boundary(
    p: int,
    nu: int = 1,
    s: int = 0,
) -> BoundaryResult:
    """Orchestrate full boundary derivation for a given scheme.

    Parameters
    ----------
    p : int
        RHS half-bandwidth.
    nu : int
        Derivative order (default 1).
    s : int
        LHS bandwidth (0 for explicit, default 0).

    Returns
    -------
    BoundaryResult with all boundary rows and interior coefficients.
    """
    from stencil_gen.interior import derive_interior, full_gamma_array

    r = 2 * p - 1
    t = r + p
    q = 2 * (p + s) - 1
    n_free_per_row = t - (q + 1)

    # Total number of free alpha parameters
    n_active_penultimate = min(n_free_per_row, 2)
    n_alpha = (r - 2) + n_active_penultimate

    # Create global alpha symbols
    all_alphas = list(symbols(f"alpha_0:{n_alpha}"))

    rows: list[BoundaryRow] = []
    alpha_idx = 0

    # Rows 0 through r-2
    for i in range(r - 1):
        if i < r - 2:
            # Single active alpha, rest zero-padded
            free = [all_alphas[alpha_idx]] + [S.Zero] * (n_free_per_row - 1)
            alpha_idx += 1
        else:
            # Penultimate row: n_active_penultimate active alphas
            active = [all_alphas[alpha_idx + k] for k in range(n_active_penultimate)]
            free = active + [S.Zero] * (n_free_per_row - n_active_penultimate)
            alpha_idx += n_active_penultimate
        rows.append(solve_boundary_row(i, t, q, nu, free))

    # Last row (r-1): placeholder symbols, all active
    phi_syms = list(symbols(f"phi_0:{n_free_per_row}"))
    rows.append(solve_boundary_row(r - 1, t, q, nu, phi_syms))

    # Interior coefficients from Phase 20.2
    interior = derive_interior(s, p, nu)
    interior_coeffs = full_gamma_array(interior)

    return BoundaryResult(
        r=r,
        t=t,
        rows=rows,
        interior_coeffs=interior_coeffs,
        all_free_params=all_alphas,
    )
