"""Conservation (SBP) constraint solver for boundary stencils.

Provides:
- build_conservation_system: constructs the conservation constraint equations
  from boundary rows and interior coefficients.
- solve_conservation: solves the constraint system for quadrature weights and
  last-row placeholder symbols.
"""

from sympy import Expr, Rational, S, Symbol, cancel, expand, linear_eq_to_matrix, symbols

from stencil_gen._util import solve_linear

from stencil_gen.boundary import BoundaryRow


def _interior_contribution(j: int, r: int, p: int, interior_coeffs: list[Rational]) -> Expr:
    """Compute the sum of interior stencil contributions to column j.

    Interior row r+m (m >= 0) is centered at grid point r+m and covers
    columns (r+m-p)..(r+m+p). Its coefficient at column j is
    interior_coeffs[j - (r+m) + p].
    """
    ic = S.Zero
    # Valid m: max(0, j - r - p) <= m <= j - r + p
    m_lo = max(0, j - r - p)
    m_hi = j - r + p
    for m in range(m_lo, m_hi + 1):
        idx = j - (r + m) + p
        if 0 <= idx <= 2 * p:
            ic += interior_coeffs[idx]
    return ic


def build_conservation_system(
    r: int,
    t: int,
    p: int,
    boundary_rows: list[BoundaryRow],
    interior_coeffs: list[Rational],
) -> tuple[list[Expr], list[Symbol], list[Symbol]]:
    """Build the conservation (SBP) constraint equations.

    Parameters
    ----------
    r : int
        Number of boundary rows.
    t : int
        Boundary stencil width.
    p : int
        RHS half-bandwidth.
    boundary_rows : list[BoundaryRow]
        The r boundary rows from derive_boundary.
    interior_coeffs : list[Rational]
        The 2p+1 interior coefficients from derive_interior.

    Returns
    -------
    (equations, w_symbols, last_row_free_symbols)
        equations: list of Expr that must equal zero
        w_symbols: [w_0, ..., w_{r-1}] quadrature weight symbols
        last_row_free_symbols: placeholder symbols from the last boundary row
    """
    # Quadrature weight symbols
    w_syms = list(symbols(f"w_0:{r}"))

    # Placeholder symbols from the last row
    last_row_free = list(boundary_rows[r - 1].free_params)

    equations = []
    for j in range(t - 1):
        # Weighted column sum from boundary rows
        col_sum = S.Zero
        for i_row in range(r):
            col_sum += w_syms[i_row] * boundary_rows[i_row].coefficients[j]

        # Interior contribution
        ic = _interior_contribution(j, r, p, interior_coeffs)
        col_sum += ic

        # SBP condition: column 0 sums to -1, all others sum to 0
        if j == 0:
            col_sum += 1  # move -1 to left side => +1

        equations.append(col_sum)

    return equations, w_syms, last_row_free


def solve_conservation(
    equations: list[Expr],
    w_symbols: list[Symbol],
    last_row_free: list[Symbol],
    all_free_params: list[Symbol],
    boundary_rows: list[BoundaryRow],
) -> tuple[dict[Symbol, Expr], list[BoundaryRow]]:
    """Solve the conservation constraint system.

    The conservation equations contain bilinear terms w_{r-1} * phi_k
    (products of two unknowns). We linearize by substituting
    theta_k = w_{r-1} * phi_k, solve the resulting linear system,
    then recover phi_k = theta_k / w_{r-1}.

    Parameters
    ----------
    equations : list[Expr]
        Expressions that must equal zero (from build_conservation_system).
    w_symbols : list[Symbol]
        Quadrature weight symbols [w_0, ..., w_{r-1}].
    last_row_free : list[Symbol]
        Placeholder symbols from the last boundary row (phi_0, ...).
    all_free_params : list[Symbol]
        Global alpha symbols (treated as parameters, not unknowns).
    boundary_rows : list[BoundaryRow]
        The r boundary rows from derive_boundary.

    Returns
    -------
    (solution_dict, updated_rows)
        solution_dict: maps each w_i and phi_j to its expression in alphas
        updated_rows: list[BoundaryRow] with last row's placeholders resolved
    """
    r = len(boundary_rows)
    n_phi = len(last_row_free)

    if n_phi == 0:
        # No placeholder symbols, straightforward linear solve
        A, b = linear_eq_to_matrix(equations, w_symbols)
        solution_dict = solve_linear(A, b, w_symbols)
        return solution_dict, list(boundary_rows)

    # Linearize: substitute w_{r-1} * phi_k -> theta_k
    w_last = w_symbols[r - 1]
    theta_syms = list(symbols(f"theta_0:{n_phi}"))
    sub_dict = {w_last * phi: theta for phi, theta in zip(last_row_free, theta_syms)}

    lin_eqs = [expand(eq).subs(sub_dict) for eq in equations]

    # Solve linearized system: unknowns are (w_0, ..., w_{r-1}, theta_0, ...)
    lin_unknowns = list(w_symbols) + theta_syms
    A, b = linear_eq_to_matrix(lin_eqs, lin_unknowns)

    lin_solution = solve_linear(A, b, lin_unknowns)

    # Recover phi_k = theta_k / w_{r-1}
    w_last_val = lin_solution[w_last]
    solution_dict = {}
    for w in w_symbols:
        solution_dict[w] = lin_solution[w]
    for phi, theta in zip(last_row_free, theta_syms):
        solution_dict[phi] = cancel(lin_solution[theta] / w_last_val)

    # Build updated rows: copy rows 0..r-2 unchanged, substitute last row
    updated_rows = list(boundary_rows[:r - 1])

    last_row = boundary_rows[r - 1]
    updated_coeffs = [cancel(c.xreplace(solution_dict)) for c in last_row.coefficients]
    updated_last = BoundaryRow(
        row_index=last_row.row_index,
        coefficients=updated_coeffs,
        free_params=[],
    )
    updated_rows.append(updated_last)

    return solution_dict, updated_rows
