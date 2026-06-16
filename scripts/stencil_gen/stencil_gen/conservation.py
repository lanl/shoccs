"""Discrete-conservation (telescoping/flux) constraint solver for boundary stencils.

Provides:
- build_conservation_system: constructs the conservation constraint equations
  from boundary rows and interior coefficients.
- solve_conservation: solves the constraint system for quadrature weights and
  last-row placeholder symbols.
- solve_cut_cell_conservation_dof: the scheme-agnostic cut-cell port of
  Mathematica ``conservationCutCell`` (taylor.wl:763-771) that fixes a free
  closure DOF from the wall-column flux condition and the interior column sums.
"""

from sympy import Expr, Rational, S, Symbol, cancel, expand, linear_eq_to_matrix, solve, symbols

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
    """Build the discrete-conservation (telescoping/flux) constraint equations.

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

        # Discrete-conservation (telescoping/flux) condition: column 0 sums to -1, all others sum to 0
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


def solve_cut_cell_conservation_dof(
    closure_rows: list[list[Expr]],
    interior_coeffs: list[Rational],
    p: int,
    free_dof: Symbol,
    psi: Symbol,
    *,
    nu: int = 1,
) -> Expr:
    """Fix a free closure DOF from the cut-cell discrete-conservation conditions.

    Scheme-agnostic port of Mathematica ``conservationCutCell`` (taylor.wl:763-771)
    for the cut-cell branch: the realized closure rows plus the interior band must
    satisfy the telescoping/flux property under the conservation (quadrature)
    weights — the WALL column weighted-sum equals the boundary flux (``-1`` for a
    1st-derivative operator) and every fully-covered interior grid-point column
    sums to zero.

    The closure rows live in the cut-cell T-frame:
      col 0 = wall point, col j (j >= 1) = grid point j-1.
    Exactly one entry across all rows is parameterized by ``free_dof``; the other
    entries are realized in the remaining free closure parameters.  The interior
    band is laid as extra rows (the ``toLeftMatrix``/``toBand`` step) so that the
    grid-point columns within the closure width are fully covered.

    Recipe (mirrors taylor.wl):
      1. Build the realized left-matrix = ``closure_rows`` stacked on interior band
         rows centered so the first fully-covered grid columns are reached.
      2. Assign one symbolic conservation weight per closure row and weight 1 to
         the interior rows (``w_0`` of the cut-cell row is carried as the realized
         coefficient itself — the rows already encode it).
      3. ``conservationUniform``: solve the fully-covered interior grid columns for
         the closure weights.  This step is bilinear in (weight, ``free_dof``) but
         solves to a single clean branch for the E2 poly case.
      4. ``conservationCutCell``: substitute the weights into the wall-column sum
         and impose it equals ``-nu``-flux (``-1`` for ``nu == 1``).  Solve that
         single equation for ``free_dof``.

    Parameters
    ----------
    closure_rows : list[list[Expr]]
        The realized cut-cell closure rows, each of length ``T`` (col 0 = wall).
        Exactly one entry is parameterized by ``free_dof``.
    interior_coeffs : list[Rational]
        The ``2*p + 1`` interior stencil coefficients.
    p : int
        Interior half-bandwidth.
    free_dof : Symbol
        The single closure DOF to be fixed by conservation.
    psi : Symbol
        The cut-cell parameter (kept symbolic throughout).
    nu : int
        Derivative order (only ``nu == 1`` is exercised; sets the wall flux).

    Returns
    -------
    Expr
        The solved ``free_dof`` expression in (psi, remaining closure params).
    """
    n_rows = len(closure_rows)
    T = len(closure_rows[0])

    # Realized left-matrix columns.  The closure block spans T-frame cols 0..T-1.
    # The interior band rows pick up where the closure block ends: the first band
    # row is centered on T-frame column T-1 (the grid column just past the last
    # closure grid point), so the fully-covered interior grid columns 1..T-2
    # receive their complete stencil once the band is added.  Extend p columns
    # beyond so every covered column reaches its full footprint.
    n_band = T  # enough rows to cover cols 1..T-2 plus the next p columns
    NC = T + n_band  # column count of the realized left-matrix
    band_center0 = T - 1  # first interior band row centered at T-frame column T-1

    def _band_row(center: int) -> list[Expr]:
        row = [S.Zero] * NC
        for k, c in enumerate(interior_coeffs):
            col = center + (k - p)
            if 0 <= col < NC:
                row[col] = c
        return row

    def _pad(row: list[Expr]) -> list[Expr]:
        return list(row) + [S.Zero] * (NC - len(row))

    matrix = [_pad(r) for r in closure_rows]
    for m in range(n_band):
        matrix.append(_band_row(band_center0 + m))

    # One conservation weight per closure row; interior band rows weight 1.
    w_syms = list(symbols(f"w_0:{n_rows}"))
    weights = list(w_syms) + [S.One] * n_band

    def _col_sum(col: int) -> Expr:
        return cancel(sum(weights[i] * matrix[i][col] for i in range(len(matrix))))

    # conservationUniform: fully-covered interior grid columns are T-frame cols
    # 1..T-2 (grid points 0..T-3).  Solve them for the closure weights.
    interior_cols = list(range(1, T - 1))
    assert len(interior_cols) == n_rows, (
        f"interior column count {len(interior_cols)} != weight count {n_rows}"
    )
    interior_eqs = [_col_sum(j) for j in interior_cols]
    branches = solve(interior_eqs, w_syms, dict=True)
    assert len(branches) == 1, (
        f"Expected unique weight solution, got {len(branches)} branches"
    )
    weight_sol = branches[0]

    # conservationCutCell: the wall-column sum (col 0) is the boundary flux.  The
    # interior band rows contribute nothing to col 0 (they are centered past it),
    # so the flux is the weighted sum of the closure rows' wall entries.
    wall = cancel(
        sum(w_syms[i] * closure_rows[i][0] for i in range(n_rows)).subs(weight_sol)
    )
    flux_target = S.NegativeOne if nu == 1 else S.Zero
    dof_branches = solve(wall - flux_target, free_dof)
    assert len(dof_branches) == 1, (
        f"Expected unique DOF solution, got {len(dof_branches)} branches"
    )
    return cancel(dof_branches[0])
