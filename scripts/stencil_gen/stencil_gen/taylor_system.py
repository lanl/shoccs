"""Taylor system builder for boundary stencil derivation.

Constructs the Vandermonde-like linear system from Taylor expansion matching
for a single boundary row of an SBP finite difference operator.
"""

from sympy import Matrix, Rational, factorial


def _unit_rhs(n_eqs: int, nu: int) -> Matrix:
    """Column vector with 1 at row *nu* and 0 elsewhere (derivative RHS)."""
    return Matrix(n_eqs, 1, lambda k, _: Rational(1) if k == nu else Rational(0))


def _interp_rhs(n_eqs: int, y) -> Matrix:
    """Column vector rhs[k] = y**k / k! — the Taylor-evaluation functional at offset y.

    Derivation: f[i+y] = Σ_k f^(k)[i] y^k/k!, so to reproduce f[i] (and its
    moments) the coefficients must satisfy Σ_j c_j (j-i)^k = y^k, i.e.
    rhs[k] = y^k/k!.  This is the interpolation analogue of ``_unit_rhs``.
    ``factorial`` is already imported above.
    """
    return Matrix(n_eqs, 1, lambda k, _: y**k / factorial(k))


def build_interp_system(i: int, t: int, q: int, y) -> tuple[Matrix, Matrix]:
    """Interpolation analogue of ``build_taylor_system``.

    Reuses the Vandermonde matrix V verbatim (it is independent of the RHS) and
    swaps the RHS to the eval-at-y functional. Returns (V, rhs).
    """
    V, _ = build_taylor_system(i, t, q, nu=1)  # reuse the matrix only
    return V, _interp_rhs(q + 1, y)


def build_taylor_system(
    i: int,
    t: int,
    q: int,
    nu: int = 1,
) -> tuple[Matrix, Matrix]:
    """Build the Taylor matching system for boundary row i.

    Parameters
    ----------
    i : int
        Row index (0..r-1), the grid point being approximated.
    t : int
        Boundary stencil width (number of columns).
    q : int
        Polynomial order of the boundary scheme.
    nu : int
        Derivative order (1 for first derivative).

    Returns
    -------
    (V, rhs) where V is (q+1) x t and rhs is (q+1) x 1.
    """
    n_rows = q + 1
    V = Matrix.zeros(n_rows, t)

    for k in range(n_rows):
        inv_fact = Rational(1, factorial(k))
        for j in range(t):
            V[k, j] = Rational((j - i) ** k) * inv_fact

    return V, _unit_rhs(n_rows, nu)
