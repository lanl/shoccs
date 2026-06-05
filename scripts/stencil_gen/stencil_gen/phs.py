"""Polyharmonic Spline (PHS) + Polynomial finite difference weight computation.

Computes FD stencil weights by solving the augmented PHS+polynomial system:

    [Φ  P] [λ]   [d_Φ]
    [P' 0] [μ] = [d_P]

where Φ is the PHS kernel matrix, P enforces polynomial reproduction to
degree q, and the RHS encodes the derivative functional.  The weights λ
are the FD stencil coefficients.

The PHS order k controls the smoothness of the implicit interpolant:
- k=2 (r^3 in 1D): cubic-spline-like, provides dissipation
- Higher k: approaches polynomial interpolation (less dissipation)

Reference: Flyer, Fornberg, Bayona, Barnett (2016), "On the role of
polynomials in RBF-FD approximations."
"""

from __future__ import annotations

import numpy as np
from sympy import (
    Abs,
    Expr,
    Matrix,
    Rational,
    S,
    Symbol,
    cancel,
    diff,
    exp as sym_exp,
    factorial,
    sqrt as sym_sqrt,
)


# ---------------------------------------------------------------------------
# Tension spline kernel: φ(r;σ) = σ|r| - 1 + exp(-σ|r|)
# ---------------------------------------------------------------------------


def _tension_kernel_eval(r_val, sigma: float) -> float:
    """Evaluate tension spline kernel φ(r;σ) = σ|r| - 1 + exp(-σ|r|).

    Uses Taylor series for small σ|r| to avoid catastrophic cancellation.
    At σ=0 this reduces to |r|³·σ²/6, but in that limit the caller should
    use PHS k=2 directly.

    Parameters
    ----------
    r_val : float
        Signed distance.
    sigma : float
        Tension parameter (≥ 0).

    Returns
    -------
    float
        Kernel value.
    """
    ar = abs(float(r_val))
    s = float(sigma)

    if ar == 0.0:
        return 0.0

    z = s * ar  # dimensionless argument

    if z < 2.0:
        # Taylor: φ = (z² / 6)(1 + z²/20 + z⁴/840 + z⁶/60480 + z⁸/6652800)
        # times ar (since φ = ar³ σ²/6 * series in z²)
        # More precisely: φ = σ|r| - 1 + e^{-σ|r|}
        #   = z - 1 + (1 - z + z²/2 - z³/6 + z⁴/24 - ...)
        #   = z²/2 - z³/6 + z⁴/24 - z⁵/120 + z⁶/720 - ...
        #   = z²/2 (1 - z/3 + z²/12 - z³/60 + z⁴/360 - z⁵/2520 + z⁶/20160 - z⁷/181440)
        z2 = z * z
        # Horner form for 1 - z/3 + z²/12 - z³/60 + z⁴/360 - z⁵/2520 + z⁶/20160 - z⁷/181440
        series = 1.0 + z * (
            -1.0 / 3 + z * (
                1.0 / 12 + z * (
                    -1.0 / 60 + z * (
                        1.0 / 360 + z * (
                            -1.0 / 2520 + z * (
                                1.0 / 20160 + z * (-1.0 / 181440)
                            )
                        )
                    )
                )
            )
        )
        return z2 / 2.0 * series
    else:
        # Direct evaluation: σ|r| - 1 + exp(-σ|r|)
        return z - 1.0 + np.exp(-z)


def _tension_kernel_deriv(r_val, nu: int, sigma: float) -> float:
    """Evaluate D^nu φ(r;σ) at r = r_val for the tension spline kernel.

    Derivatives of φ(r;σ) = σ|r| - 1 + exp(-σ|r|):
      D¹φ = σ sign(r)(1 - exp(-σ|r|))
      D²φ = σ² exp(-σ|r|)   [plus a 2σδ(r) term, but r≠0 on our grids]

    For small σ|r|, Taylor series are used.

    Parameters
    ----------
    r_val : float
        Signed distance.
    nu : int
        Derivative order (1 or 2).
    sigma : float
        Tension parameter.

    Returns
    -------
    float
        D^nu φ evaluated at r_val.
    """
    r = float(r_val)
    s = float(sigma)
    ar = abs(r)
    z = s * ar

    if nu == 0:
        return _tension_kernel_eval(r_val, sigma)

    if nu == 1:
        if r == 0.0:
            return 0.0
        sgn = 1.0 if r > 0 else -1.0
        if z < 2.0:
            # D¹φ = σ sign(r)(1 - e^{-z}) = σ sign(r)(z - z²/2 + z³/6 - ...)
            # = σ sign(r) z (1 - z/2 + z²/6 - z³/24 + z⁴/120 - z⁵/720 + z⁶/5040 - z⁷/40320)
            series = 1.0 + z * (
                -1.0 / 2 + z * (
                    1.0 / 6 + z * (
                        -1.0 / 24 + z * (
                            1.0 / 120 + z * (
                                -1.0 / 720 + z * (
                                    1.0 / 5040 + z * (-1.0 / 40320)
                                )
                            )
                        )
                    )
                )
            )
            return sgn * s * z * series
        else:
            return sgn * s * (1.0 - np.exp(-z))

    if nu == 2:
        # D²φ = σ² exp(-σ|r|) for r ≠ 0
        # At r=0: D²φ = σ² (the limit from either side, ignoring the delta)
        if z < 2.0:
            # exp(-z) via Taylor: 1 - z + z²/2 - z³/6 + ...
            # σ² exp(-z) = σ² (1 - z + z²/2 - z³/6 + z⁴/24 - z⁵/120 + z⁶/720 - z⁷/5040)
            series = 1.0 + z * (
                -1.0 + z * (
                    1.0 / 2 + z * (
                        -1.0 / 6 + z * (
                            1.0 / 24 + z * (
                                -1.0 / 120 + z * (
                                    1.0 / 720 + z * (-1.0 / 5040)
                                )
                            )
                        )
                    )
                )
            )
            return s * s * series
        else:
            return s * s * np.exp(-z)

    raise NotImplementedError(f"Tension kernel derivative for nu={nu}")


def _kernel_eval(r, kernel: str, k: int | None = None, epsilon=None):
    """Evaluate RBF kernel φ(r).

    Parameters
    ----------
    r : Expr or numeric
        Signed distance.
    kernel : str
        ``"phs"``, ``"gaussian"``, or ``"multiquadric"``.
    k : int, optional
        PHS order (required for ``"phs"``).
    epsilon : numeric, optional
        Shape parameter (required for ``"gaussian"`` and ``"multiquadric"``).

    Returns
    -------
    Expr
        Kernel value φ(r).
    """
    if kernel == "phs":
        m = 2 * k - 1
        return _phi_val(r, m)
    elif kernel == "gaussian":
        return sym_exp(-(epsilon**2) * r**2)
    elif kernel == "multiquadric":
        return sym_sqrt(1 + (epsilon**2) * r**2)
    elif kernel == "tension":
        return _tension_kernel_eval(r, epsilon)
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")


def _kernel_deriv(r_val, nu: int, kernel: str, k: int | None = None, epsilon=None):
    """Evaluate D^nu φ(r) at r = r_val.

    Parameters
    ----------
    r_val : Expr or numeric
        Point at which to evaluate.
    nu : int
        Derivative order.
    kernel : str
        ``"phs"``, ``"gaussian"``, ``"multiquadric"``, or ``"tension"``.
    k : int, optional
        PHS order (required for ``"phs"``).
    epsilon : numeric, optional
        Shape parameter (required for ``"gaussian"``, ``"multiquadric"``,
        and ``"tension"``).

    Returns
    -------
    Expr
        D^nu φ(r) evaluated at r_val.
    """
    if kernel == "phs":
        m = 2 * k - 1
        return _eval_phs_deriv(r_val, nu, m)

    if kernel == "tension":
        return _tension_kernel_deriv(r_val, nu, epsilon)

    # For Gaussian and Multiquadric, use symbolic differentiation.
    r = Symbol("_rbf_r")
    if kernel == "gaussian":
        expr = sym_exp(-(epsilon**2) * r**2)
    elif kernel == "multiquadric":
        expr = sym_sqrt(1 + (epsilon**2) * r**2)
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")

    d_expr = diff(expr, r, nu)
    return d_expr.subs(r, r_val)


# ---------------------------------------------------------------------------
# Numeric (numpy) path for Gaussian / Multiquadric kernels
# ---------------------------------------------------------------------------


def _rbf_weights_numeric(
    points: list,
    x_eval,
    nu: int,
    q: int,
    kernel: str,
    epsilon: float,
) -> list[float]:
    """Compute RBF+poly FD weights using numpy.

    Used for Gaussian and Multiquadric kernels where exact symbolic
    computation is neither necessary nor efficient.

    Parameters
    ----------
    points : list
        Grid point locations.
    x_eval : numeric
        Evaluation point.
    nu : int
        Derivative order (1 or 2).
    q : int
        Polynomial degree for augmentation.
    kernel : str
        ``"gaussian"`` or ``"multiquadric"``.
    epsilon : float
        Shape parameter.

    Returns
    -------
    list of float
        FD weights.
    """
    n = len(points)
    n_poly = q + 1
    eps = float(epsilon)
    pts = np.array([float(p) for p in points])
    x0 = float(x_eval)

    # Kernel matrix Φ_{ij} = φ(x_i - x_j)
    diffs = pts[:, None] - pts[None, :]
    if kernel == "gaussian":
        Phi = np.exp(-(eps**2) * diffs**2)
    elif kernel == "multiquadric":
        Phi = np.sqrt(1 + (eps**2) * diffs**2)
    elif kernel == "tension":
        Phi = np.array(
            [[_tension_kernel_eval(diffs[i, j], eps) for j in range(n)]
             for i in range(n)]
        )
    else:
        raise ValueError(f"Unknown kernel for numeric path: {kernel!r}")

    # RHS: D^nu φ(x_eval - x_i)
    r = x0 - pts
    if kernel == "gaussian":
        base = np.exp(-(eps**2) * r**2)
        if nu == 0:
            dPhi = base
        elif nu == 1:
            dPhi = -2 * eps**2 * r * base
        elif nu == 2:
            dPhi = (4 * eps**4 * r**2 - 2 * eps**2) * base
        else:
            raise NotImplementedError(f"Gaussian derivative for nu={nu}")
    elif kernel == "multiquadric":
        s2 = 1 + (eps**2) * r**2
        if nu == 0:
            dPhi = np.sqrt(s2)
        elif nu == 1:
            dPhi = eps**2 * r / np.sqrt(s2)
        elif nu == 2:
            dPhi = eps**2 / s2**1.5
        else:
            raise NotImplementedError(f"Multiquadric derivative for nu={nu}")
    elif kernel == "tension":
        dPhi = np.array([_tension_kernel_deriv(r[i], nu, eps) for i in range(n)])
    else:
        raise NotImplementedError(f"Kernel {kernel!r} derivative for nu={nu}")

    # Polynomial matrix P_{ij} = x_j^i, i = 0 .. q
    P = np.zeros((n_poly, n))
    for i in range(n_poly):
        P[i, :] = pts**i

    # Polynomial RHS: D^nu x_eval^i
    dP = np.zeros(n_poly)
    for i in range(n_poly):
        if i >= nu:
            coeff = 1.0
            for j in range(nu):
                coeff *= i - j
            dP[i] = coeff * x0 ** (i - nu)

    # Assemble augmented system  [Φ P'; P 0] [λ; μ] = [dΦ; dP]
    N = n + n_poly
    A = np.zeros((N, N))
    A[:n, :n] = Phi
    A[:n, n:] = P.T
    A[n:, :n] = P

    b_vec = np.zeros(N)
    b_vec[:n] = dPhi
    b_vec[n:] = dP

    x = np.linalg.solve(A, b_vec)
    return list(x[:n])


def phs_stencil_weights(
    points: list,
    x_eval,
    nu: int,
    q: int,
    k: int = None,
    *,
    kernel: str = "phs",
    epsilon=None,
) -> list:
    """Compute FD weights using RBF+polynomial augmentation.

    Solves the augmented system:
        [Φ  P] [λ]   [d_Φ]
        [P' 0] [μ] = [d_P]

    Parameters
    ----------
    points : list of Expr/Rational
        Grid point locations {x_0, ..., x_{n-1}}.
    x_eval : Expr/Rational
        Point at which the derivative is evaluated.
    nu : int
        Derivative order (1 for first derivative, 2 for second, etc.).
    q : int
        Polynomial degree for augmentation.  The stencil will be exact
        for polynomials up to degree q.
    k : int, optional
        PHS order.  φ(r) = |r|^(2k-1).  Required when kernel="phs".
    kernel : str
        ``"phs"`` (default), ``"gaussian"``, or ``"multiquadric"``.
    epsilon : float, optional
        Shape parameter for Gaussian/Multiquadric kernels.

    Returns
    -------
    list
        FD weights [w_0, ..., w_{n-1}] such that
        f^(nu)(x_eval) ≈ Σ_j w_j f(x_j).
    """
    # --- Parameter validation and dispatch ---
    if kernel == "phs":
        if k is None:
            raise ValueError("k is required for PHS kernel")
        if k < nu:
            raise ValueError(f"PHS order k={k} must be >= nu={nu}")
    elif kernel in ("gaussian", "multiquadric", "tension"):
        if epsilon is None:
            raise ValueError(f"epsilon is required for {kernel} kernel")
        # σ=0 limit of tension spline is PHS k=2; dispatch to exact PHS path
        # to avoid the singular Φ matrix that tension produces at σ=0.
        if kernel == "tension" and abs(float(epsilon)) < 1e-14:
            sympy_weights = phs_stencil_weights(
                points, x_eval, nu, q, k=2, kernel="phs"
            )
            return [float(w) for w in sympy_weights]
        return _rbf_weights_numeric(points, x_eval, nu, q, kernel, epsilon)
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")

    # --- PHS path (exact SymPy computation) ---
    n = len(points)
    n_poly = q + 1  # number of polynomial basis functions: 1, x, x^2, ..., x^q

    # Build Φ matrix using kernel dispatch
    Phi = Matrix(
        n, n, lambda i, j: _kernel_eval(points[i] - points[j], "phs", k=k)
    )

    # Build P matrix: P_{ij} = x_j^i for i=0..q (polynomial basis)
    P = Matrix(n_poly, n, lambda i, j: points[j] ** i)

    # Build RHS: d_Φ_i = D^nu φ(x_eval - x_i)
    d_Phi = Matrix(
        n, 1, lambda i, _: _kernel_deriv(x_eval - points[i], nu, "phs", k=k)
    )

    # Build d_P: D^nu x_eval^i
    d_P = Matrix(n_poly, 1, lambda i, _: _monomial_deriv(x_eval, i, nu))

    # Assemble augmented system
    # Top block: [Φ | P'] [λ]   [d_Φ]
    # Bot block: [P | 0 ] [μ] = [d_P]
    Z = Matrix.zeros(n_poly, n_poly)
    A = Matrix([
        [Phi, P.T],
        [P, Z],
    ])
    b = Matrix([d_Phi, d_P])

    # Solve
    x = A.solve(b)

    # Extract weights (first n entries)
    weights = [cancel(x[i]) for i in range(n)]
    return weights


def _phi_val(r, m: int):
    """Compute |r|^m for a potentially symbolic r, with m odd."""
    if r == 0:
        return S.Zero
    return Abs(r) ** m


def _eval_phs_deriv(r_val, nu: int, m: int):
    """Evaluate D^nu |r|^m at a specific r value.

    For concrete (numeric) r: differentiates |r|^m = sign(r)^m * r^m
    using the fact that m is odd.

    For r=0: returns 0 (valid when m >= 2*nu - 1, which we assume).
    """
    if r_val == 0:
        return S.Zero

    # For nonzero concrete r, |r|^m is smooth (m odd, m >= 1).
    # D^nu |r|^m at r:
    # Use the identity for m odd: |r|^m = r * (r^2)^((m-1)/2)
    # Differentiate: use falling factorial coefficient times appropriate power.
    #
    # Explicit formula for |r|^m derivatives (m odd, r ≠ 0):
    #   For even nu: D^nu |r|^m = P(m, nu) * |r|^(m-nu)
    #   For odd nu:  D^nu |r|^m = P(m, nu) * sign(r) * |r|^(m-nu)
    #                            = P(m, nu) * r * |r|^(m-nu-1)  [since m-nu-1 is even]
    # where P(m, nu) = m * (m-1) * ... * (m-nu+1) = m! / (m-nu)!
    #
    # But this only works when m-nu >= 0. For m-nu < 0 we'd need distributions.
    # We require k >= nu, so m = 2k-1 >= 2nu-1, hence m-nu >= nu-1 >= 0.

    falling = S.One
    for j in range(nu):
        falling *= (m - j)

    remaining_exp = m - nu

    if isinstance(r_val, (int, float)) or r_val.is_number:
        abs_r = Abs(r_val)
        sign_r = S.One if r_val > 0 else S.NegativeOne
    else:
        # Symbolic: use Abs
        abs_r = Abs(r_val)
        # For symbolic, we need sign. In cut-cell grids, wall offset is negative.
        # We'll use Abs and handle sign via piecewise or assume known sign.
        # For now, use the r * |r|^(p-1) trick for odd powers.
        if remaining_exp == 0:
            if nu % 2 == 0:
                return falling
            else:
                # sign(r) * falling — for symbolic r, return r/|r| * falling
                # But |r|^0 = 1 and we need sign(r).
                # Use r_val / Abs(r_val) if nonzero
                return falling * r_val / Abs(r_val)
        if nu % 2 == 0:
            return falling * abs_r ** remaining_exp
        else:
            # Odd nu: need sign(r) * |r|^remaining_exp
            # = r * |r|^(remaining_exp - 1) since remaining_exp is even (m-nu, m odd, nu odd => even)
            return falling * r_val * abs_r ** (remaining_exp - 1)

    if nu % 2 == 0:
        return falling * abs_r ** remaining_exp
    else:
        return falling * sign_r * abs_r ** remaining_exp


def _monomial_deriv(x_eval, degree: int, nu: int):
    """Compute D^nu (x^degree) evaluated at x_eval."""
    if degree < nu:
        return S.Zero
    # D^nu x^d = d!/(d-nu)! * x^(d-nu)
    coeff = S.One
    for j in range(nu):
        coeff *= (degree - j)
    return coeff * x_eval ** (degree - nu)


# ---------------------------------------------------------------------------
# Convenience wrappers for uniform and cut-cell grids
# ---------------------------------------------------------------------------


def uniform_interior_weights(p: int, nu: int, k: int, q: int) -> list:
    """Compute interior FD weights on a uniform grid using PHS+poly.

    The stencil uses 2p+1 points centered at 0: {-p, ..., -1, 0, 1, ..., p}.
    """
    points = [Rational(j) for j in range(-p, p + 1)]
    x_eval = Rational(0)
    return phs_stencil_weights(points, x_eval, nu, q, k)


def uniform_boundary_weights(i: int, t: int, nu: int, k: int, q: int) -> list:
    """Compute boundary row i FD weights on a uniform grid using PHS+poly.

    The stencil uses t points: {0, 1, ..., t-1}, evaluating D^nu at grid
    point i.
    """
    points = [Rational(j) for j in range(t)]
    x_eval = Rational(i)
    return phs_stencil_weights(points, x_eval, nu, q, k)


def uniform_interior_weights_rbf(
    p: int, nu: int, q: int, epsilon: float, kernel: str = "gaussian"
) -> list[float]:
    """Compute interior FD weights on a uniform grid using RBF+poly.

    The stencil uses 2p+1 points centered at 0: {-p, ..., -1, 0, 1, ..., p}.
    """
    points = [Rational(j) for j in range(-p, p + 1)]
    x_eval = Rational(0)
    return phs_stencil_weights(points, x_eval, nu, q, kernel=kernel, epsilon=epsilon)


def uniform_boundary_weights_rbf(
    i: int, t: int, nu: int, q: int, epsilon: float, kernel: str = "gaussian"
) -> list[float]:
    """Compute boundary row i FD weights on a uniform grid using RBF+poly.

    The stencil uses t points: {0, 1, ..., t-1}, evaluating D^nu at grid
    point i.
    """
    points = [Rational(j) for j in range(t)]
    x_eval = Rational(i)
    return phs_stencil_weights(points, x_eval, nu, q, kernel=kernel, epsilon=epsilon)


def uniform_interior_weights_tension(
    p: int, nu: int, q: int, sigma: float
) -> list[float]:
    """Compute interior FD weights on a uniform grid using tension spline+poly.

    The stencil uses 2p+1 points centered at 0: {-p, ..., -1, 0, 1, ..., p}.
    """
    return uniform_interior_weights_rbf(p, nu, q, sigma, kernel="tension")


def uniform_boundary_weights_tension(
    i: int, t: int, nu: int, q: int, sigma: float
) -> list[float]:
    """Compute boundary row i FD weights using tension spline+poly.

    The stencil uses t points: {0, 1, ..., t-1}, evaluating D^nu at grid
    point i.  The tension parameter sigma controls kernel shape (sigma=0
    gives PHS k=2 limit).
    """
    return uniform_boundary_weights_rbf(i, t, nu, q, sigma, kernel="tension")


# ---------------------------------------------------------------------------
# Differentiation matrix and eigenvalue diagnostics (Phase 29.6)
# ---------------------------------------------------------------------------


def build_diff_matrix_rbf(
    n: int,
    p: int,
    q: int,
    epsilon: float,
    kernel: str = "gaussian",
    nu: int = 1,
    nextra: int = 0,
) -> np.ndarray:
    """Build n×n differentiation matrix with RBF boundary stencils.

    Interior rows use classical centered 2p+1 FD stencils.  Left and right
    boundary rows use RBF+polynomial stencils.  Right boundary rows are the
    antisymmetric (nu odd) or symmetric (nu even) reflection of the left.

    Parameters
    ----------
    n : int
        Grid size (number of points).
    p : int
        Interior half-bandwidth (interior stencil width = 2p+1).
    q : int
        Polynomial degree for boundary RBF augmentation.
    epsilon : float
        RBF shape parameter.
    kernel : str
        RBF kernel type: ``"gaussian"``, ``"multiquadric"``, or ``"tension"``.
        For ``"tension"``, *epsilon* is the tension parameter σ.
    nu : int
        Derivative order (1 or 2).
    nextra : int
        Extra boundary rows/columns (matches TEMO nextra parameter).

    Returns
    -------
    np.ndarray
        n×n differentiation matrix.
    """
    from stencil_gen.interior import derive_interior, full_gamma_array
    from stencil_gen.temo import compute_dimensions

    dims = compute_dimensions(p, q, 0, nextra, nu)
    r, t = dims.r, dims.t

    if n < 2 * r:
        raise ValueError(f"Grid too small: n={n} < 2*r={2*r}")
    if t > n:
        raise ValueError(f"Boundary stencil wider than grid: t={t} > n={n}")

    # Classical interior weights
    interior_coeffs = derive_interior(0, p, nu)
    interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

    D = np.zeros((n, n))

    # Left boundary rows: row i uses t points {0, ..., t-1}
    for i in range(r):
        w = uniform_boundary_weights_rbf(i, t, nu, q, epsilon, kernel=kernel)
        for j in range(t):
            D[i, j] = w[j]

    # Interior rows: centered 2p+1 stencil
    for i in range(r, n - r):
        for k_idx, j in enumerate(range(i - p, i + p + 1)):
            D[i, j] = interior_w[k_idx]

    # Right boundary rows: antisymmetric reflection for odd nu, symmetric for even
    sign = (-1.0) ** nu
    for i in range(r):
        w = uniform_boundary_weights_rbf(i, t, nu, q, epsilon, kernel=kernel)
        row = n - 1 - i
        for j in range(t):
            col = n - 1 - j
            D[row, col] = sign * w[j]

    return D


def build_diff_matrix_mixed_epsilon(
    n: int,
    p: int,
    q: int,
    epsilons: list[float],
    kernel: str = "gaussian",
    nu: int = 1,
    nextra: int = 0,
) -> np.ndarray:
    """Build n×n differentiation matrix with per-row RBF shape parameters.

    Like :func:`build_diff_matrix_rbf`, but each boundary row can use a
    different epsilon value.  This enables searching over mixed-epsilon
    configurations where a single epsilon is insufficient for stability.

    Parameters
    ----------
    n : int
        Grid size.
    p : int
        Interior half-bandwidth.
    q : int
        Polynomial degree for boundary RBF augmentation.
    epsilons : list of float
        Shape parameter per boundary row.  Length must equal r (the number
        of boundary rows per side).
    kernel : str
        RBF kernel type: ``"gaussian"``, ``"multiquadric"``, or ``"tension"``.
        For ``"tension"``, each entry of *epsilons* is a tension parameter σ.
    nu : int
        Derivative order (1 or 2).
    nextra : int
        Extra boundary rows/columns.

    Returns
    -------
    np.ndarray
        n×n differentiation matrix.
    """
    from stencil_gen.interior import derive_interior, full_gamma_array
    from stencil_gen.temo import compute_dimensions

    dims = compute_dimensions(p, q, 0, nextra, nu)
    r, t = dims.r, dims.t

    if len(epsilons) != r:
        raise ValueError(f"epsilons has length {len(epsilons)}, expected r={r}")
    if n < 2 * r:
        raise ValueError(f"Grid too small: n={n} < 2*r={2*r}")
    if t > n:
        raise ValueError(f"Boundary stencil wider than grid: t={t} > n={n}")

    # Classical interior weights
    interior_coeffs = derive_interior(0, p, nu)
    interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

    D = np.zeros((n, n))

    # Left boundary rows: each row i uses its own epsilon
    for i in range(r):
        w = uniform_boundary_weights_rbf(i, t, nu, q, epsilons[i], kernel=kernel)
        for j in range(t):
            D[i, j] = w[j]

    # Interior rows
    for i in range(r, n - r):
        for k_idx, j in enumerate(range(i - p, i + p + 1)):
            D[i, j] = interior_w[k_idx]

    # Right boundary rows: reflected
    sign = (-1.0) ** nu
    for i in range(r):
        w = uniform_boundary_weights_rbf(i, t, nu, q, epsilons[i], kernel=kernel)
        row = n - 1 - i
        for j in range(t):
            col = n - 1 - j
            D[row, col] = sign * w[j]

    return D


def max_real_eigenvalue(
    n: int,
    p: int,
    q: int,
    epsilon: float,
    kernel: str = "gaussian",
    nu: int = 1,
    nextra: int = 0,
) -> float:
    """Compute maximum real part of eigenvalues of the differentiation matrix.

    Parameters
    ----------
    n, p, q, epsilon, kernel, nu, nextra
        Passed to :func:`build_diff_matrix_rbf`.

    Returns
    -------
    float
        max Re(λ) over all eigenvalues of D.
    """
    D = build_diff_matrix_rbf(n, p, q, epsilon, kernel, nu, nextra)
    eigvals = np.linalg.eigvals(D)
    return float(np.max(np.real(eigvals)))


def stability_eigenvalue(
    n: int,
    p: int,
    q: int,
    epsilon: float,
    kernel: str = "gaussian",
    nu: int = 1,
    nextra: int = 0,
) -> float:
    """Return the maximum real part of the eigenvalues of -D with inflow BCs.

    For the advection equation u_t + u_x = 0, the semi-discrete system is
    du/dt = -D u.  Stability requires all eigenvalues of -D_bc to have
    non-positive real parts, where D_bc is D with the inflow row/column
    removed (Dirichlet at inflow, floating at outflow).

    Returns the maximum real part of eigenvalues of -D_bc.
    A non-positive return value means the scheme is stable.
    """
    D = build_diff_matrix_rbf(n, p, q, epsilon, kernel, nu, nextra)
    D_bc = D[1:, 1:]  # remove inflow (first row and column)
    eigs = np.linalg.eigvals(-D_bc)
    return float(np.max(eigs.real))


def stability_eigenvalue_from_matrix(D: np.ndarray) -> float:
    """Return the maximum real part of eigenvalues of -D_bc.

    D_bc = D[1:, 1:] removes the inflow row/column (Dirichlet at left,
    floating at right).  For the advection equation u_t + u_x = 0, the
    semi-discrete operator is -D.  Stability requires all eigenvalues of
    -D_bc to have non-positive real parts.

    Returns
    -------
    float
        max Re(eigenvalue of -D_bc).  Non-positive means stable.
    """
    D_bc = D[1:, 1:]
    return float(np.max(np.linalg.eigvals(-D_bc).real))


def build_diff_matrix_rbf_penalty(
    n: int,
    p: int,
    q: int,
    epsilon: float,
    kernel: str = "tension",
    nu: int = 1,
    nextra: int = 0,
    gamma: float = 0.0,
) -> np.ndarray:
    """Build n×n differentiation matrix with soft conservation penalty.

    Same as :func:`build_diff_matrix_rbf` but with an additional penalty
    that softly enforces conservation (zero column sums).  The boundary
    weights are adjusted to minimize:

        ‖δ‖² + γ ‖C(b₀ + δ) - target‖²
        subject to  P δ = 0   (polynomial exactness preserved)

    where b₀ are the standard RBF weights, C computes column sums of the
    left boundary block, target = −(interior column sums) so that the full
    matrix has zero column sums, and δ lies in the null space of the
    polynomial constraint matrix P.

    This distributes conservation across all boundary rows rather than
    dumping it on a single row (as TEMO does).

    Parameters
    ----------
    n, p, q, epsilon, kernel, nu, nextra
        Passed to :func:`build_diff_matrix_rbf`.
    gamma : float
        Conservation penalty weight.  γ=0 gives the standard RBF weights.
        As γ→∞, boundary weights approach conservation-enforced values
        while maintaining polynomial exactness.

    Returns
    -------
    np.ndarray
        n×n differentiation matrix.
    """
    # Build the standard (γ=0) matrix
    D_std = build_diff_matrix_rbf(n, p, q, epsilon, kernel, nu, nextra)

    if gamma <= 0:
        return D_std

    from stencil_gen.temo import compute_dimensions

    dims = compute_dimensions(p, q, 0, nextra, nu)
    r, t = dims.r, dims.t

    # Extract standard left boundary weights: B_std ∈ R^{r×t}
    B_std = D_std[:r, :t].copy()
    b_0 = B_std.ravel()  # R^{r*t}, row-major

    # Polynomial matrix P_{d,j} = j^d, d=0..q, j=0..t-1
    pts = np.arange(t, dtype=float)
    n_poly = q + 1
    P_mat = np.zeros((n_poly, t))
    for d in range(n_poly):
        P_mat[d, :] = pts**d

    # Null space of P_mat via SVD (orthonormal basis)
    _, _, Vt = np.linalg.svd(P_mat, full_matrices=True)
    null_dim = t - n_poly
    Z_single = Vt[n_poly:, :].T  # t × null_dim

    # Block-diagonal null space for all r rows: Z_block ∈ R^{r*t × r*null_dim}
    Z_block = np.zeros((r * t, r * null_dim))
    for i in range(r):
        Z_block[i * t : (i + 1) * t, i * null_dim : (i + 1) * null_dim] = Z_single

    # Conservation constraint: C ∈ R^{t × r*t}
    # C[j, i*t + j] = 1  ⟹  (C b)[j] = Σᵢ B[i,j] = left boundary column sum
    C = np.zeros((t, r * t))
    for j in range(t):
        for i in range(r):
            C[j, i * t + j] = 1.0

    # Interior column sums for columns 0..t-1
    interior_col_sums = np.sum(D_std[r : n - r, :t], axis=0)

    # Target: left boundary column sums = −interior column sums
    # so that total column sum = 0 for each column
    target = -interior_col_sums

    # Conservation deficit at standard solution
    r_0 = target - C @ b_0

    # Solve via null-space projection:
    #   b = b₀ + Z α,  minimize ‖Z α‖² + γ ‖C Z α − r₀‖²
    #   ⟹  (I + γ GᵀG) α = γ Gᵀr₀   where G = C Z_block
    # Since Z has orthonormal columns, ZᵀZ = I.
    G = C @ Z_block
    M = np.eye(r * null_dim) + gamma * (G.T @ G)
    rhs_vec = gamma * (G.T @ r_0)
    alpha = np.linalg.solve(M, rhs_vec)

    # Adjusted boundary weights
    b_new = b_0 + Z_block @ alpha
    B_new = b_new.reshape(r, t)

    # Assemble D with adjusted left boundary
    D = D_std.copy()
    D[:r, :t] = B_new

    # Right boundary: reflected (automatically satisfies conservation by symmetry)
    sign = (-1.0) ** nu
    for i in range(r):
        for j in range(t):
            D[n - 1 - i, n - 1 - j] = sign * B_new[i, j]

    return D


def cut_cell_weights(
    i: int,
    T: int,
    nu: int,
    k: int,
    q: int,
    psi,
) -> list:
    """Compute cut-cell FD weights using PHS+poly.

    The grid has T points: {-psi, 0, 1, ..., T-2} (wall + T-1 grid points).
    Evaluates D^nu at grid point i (where grid point 0 is at position 0).

    Parameters
    ----------
    i : int
        Grid point index for derivative evaluation.
    T : int
        Total number of points (including wall).
    nu : int
        Derivative order.
    k : int
        PHS order.
    q : int
        Polynomial degree.
    psi : Symbol or Rational
        Wall offset parameter (0 < psi <= 1).

    Returns
    -------
    list of Expr
        FD weights, potentially symbolic in psi.
    """
    # Points: wall at -psi, then grid points 0, 1, ..., T-2
    points = [-psi] + [Rational(j) for j in range(T - 1)]
    x_eval = Rational(i)
    return phs_stencil_weights(points, x_eval, nu, q, k)


# ---------------------------------------------------------------------------
# B-spline knot-parameterized FD weights
# ---------------------------------------------------------------------------


def bspline_fd_weights(
    x: np.ndarray,
    x_eval: float,
    nu: int,
    k: int,
    interior_knots: np.ndarray,
) -> np.ndarray:
    """Compute FD weights from B-spline interpolation with given knots.

    Builds a B-spline space of degree k with a clamped knot vector
    (k+1 repeated knots at each end, plus the given interior knots),
    forms the collocation matrix at the data sites x, evaluates the
    derivative of the basis functions at x_eval, and solves for the
    FD weights.

    Parameters
    ----------
    x : array_like, shape (n,)
        Data sites (grid points), must be sorted.
    x_eval : float
        Point at which to approximate the derivative.
    nu : int
        Derivative order (1, 2, ...).
    k : int
        B-spline degree.  Must satisfy k >= nu.  Polynomial reproduction
        is exact up to degree k.
    interior_knots : array_like, shape (n - k - 1,)
        Interior knot positions.  Must be sorted and lie strictly inside
        (x[0], x[-1]).  The number of interior knots must equal n - k - 1
        so that the B-spline space has dimension n (one basis function per
        data point).

    Returns
    -------
    np.ndarray, shape (n,)
        FD weights w such that f^(nu)(x_eval) ≈ Σ w_j f(x_j).
    """
    from scipy.interpolate import BSpline

    x = np.asarray(x, dtype=float)
    interior_knots = np.asarray(interior_knots, dtype=float)
    n = len(x)

    # Build clamped knot vector: k+1 copies of x[0], interior, k+1 copies of x[-1]
    t = np.concatenate([
        np.full(k + 1, x[0]),
        interior_knots,
        np.full(k + 1, x[-1]),
    ])

    # Identity coefficients: column j gives basis function B_j
    c = np.eye(n)
    spl = BSpline(t, c, k, extrapolate=True)

    # Collocation matrix A[i,j] = B_j(x[i])
    A = spl(x)  # (n, n)

    # Derivative vector d[j] = D^nu B_j(x_eval)
    d = spl(np.atleast_1d(x_eval), nu=nu).ravel()  # (n,)

    # Weights: A^T w = d
    return np.linalg.solve(A.T, d)


def bspline_boundary_weights(
    i: int,
    t: int,
    nu: int,
    k: int,
    interior_knots: np.ndarray,
) -> np.ndarray:
    """Boundary row i FD weights using B-spline interpolation.

    Convenience wrapper for uniform grid {0, 1, ..., t-1}.
    """
    x = np.arange(t, dtype=float)
    return bspline_fd_weights(x, float(i), nu, k, interior_knots)


def build_diff_matrix_bspline(
    n: int,
    p: int,
    q: int,
    interior_knots: np.ndarray,
    nu: int = 1,
) -> np.ndarray:
    """Build n×n differentiation matrix with B-spline boundary stencils.

    Left and right boundary closures use B-spline interpolation with the
    given interior knots.  The interior uses the standard 2p+1-point
    centered FD stencil.

    Parameters
    ----------
    n : int
        Grid size.
    p : int
        Interior half-width.
    q : int
        B-spline degree (= polynomial reproduction order).
    interior_knots : array_like
        Interior knots for the boundary B-spline.  Length must be
        t - q - 1 where t = 2p + 1 (the boundary stencil width matching
        the interior stencil width).
    nu : int
        Derivative order.
    """
    from stencil_gen.interior import derive_interior, full_gamma_array

    t = 2 * p + 1  # boundary width = interior width
    r = p + 1      # number of boundary rows

    interior_knots = np.asarray(interior_knots, dtype=float)
    x_bdy = np.arange(t, dtype=float)

    # Boundary stencils
    bdy = []
    for i in range(r):
        w = bspline_fd_weights(x_bdy, float(i), nu, q, interior_knots)
        bdy.append(w)

    # Interior stencil
    int_result = derive_interior(0, p, nu)
    int_coeffs = [float(c) for c in full_gamma_array(int_result)]

    # Assemble
    D = np.zeros((n, n))
    for i in range(r):
        for j in range(t):
            D[i, j] = bdy[i][j]
    for i in range(r, n - r):
        for j_off in range(-p, p + 1):
            D[i, i + j_off] = int_coeffs[j_off + p]
    sign = (-1) ** nu
    for i in range(r):
        for j in range(t):
            D[n - 1 - i, n - 1 - j] = sign * bdy[i][j]

    return D
