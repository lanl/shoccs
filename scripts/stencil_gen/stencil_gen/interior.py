"""Interior stencil derivation for implicit finite difference schemes.

Given parameters (s, p, nu) representing the LHS bandwidth, RHS bandwidth,
and derivative order, derives exact rational coefficients by Taylor series
matching using SymPy linear algebra.
"""

from dataclasses import dataclass

from sympy import Matrix, Rational, factorial, symbols

from stencil_gen._util import solve_linear


@dataclass(frozen=True)
class StencilSpec:
    """Specification for an interior finite difference stencil."""

    s: int  # LHS half-bandwidth (0 = explicit, 1 = tridiagonal)
    p: int  # RHS half-bandwidth
    nu: int  # derivative order (1 or 2)

    @property
    def order(self) -> int:
        return 2 * (self.p + self.s)

    @property
    def n_unknowns(self) -> int:
        return self.p + self.s

    @property
    def scheme_name(self) -> str:
        if self.s == 0:
            prefix = "E"
        elif self.s == 1:
            prefix = "T"
        else:
            prefix = "P"
        return f"{prefix}{self.order}_{self.nu}"


@dataclass(frozen=True)
class InteriorCoefficients:
    """Exact rational coefficients for an interior stencil.

    gamma maps positive index j (1..p) to coefficient value.
    delta maps positive index k (1..s) to coefficient value.
    Negative indices are inferred from symmetry.
    """

    spec: StencilSpec
    gamma: dict[int, Rational]
    delta: dict[int, Rational]


def full_gamma_array(coeffs: InteriorCoefficients) -> list[Rational]:
    """Expand gamma to full array [gamma_{-p}, ..., gamma_0, ..., gamma_p]."""
    p = coeffs.spec.p
    nu = coeffs.spec.nu
    result = [Rational(0)] * (2 * p + 1)

    for j in range(1, p + 1):
        gj = coeffs.gamma[j]
        result[p + j] = gj
        if nu == 1:
            result[p - j] = -gj  # antisymmetry
        else:
            result[p - j] = gj  # symmetry

    if nu == 1:
        result[p] = Rational(0)
    else:
        # gamma_0 = -2 * sum(gamma_j)
        result[p] = Rational(-2) * sum(coeffs.gamma[j] for j in range(1, p + 1))

    return result


def full_delta_array(coeffs: InteriorCoefficients) -> list[Rational]:
    """Expand delta to full array [delta_{-s}, ..., delta_0, ..., delta_s]."""
    s = coeffs.spec.s
    if s == 0:
        return [Rational(1)]

    result = [Rational(0)] * (2 * s + 1)
    result[s] = Rational(1)  # delta_0 = 1
    for k in range(1, s + 1):
        dk = coeffs.delta[k]
        result[s + k] = dk
        result[s - k] = dk  # symmetry
    return result


def derive_interior(s: int, p: int, nu: int) -> InteriorCoefficients:
    """Derive exact rational interior stencil coefficients.

    Parameters
    ----------
    s : int  -- LHS half-bandwidth (0 = explicit, 1 = tridiagonal)
    p : int  -- RHS half-bandwidth (>= 1)
    nu : int -- derivative order (1 or 2)

    Returns
    -------
    InteriorCoefficients with exact Rational values.
    """
    if s < 0:
        raise ValueError(f"s must be >= 0, got {s}")
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    if nu not in (1, 2):
        raise ValueError(f"nu must be 1 or 2, got {nu}")

    spec = StencilSpec(s, p, nu)
    n = spec.n_unknowns

    gamma_syms = symbols(f"gamma_1:{p + 1}")
    delta_syms = symbols(f"delta_1:{s + 1}") if s > 0 else ()
    unknowns = list(gamma_syms) + list(delta_syms)

    # Build Taylor matching equations for m = 0 .. 2(p+s)-1.
    # Collect only non-trivial equations.
    A_rows = []
    b_vals = []

    for m in range(2 * (p + s) + nu):
        # --- RHS coefficient of f^(m)_i ---
        # rhs = sum_{j=-p}^{p} gamma_j * j^m / m!
        # Using symmetry, express in terms of gamma_1..gamma_p only.
        rhs_expr = Rational(0)

        if nu == 1:
            # antisymmetric: gamma_{-j} = -gamma_j, gamma_0 = 0
            # sum = sum_j gamma_j * (j^m - (-j)^m) / m!
            # = 0 for even m, = 2*sum_j gamma_j * j^m / m! for odd m
            if m % 2 == 1:
                inv_fact = Rational(1, factorial(m))
                for j in range(1, p + 1):
                    rhs_expr += 2 * gamma_syms[j - 1] * Rational(j**m) * inv_fact
        else:
            # nu == 2, symmetric: gamma_{-j} = gamma_j
            # sum = gamma_0 * 0^m / m! + 2*sum_j gamma_j * j^m / m!  for even m
            # = 0 for odd m
            if m % 2 == 0:
                inv_fact = Rational(1, factorial(m))
                # gamma_0 contributes only for m=0: gamma_0 * 0^0 / 0! = gamma_0
                if m == 0:
                    # gamma_0 = -2*sum(gamma_j), express in terms of unknowns
                    gamma_0_expr = -2 * sum(gamma_syms[j - 1] for j in range(1, p + 1))
                    rhs_expr += gamma_0_expr * inv_fact
                # j >= 1 terms
                for j in range(1, p + 1):
                    rhs_expr += 2 * gamma_syms[j - 1] * Rational(j**m) * inv_fact

        # --- LHS coefficient of f^(m)_i ---
        # lhs = sum_{k=-s}^{s} delta_k * k^(m-nu) / (m-nu)!  for m >= nu
        # Using symmetry: delta_{-k} = delta_k
        lhs_const = Rational(0)  # constant part (from delta_0 = 1)
        lhs_expr = Rational(0)  # symbolic part (from delta_1..delta_s)

        if m >= nu:
            q = m - nu
            inv_fact_q = Rational(1, factorial(q))
            # delta_0 = 1 contributes: 1 * 0^q / q! = (1 if q==0 else 0) / q!
            if q == 0:
                lhs_const = Rational(1)
            # k >= 1: delta_k * (k^q + (-k)^q) / q!
            # = 0 for odd q, = 2*delta_k * k^q / q! for even q
            if q % 2 == 0:
                for k in range(1, s + 1):
                    lhs_expr += 2 * delta_syms[k - 1] * Rational(k**q) * inv_fact_q

        # Equation: rhs_expr = lhs_const + lhs_expr
        # => rhs_expr - lhs_expr = lhs_const
        eq_expr = rhs_expr - lhs_expr
        eq_const = lhs_const

        # Skip trivial 0 = 0 equations
        if eq_expr == 0 and eq_const == 0:
            continue
        # Skip m=0 for nu=2 (gamma_0 constraint, automatically satisfied)
        if nu == 2 and m == 0:
            continue

        # Extract coefficients for the matrix row
        row = []
        for sym in unknowns:
            row.append(eq_expr.coeff(sym))
        A_rows.append(row)
        b_vals.append(eq_const)

    assert len(A_rows) == n, (
        f"Expected {n} non-trivial equations, got {len(A_rows)} "
        f"for (s={s}, p={p}, nu={nu})"
    )

    A = Matrix(A_rows)
    b = Matrix(b_vals)
    sol = solve_linear(A, b, unknowns)

    gamma_dict = {j: sol[gamma_syms[j - 1]] for j in range(1, p + 1)}
    delta_dict = {k: sol[delta_syms[k - 1]] for k in range(1, s + 1)}

    return InteriorCoefficients(spec=spec, gamma=gamma_dict, delta=delta_dict)
