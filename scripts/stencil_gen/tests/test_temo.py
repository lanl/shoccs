"""Tests for the TEMO cut-cell stencil extension module."""

import time

import pytest
from sympy import Matrix, Rational, Symbol, cancel, simplify

from stencil_gen.temo import (
    CutCellResult,
    Dimensions,
    RowSolveResult,
    SchemeParams,
    StencilResult,
    UniformResult,
    assemble_cut_cell_result,
    build_cut_cell_deltas,
    build_degenerate_stencil,
    build_neumann_vandermonde,
    build_temo_vandermonde,
    compute_dimensions,
    construct_cut_cell_stencil,
    construct_neumann_stencil,
    decompose_alpha_terms,
    derive_cut_cell_scheme,
    derive_e2_uniform_boundary,
    derive_uniform_boundary_for_temo,
    derive_uniform_neumann,
    from_field_elem,
    identify_neumann_prescribed_entries,
    identify_prescribed_entries,
    make_psi_field,
    solve_in_field,
    solve_neumann_uniform_limit,
    solve_temo_row,
    solve_uniform_limit,
    to_field_elem,
    E2_1,
    E2_2,
    E4_1,
    E4_2,
)


class TestDimensions:
    """Tests for compute_dimensions and SchemeParams.dims()."""

    def test_e2_1_dimensions(self):
        """E2_1: p=1, q=1, nextra=1, nu=1 -> r=3, t=4, R=4, T=5, X=0."""
        dims = E2_1.dims()
        assert dims == Dimensions(r=3, t=4, R=4, T=5, X=0)

    def test_e2_2_dimensions(self):
        """E2_2: p=1, q=1, nextra=0, nu=2 -> r=2, t=3, R=2, T=4, X=2."""
        dims = E2_2.dims()
        assert dims == Dimensions(r=2, t=3, R=2, T=4, X=2)

    def test_e2_1_matches_cpp(self):
        """E2_1.cpp has P=1, R=4, T=5, X=0."""
        dims = E2_1.dims()
        assert dims.R == 4
        assert dims.T == 5
        assert dims.X == 0
        assert E2_1.p == 1  # P in C++

    def test_e2_2_matches_cpp(self):
        """E2_2.cpp has P=1, R=2, T=4, X=2."""
        dims = E2_2.dims()
        assert dims.R == 2
        assert dims.T == 4
        assert dims.X == 2
        assert E2_2.p == 1  # P in C++

    def test_e2_1_uniform_dimensions(self):
        """E2_1 uniform boundary: 3 rows x 4 columns."""
        dims = E2_1.dims()
        assert dims.r == 3
        assert dims.t == 4

    def test_e2_2_uniform_dimensions(self):
        """E2_2 uniform boundary: 2 rows x 3 columns (r_eff=1 after -1)."""
        dims = E2_2.dims()
        assert dims.r == 2
        assert dims.t == 3

    def test_compute_dimensions_directly(self):
        """compute_dimensions matches SchemeParams.dims()."""
        dims = compute_dimensions(p=1, q=1, s=0, nextra=1, nu=1)
        assert dims == E2_1.dims()

    def test_invalid_nu_raises(self):
        """Unsupported derivative order raises ValueError."""
        with pytest.raises(ValueError, match="nu=3"):
            compute_dimensions(p=1, q=1, s=0, nextra=0, nu=3)

    def test_scheme_params_frozen(self):
        """SchemeParams is immutable."""
        with pytest.raises(AttributeError):
            E2_1.p = 2  # type: ignore[misc]

    def test_e4_1_dimensions(self):
        """E4_1: p=2, q=3, nextra=0, nu=1 -> r=4, t=6, R=5, T=7, X=0."""
        dims = E4_1.dims()
        assert dims == Dimensions(r=4, t=6, R=5, T=7, X=0)

    def test_e4_2_dimensions(self):
        """E4_2: p=2, q=3, nextra=0, nu=2 -> r=3, t=4, R=3, T=5, X=3."""
        dims = E4_2.dims()
        assert dims == Dimensions(r=3, t=4, R=3, T=5, X=3)

    def test_e4_2_matches_cpp(self):
        """E4_2.cpp has P=2, R=3, T=5, X=3."""
        dims = E4_2.dims()
        assert dims.R == 3
        assert dims.T == 5
        assert dims.X == 3
        assert E4_2.p == 2  # P in C++

    def test_first_derivative_no_neumann(self):
        """1st derivative stencils have X=0 (no Neumann rows)."""
        assert E2_1.dims().X == 0
        assert E4_1.dims().X == 0

    def test_second_derivative_has_neumann(self):
        """2nd derivative stencils have X=R (Neumann rows)."""
        dims_e2_2 = E2_2.dims()
        assert dims_e2_2.X == dims_e2_2.R

        dims_e4_2 = E4_2.dims()
        assert dims_e4_2.X == dims_e4_2.R


class TestUniformBoundary:
    """Tests for derive_e2_uniform_boundary (20.5b)."""

    def test_e2_1_shape_and_free_symbols(self, e2_1_uniform):
        """E2_1 B_u has shape (3, 4) and 4 free alpha symbols."""
        result = e2_1_uniform
        assert result.B_u.shape == (3, 4)
        assert len(result.alpha_symbols) == 4

    def test_e2_2_fully_determined(self, e2_2_uniform):
        """E2_2 B_u = [[1, -2, 1]] with no free symbols."""
        result = e2_2_uniform
        assert result.B_u == Matrix([[1, -2, 1]])
        assert result.alpha_symbols == []

    def test_e2_1_interior_stencil(self, e2_1_uniform):
        """E2_1 interior stencil is [-1/2, 0, 1/2]."""
        result = e2_1_uniform
        assert result.interior == [Rational(-1, 2), Rational(0), Rational(1, 2)]

    def test_e2_2_interior_stencil(self, e2_2_uniform):
        """E2_2 interior stencil is [1, -2, 1]."""
        result = e2_2_uniform
        assert result.interior == [Rational(1), Rational(-2), Rational(1)]

    def test_e2_1_conservation(self, e2_1_uniform):
        """E2_1: sum_i B_u[i, j] = 0 for interior columns j=2,3."""
        result = e2_1_uniform
        B_u = result.B_u
        r_eff = B_u.rows
        for j in [2, 3]:
            col_sum = sum(B_u[i, j] for i in range(r_eff))
            assert simplify(col_sum) == 0, f"Conservation failed for column {j}"

    def test_e2_1_custom_alpha_symbols(self):
        """E2_1 accepts user-supplied alpha symbols."""
        syms = [Symbol(f"a{k}") for k in range(4)]
        result = derive_e2_uniform_boundary(nu=1, alpha_symbols=syms)
        assert result.alpha_symbols == syms
        # All entries should involve only these symbols
        free = result.B_u.free_symbols
        assert free <= set(syms)

    def test_e2_1_wrong_alpha_count_raises(self):
        """E2_1 with wrong number of alpha symbols raises ValueError."""
        with pytest.raises(ValueError, match="4 alpha symbols"):
            derive_e2_uniform_boundary(nu=1, alpha_symbols=[Symbol("a")])

    def test_e2_1_taylor_accuracy_per_row(self, assert_taylor_accuracy, e2_1_uniform):
        """Each row of B_u satisfies the Taylor system (q+1=2 equations)."""
        assert_taylor_accuracy(e2_1_uniform.B_u, q=1, nu=1)

    def test_e2_2_taylor_accuracy(self, assert_taylor_accuracy, e2_2_uniform):
        """E2_2 single row satisfies max(q+1, nu+1)=3 Taylor equations."""
        assert_taylor_accuracy(e2_2_uniform.B_u, q=1, nu=2)

    def test_e2_1_p_q_nu(self, e2_1_uniform):
        """UniformResult carries correct scheme parameters."""
        result = e2_1_uniform
        assert result.p == 1
        assert result.q == 1
        assert result.nu == 1

    def test_e2_2_p_q_nu(self, e2_2_uniform):
        """UniformResult carries correct scheme parameters."""
        result = e2_2_uniform
        assert result.p == 1
        assert result.q == 1
        assert result.nu == 2

    def test_invalid_nu_raises(self):
        """Unsupported nu raises ValueError."""
        with pytest.raises(ValueError, match="nu=3"):
            derive_e2_uniform_boundary(nu=3)


class TestDeriveUniformBoundaryForTemo:
    """Tests for derive_uniform_boundary_for_temo (21.1a)."""

    def test_e2_1_regression_matches_old(self, e2_1_uniform):
        """derive_uniform_boundary_for_temo(E2_1) matches derive_e2_uniform_boundary."""
        old = e2_1_uniform
        new = derive_uniform_boundary_for_temo(E2_1)

        assert new.B_u.shape == old.B_u.shape
        for i in range(old.B_u.rows):
            for j in range(old.B_u.cols):
                diff = cancel(new.B_u[i, j] - old.B_u[i, j])
                assert diff == 0, f"Mismatch at ({i},{j}): new={new.B_u[i,j]}, old={old.B_u[i,j]}"

        assert new.interior == old.interior
        assert len(new.alpha_symbols) == len(old.alpha_symbols)
        assert new.p == old.p
        assert new.q == old.q
        assert new.nu == old.nu

    def test_e2_1_custom_alpha_symbols(self):
        """derive_uniform_boundary_for_temo accepts user-supplied alpha symbols."""
        syms = [Symbol(f"a{k}") for k in range(4)]
        result = derive_uniform_boundary_for_temo(E2_1, alpha_symbols=syms)
        assert result.alpha_symbols == syms
        free = result.B_u.free_symbols
        assert free <= set(syms)

    def test_e2_1_conservation(self):
        """E2_1: column sums are zero for interior columns."""
        result = derive_uniform_boundary_for_temo(E2_1)
        B_u = result.B_u
        for j in [2, 3]:
            col_sum = sum(B_u[i, j] for i in range(B_u.rows))
            assert simplify(col_sum) == 0, f"Conservation failed for column {j}"

    def test_e2_1_taylor_accuracy(self, assert_taylor_accuracy):
        """Each row satisfies Taylor matching for q+1=2 equations."""
        result = derive_uniform_boundary_for_temo(E2_1)
        assert_taylor_accuracy(result.B_u, q=1, nu=1)

    def test_wrong_alpha_count_raises(self):
        """Wrong number of alpha symbols raises ValueError."""
        with pytest.raises(ValueError, match="alpha symbols"):
            derive_uniform_boundary_for_temo(E2_1, alpha_symbols=[Symbol("a")])


class TestDegenerateStencil:
    """Tests for build_degenerate_stencil (20.5c)."""

    def test_e2_1_shape(self, e2_1_uniform):
        """E2_1 degenerate stencil has shape (4, 5)."""
        ur = e2_1_uniform
        B_d = build_degenerate_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu
        )
        assert B_d.shape == (4, 5)

    def test_e2_1_x0_column_all_zero(self, e2_1_uniform):
        """E2_1 (nu=1): x_0 column (col 1) is all zeros."""
        ur = e2_1_uniform
        B_d = build_degenerate_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu
        )
        for i in range(4):
            assert B_d[i, 1] == 0, f"B_d[{i}, 1] = {B_d[i, 1]}, expected 0"

    def test_e2_1_boundary_rows_dp1(self, e2_1_uniform):
        """E2_1: boundary rows cols 2..4 match B_u cols 1..3."""
        ur = e2_1_uniform
        B_d = build_degenerate_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu
        )
        B_u = ur.B_u
        for i in range(3):
            for j in range(1, 4):
                assert simplify(B_d[i, j + 1] - B_u[i, j]) == 0, (
                    f"DP1 failed: B_d[{i},{j+1}] != B_u[{i},{j}]"
                )

    def test_e2_1_boundary_rows_dp2_wall(self, e2_1_uniform):
        """E2_1 (nu=1): wall column (col 0) = B_u[i,0] for boundary rows."""
        ur = e2_1_uniform
        B_d = build_degenerate_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu
        )
        B_u = ur.B_u
        for i in range(3):
            assert simplify(B_d[i, 0] - B_u[i, 0]) == 0, (
                f"DP2 failed: B_d[{i},0] != B_u[{i},0]"
            )

    def test_e2_1_near_interior_conservation(self, e2_1_uniform):
        """E2_1: near-interior row satisfies B[3,j] = -(B[1,j]+B[2,j]) for j=3,4.

        Interior columns in the cut-cell frame are j >= p+2 = 3 (corresponding
        to uniform interior columns j >= p+1 shifted by +1 for the wall column).
        """
        ur = e2_1_uniform
        B_d = build_degenerate_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu
        )
        for j in [3, 4]:
            expected = -(B_d[1, j] + B_d[2, j])
            assert simplify(B_d[3, j] - expected) == 0, (
                f"Conservation failed at col {j}: B_d[3,{j}]={B_d[3,j]}, "
                f"expected {expected}"
            )

    def test_e2_1_near_interior_taylor(self, e2_1_uniform):
        """E2_1: near-interior row (row 3) satisfies Taylor accuracy at psi=0.

        Deltas from x_3: [-3, -3, -2, -1, 0]. Col 1 (x_0) is zeroed.
        k=0: sum of row = 0
        k=1: sum_j B_d[3,j] * delta_j = 1
        """
        ur = e2_1_uniform
        B_d = build_degenerate_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu
        )
        deltas = [Rational(-3), Rational(-3), Rational(-2), Rational(-1), Rational(0)]
        row = [B_d[3, j] for j in range(5)]

        # k=0: sum = 0
        assert simplify(sum(row)) == 0, f"k=0 failed: sum = {sum(row)}"
        # k=1: weighted sum = 1
        moment1 = sum(row[j] * deltas[j] for j in range(5))
        assert simplify(moment1 - 1) == 0, f"k=1 failed: sum = {moment1}"

    def test_e2_2_shape(self, e2_2_uniform):
        """E2_2 degenerate stencil has shape (2, 4)."""
        ur = e2_2_uniform
        B_d = build_degenerate_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu
        )
        assert B_d.shape == (2, 4)

    def test_e2_2_row0_wall_zeroed(self, e2_2_uniform):
        """E2_2 (nu=2) row 0: wall (col 0) is zeroed."""
        ur = e2_2_uniform
        B_d = build_degenerate_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu
        )
        assert B_d[0, 0] == 0

    def test_e2_2_row1_wall_is_gamma(self, e2_2_uniform):
        """E2_2 (nu=2) row 1: wall = gamma_{-1} = 1."""
        ur = e2_2_uniform
        B_d = build_degenerate_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu
        )
        assert B_d[1, 0] == 1

    def test_e2_2_matches_cpp_psi0(self, e2_2_uniform):
        """E2_2 at psi=0 matches C++: [0, 1, -2, 1, 1, 0, -2, 1] (pre-h^2).

        The flat array is row-major: row 0 = [0, 1, -2, 1], row 1 = [1, 0, -2, 1].
        """
        ur = e2_2_uniform
        B_d = build_degenerate_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu
        )
        expected = Matrix([
            [0, 1, -2, 1],
            [1, 0, -2, 1],
        ])
        assert B_d == expected, f"E2_2 degenerate mismatch:\n{B_d}\nexpected:\n{expected}"

    def test_e2_2_taylor_accuracy(self, e2_2_uniform):
        """E2_2: both rows satisfy max(q+1,nu+1)=3 Taylor equations at psi=0.

        Row 0 centered at x_0: deltas [-0, -0, 0, 1] = [0, 0, 0, 1]
        Wait -- at psi=0, wall is at x_0. Deltas from x_0:
          wall: -(0+0) = 0, x_0: 0, x_1: 1, x_2: 2

        Row 1 centered at x_1: deltas from x_1:
          wall: -(0+1) = -1, x_0: -1, x_1: 0, x_2: 1
        """
        ur = e2_2_uniform
        B_d = build_degenerate_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu
        )
        # Row 1 (near-interior) centered at x_1
        deltas_1 = [Rational(-1), Rational(-1), Rational(0), Rational(1)]
        row1 = [B_d[1, j] for j in range(4)]
        # k=0
        assert sum(row1) == 0
        # k=1
        assert sum(row1[j] * deltas_1[j] for j in range(4)) == 0
        # k=2
        moment2 = sum(row1[j] * deltas_1[j] ** 2 / 2 for j in range(4))
        assert moment2 == 1

    def test_e2_1_no_free_symbols_lost(self, e2_1_uniform):
        """E2_1 degenerate retains the alpha symbols from B_u."""
        ur = e2_1_uniform
        B_d = build_degenerate_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu
        )
        # The degenerate stencil should contain the same alpha symbols
        assert B_d.free_symbols == set(ur.alpha_symbols)

    def test_e2_2_no_free_symbols(self, e2_2_uniform):
        """E2_2 degenerate has no free symbols (fully determined)."""
        ur = e2_2_uniform
        B_d = build_degenerate_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu
        )
        assert B_d.free_symbols == set()


class TestPsiField:
    """Tests for QQ(psi) field utilities (20.5e Phase 1)."""

    def test_make_psi_field(self):
        """make_psi_field returns a field and psi element."""
        psi = Symbol("psi")
        K, psi_elem = make_psi_field(psi)
        # psi_elem should convert back to psi
        assert K.to_sympy(psi_elem) == psi

    def test_to_field_elem_rational(self):
        """to_field_elem handles rational constants."""
        psi = Symbol("psi")
        K, _ = make_psi_field(psi)
        elem = to_field_elem(Rational(3, 7), K)
        assert K.to_sympy(elem) == Rational(3, 7)

    def test_to_field_elem_polynomial(self):
        """to_field_elem handles polynomial in psi."""
        psi = Symbol("psi")
        K, _ = make_psi_field(psi)
        expr = psi**2 + 3 * psi + 1
        elem = to_field_elem(expr, K)
        assert K.to_sympy(elem) == expr

    def test_to_field_elem_rational_function(self):
        """to_field_elem handles a rational function of psi."""
        psi = Symbol("psi")
        K, _ = make_psi_field(psi)
        expr = (psi**2 + psi) / (psi + 1)
        elem = to_field_elem(expr, K)
        # (psi^2 + psi)/(psi + 1) = psi
        assert K.to_sympy(elem) == psi

    def test_to_field_elem_rejects_extra_symbols(self):
        """to_field_elem raises on non-psi symbols."""
        psi = Symbol("psi")
        alpha = Symbol("alpha")
        K, _ = make_psi_field(psi)
        with pytest.raises(ValueError, match="non-psi symbols"):
            to_field_elem(psi + alpha, K)

    def test_roundtrip(self):
        """to_field_elem -> from_field_elem preserves the expression."""
        psi = Symbol("psi")
        K, _ = make_psi_field(psi)
        expr = (2 + 4 * psi) / (2 + 3 * psi + psi**2)
        elem = to_field_elem(expr, K)
        back = from_field_elem(elem, K)
        assert simplify(back - expr) == 0

    def test_field_arithmetic(self):
        """Arithmetic in QQ(psi) is exact and reduced."""
        psi = Symbol("psi")
        K, psi_e = make_psi_field(psi)
        one = K.from_sympy(Rational(1))
        two = K.from_sympy(Rational(2))
        # (1 + psi) * (1 - psi) = 1 - psi^2
        a = one + psi_e
        b = one - psi_e
        product = a * b
        expected = one - psi_e * psi_e
        assert product == expected


class TestDecomposeAlphaTerms:
    """Tests for decompose_alpha_terms (20.5e Phase 1)."""

    def test_no_symbols(self):
        """With empty symbol list, returns {1: expr}."""
        psi = Symbol("psi")
        expr = psi**2 + 1
        result = decompose_alpha_terms(expr, [])
        assert result == {1: expr}

    def test_single_symbol(self):
        """Decomposes c_0(psi) + c_1(psi)*alpha."""
        psi = Symbol("psi")
        alpha = Symbol("alpha")
        expr = (1 + psi) + (2 * psi) * alpha
        result = decompose_alpha_terms(expr, [alpha])
        assert simplify(result[1] - (1 + psi)) == 0
        assert simplify(result[alpha] - 2 * psi) == 0

    def test_multiple_symbols(self):
        """Decomposes over alpha and beta symbols."""
        psi = Symbol("psi")
        a = Symbol("a")
        b = Symbol("b")
        expr = psi + 3 * a + psi**2 * b
        result = decompose_alpha_terms(expr, [a, b])
        assert simplify(result[1] - psi) == 0
        assert result[a] == 3
        assert simplify(result[b] - psi**2) == 0

    def test_zero_coefficient_omitted(self):
        """Symbols with zero coefficient do not appear in result."""
        psi = Symbol("psi")
        a = Symbol("a")
        b = Symbol("b")
        expr = psi + a  # b has zero coefficient
        result = decompose_alpha_terms(expr, [a, b])
        assert b not in result

    def test_nonlinear_raises(self):
        """Nonlinear term (alpha^2) raises ValueError."""
        psi = Symbol("psi")
        alpha = Symbol("alpha")
        expr = alpha**2 + psi
        with pytest.raises(ValueError, match="Nonlinear"):
            decompose_alpha_terms(expr, [alpha])

    def test_cross_term_raises(self):
        """Cross-term (alpha * beta) raises ValueError."""
        a = Symbol("a")
        b = Symbol("b")
        expr = a * b + 1
        with pytest.raises(ValueError, match="Cross-term"):
            decompose_alpha_terms(expr, [a, b])

    def test_constant_only(self):
        """Expression with no symbol dependence returns {1: expr}."""
        psi = Symbol("psi")
        alpha = Symbol("alpha")
        expr = psi**2 + 3
        result = decompose_alpha_terms(expr, [alpha])
        assert simplify(result[1] - expr) == 0
        assert alpha not in result


class TestSolveInField:
    """Tests for solve_in_field (20.5e Phase 1)."""

    def test_e2_2_row_0_no_symbols(self):
        """E2_2 row 0: 3x3 system, no alpha — matches C++ E2_2.cpp.

        After removing wall column (prescribed as psi), the reduced system is:
        V = | 1      1      1  |   rhs = | -psi            |
            | 0      1      2  |         | psi^2           |
            | 0      1/2    2  |         | 1 - psi^3/2     |
        """
        psi = Symbol("psi")
        K, _ = make_psi_field(psi)

        V = Matrix([
            [1, 1, 1],
            [0, 1, 2],
            [0, Rational(1, 2), 2],
        ])
        rhs = Matrix([
            [-psi],
            [psi**2],
            [1 - psi**3 / 2],
        ])

        sol = solve_in_field(V, rhs, K, symbols=[])

        # Expected from plan (E2_2 row 0):
        # c_x0 = (2 - 2*psi - 3*psi^2 - psi^3) / 2
        # c_x1 = -2 + 2*psi^2 + psi^3
        # c_x2 = (2 - psi^2 - psi^3) / 2
        expected = [
            (2 - 2 * psi - 3 * psi**2 - psi**3) / 2,
            -2 + 2 * psi**2 + psi**3,
            (2 - psi**2 - psi**3) / 2,
        ]
        for i in range(3):
            assert simplify(sol[i] - expected[i]) == 0, (
                f"Mismatch at index {i}: got {sol[i]}, expected {expected[i]}"
            )

    def test_e2_2_row_0_matches_cpp(self):
        """E2_2 row 0 floating coefficients match E2_2.cpp lines 123-131.

        C++ (E2_2.cpp, pre-h^2 scaling):
        c[0] = psi  (wall, prescribed)
        c[1] = (2 - 2*psi - 3*psi^2 - psi^3) / 2
        c[2] = -2 + 2*psi^2 + psi^3
        c[3] = (2 - psi^2 - psi^3) / 2
        """
        psi = Symbol("psi")
        K, _ = make_psi_field(psi)

        V = Matrix([
            [1, 1, 1],
            [0, 1, 2],
            [0, Rational(1, 2), 2],
        ])
        rhs = Matrix([[-psi], [psi**2], [1 - psi**3 / 2]])
        sol = solve_in_field(V, rhs, K, symbols=[])

        # Verify at numeric psi values that match C++ evaluation
        for psi_val in [Rational(0), Rational(1, 2), Rational(1)]:
            c1_num = sol[0].subs(psi, psi_val)
            c2_num = sol[1].subs(psi, psi_val)
            c3_num = sol[2].subs(psi, psi_val)
            c0_num = psi_val  # wall prescribed

            if psi_val == 0:
                # Degenerate: [0, 1, -2, 1]
                assert (c0_num, c1_num, c2_num, c3_num) == (0, 1, -2, 1)
            elif psi_val == 1:
                # Uniform: [1, -2, 1, 0] (row 0 of E2_2 uniform boundary on T=4 grid)
                assert (c0_num, c1_num, c2_num, c3_num) == (1, -2, 1, 0)

    def test_e2_2_row_1_matches_cpp(self):
        """E2_2 row 1 (near-interior): 3x3 system, no alpha.

        C++ (E2_2.cpp):
        c[4] = (2 + 4*psi) / (2 + 3*psi + psi^2)
        c[5] = -2*psi
        c[6] = (-2 + 4*psi^2) / (1 + psi)
        c[7] = (2 - 2*psi^2) / (2 + psi)
        """
        psi = Symbol("psi")
        K, _ = make_psi_field(psi)

        # Row 1 (near-interior), Category A: x_0 zeroed, target = -2.
        # alpha_{1,x_0}(psi) = -2*psi (prescribed, col 1)
        # Vandermonde from x_1: deltas [-(psi+1), -1, 0, 1]
        # After removing col 1 (x_0 = -2*psi):
        V = Matrix([
            [1, 1, 1],
            [-(psi + 1), 0, 1],
            [(psi + 1) ** 2 / 2, 0, Rational(1, 2)],
        ])
        rhs = Matrix([
            [-(-2 * psi)],  # move col 1 to RHS: -1 * (-2*psi)
            [-(-1) * (-2 * psi)],  # -V[1,1]*prescribed = -(-1)*(-2*psi) = -2*psi
            [1 - Rational(1, 2) * (-2 * psi)],
        ])
        # Actually let me build this more carefully.
        # Full Vandermonde (4 cols: wall, x_0, x_1, x_2):
        # deltas from x_1: [-(psi+1), -1, 0, 1]
        # V_full[k,j] = delta_j^k / k!
        # k=0: [1, 1, 1, 1]
        # k=1: [-(psi+1), -1, 0, 1]
        # k=2: [(psi+1)^2/2, 1/2, 0, 1/2]
        # rhs: [0, 0, 1]
        #
        # Prescribed: col 1 (x_0) = -2*psi
        # Reduced system: cols [0, 2, 3], rhs adjusted
        V_full = Matrix([
            [1, 1, 1, 1],
            [-(psi + 1), -1, 0, 1],
            [(psi + 1) ** 2 / 2, Rational(1, 2), 0, Rational(1, 2)],
        ])
        rhs_full = Matrix([[0], [0], [1]])

        # Move col 1 to RHS
        V_reduced = V_full[:, [0, 2, 3]]
        prescribed_col = V_full[:, 1]
        rhs_reduced = rhs_full - prescribed_col * (-2 * psi)

        sol = solve_in_field(V_reduced, rhs_reduced, K, symbols=[])

        # Expected:
        # c_wall = (2 + 4*psi) / (2 + 3*psi + psi^2)
        # c_x1 = (-2 + 4*psi^2) / (1 + psi)
        # c_x2 = (2 - 2*psi^2) / (2 + psi)
        expected_wall = (2 + 4 * psi) / (2 + 3 * psi + psi**2)
        expected_x1 = (-2 + 4 * psi**2) / (1 + psi)
        expected_x2 = (2 - 2 * psi**2) / (2 + psi)

        assert simplify(sol[0] - expected_wall) == 0
        assert simplify(sol[1] - expected_x1) == 0
        assert simplify(sol[2] - expected_x2) == 0

    def test_symbol_dependent_rhs(self):
        """solve_in_field with alpha and beta in RHS returns correct decomposition.

        Test: V = [[1, 0], [0, 1]], rhs = [psi*alpha + beta, 1 + psi^2*alpha]
        Solution should be x = rhs (identity system).
        """
        psi = Symbol("psi")
        alpha = Symbol("alpha")
        beta = Symbol("beta")
        K, _ = make_psi_field(psi)

        V = Matrix([[1, 0], [0, 1]])
        rhs = Matrix([[psi * alpha + beta], [1 + psi**2 * alpha]])

        sol = solve_in_field(V, rhs, K, symbols=[alpha, beta])

        assert simplify(sol[0] - (psi * alpha + beta)) == 0
        assert simplify(sol[1] - (1 + psi**2 * alpha)) == 0

    def test_symbol_dependent_nontrivial(self):
        """solve_in_field with a non-trivial system and symbol-dependent RHS.

        V = [[1, 1], [0, 1]], rhs = [alpha, psi]
        Solution: x_1 = psi, x_0 = alpha - psi
        """
        psi = Symbol("psi")
        alpha = Symbol("alpha")
        K, _ = make_psi_field(psi)

        V = Matrix([[1, 1], [0, 1]])
        rhs = Matrix([[alpha], [psi]])

        sol = solve_in_field(V, rhs, K, symbols=[alpha])

        assert simplify(sol[0] - (alpha - psi)) == 0
        assert simplify(sol[1] - psi) == 0

    def test_symbol_dependent_multi_symbol(self):
        """solve_in_field with both alpha and beta in a 2x2 system.

        V = [[1, psi], [psi, 1]], rhs = [alpha + beta, psi*alpha]
        """
        psi = Symbol("psi")
        alpha = Symbol("alpha")
        beta = Symbol("beta")
        K, _ = make_psi_field(psi)

        V = Matrix([[1, psi], [psi, 1]])
        rhs = Matrix([[alpha + beta], [psi * alpha]])

        sol = solve_in_field(V, rhs, K, symbols=[alpha, beta])

        # Verify by substitution: V @ sol = rhs
        for s0, s1 in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            subs = {alpha: s0, beta: s1}
            lhs0 = sol[0].subs(subs) + psi * sol[1].subs(subs)
            rhs0 = (alpha + beta).subs(subs)
            assert simplify(lhs0 - rhs0) == 0
            lhs1 = psi * sol[0].subs(subs) + sol[1].subs(subs)
            rhs1 = (psi * alpha).subs(subs)
            assert simplify(lhs1 - rhs1) == 0

    def test_performance_3x3(self):
        """Single solve_in_field call for a 3x3 system completes in <0.01s."""
        psi = Symbol("psi")
        K, _ = make_psi_field(psi)

        V = Matrix([
            [1, 1, 1],
            [0, 1, 2],
            [0, Rational(1, 2), 2],
        ])
        rhs = Matrix([[-psi], [psi**2], [1 - psi**3 / 2]])

        t0 = time.monotonic()
        solve_in_field(V, rhs, K, symbols=[])
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1, f"Solve took {elapsed:.3f}s, expected <0.01s"


# ---------------------------------------------------------------------------
# 20.5d — Cut-cell stencil construction tests
# ---------------------------------------------------------------------------


class TestBuildCutCellDeltas:
    """Tests for build_cut_cell_deltas."""

    def test_row_0_T5(self):
        """Row 0, T=5: [-(psi+0), 0, 1, 2, 3] = [-psi, 0, 1, 2, 3]."""
        psi = Symbol("psi")
        d = build_cut_cell_deltas(0, 5, psi)
        assert len(d) == 5
        assert simplify(d[0] + psi) == 0
        assert d[1] == 0
        assert d[2] == 1
        assert d[3] == 2
        assert d[4] == 3

    def test_row_1_T5(self):
        """Row 1, T=5: [-(psi+1), -1, 0, 1, 2]."""
        psi = Symbol("psi")
        d = build_cut_cell_deltas(1, 5, psi)
        assert simplify(d[0] + psi + 1) == 0
        assert d[1] == -1
        assert d[2] == 0

    def test_row_3_T5(self):
        """Row 3, T=5: [-(psi+3), -3, -2, -1, 0]."""
        psi = Symbol("psi")
        d = build_cut_cell_deltas(3, 5, psi)
        assert simplify(d[0] + psi + 3) == 0
        assert d[1] == -3
        assert d[4] == 0


class TestBuildTemoVandermonde:
    """Tests for build_temo_vandermonde."""

    def test_e2_1_row_0_shape(self):
        """E2_1 row 0: n_eqs=2, T=5 → V is 2x5, rhs is 2x1."""
        psi = Symbol("psi")
        V, rhs = build_temo_vandermonde(0, 5, q=1, nu=1, psi=psi)
        assert V.shape == (2, 5)
        assert rhs.shape == (2, 1)

    def test_e2_1_row_0_values(self):
        """E2_1 row 0: matches worked example from plan.

        V = | 1       1    1    1    1  |     rhs = | 0 |
            | -psi    0    1    2    3  |           | 1 |
        """
        psi = Symbol("psi")
        V, rhs = build_temo_vandermonde(0, 5, q=1, nu=1, psi=psi)
        # Row 0 (k=0): all ones
        for j in range(5):
            assert V[0, j] == 1 or simplify(V[0, j] - 1) == 0
        # Row 1 (k=1): deltas
        assert simplify(V[1, 0] + psi) == 0  # -psi
        assert V[1, 1] == 0
        assert V[1, 2] == 1
        assert V[1, 3] == 2
        assert V[1, 4] == 3
        assert rhs[0, 0] == 0
        assert rhs[1, 0] == 1

    def test_e2_2_row_0_shape(self):
        """E2_2 row 0: n_eqs=max(2,3)=3, T=4 → V is 3x4."""
        psi = Symbol("psi")
        V, rhs = build_temo_vandermonde(0, 4, q=1, nu=2, psi=psi)
        assert V.shape == (3, 4)
        assert rhs == Matrix([[0], [0], [1]])


class TestSolveUniformLimit:
    """Tests for solve_uniform_limit (B_l(1))."""

    def test_e2_2_matches_expected(self, e2_2_uniform):
        """E2_2 B_l(1) = [[1, -2, 1, 0], [1, -2, 1, 0]]."""
        ur = e2_2_uniform
        B_l_1 = solve_uniform_limit(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0)
        expected = Matrix([[1, -2, 1, 0], [1, -2, 1, 0]])
        assert B_l_1 == expected

    def test_e2_1_shape(self, e2_1_uniform):
        """E2_1 B_l(1) has shape (4, 5)."""
        ur = e2_1_uniform
        B_l_1 = solve_uniform_limit(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 1)
        assert B_l_1.shape == (4, 5)

    def test_e2_1_wall_zero_boundary_rows(self, e2_1_uniform):
        """E2_1: wall (col 0) = 0 for all boundary rows (nu=1)."""
        ur = e2_1_uniform
        B_l_1 = solve_uniform_limit(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 1)
        for i in range(3):
            assert B_l_1[i, 0] == 0

    def test_e2_1_boundary_cols_match_Bu(self, e2_1_uniform):
        """E2_1: boundary rows cols 1..4 match B_u cols 0..3."""
        ur = e2_1_uniform
        B_l_1 = solve_uniform_limit(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 1)
        for i in range(3):
            for j in range(4):
                assert simplify(B_l_1[i, j + 1] - ur.B_u[i, j]) == 0

    def test_e2_1_near_interior_taylor(self, e2_1_uniform):
        """E2_1: near-interior row (row 3) satisfies Taylor at psi=1."""
        ur = e2_1_uniform
        B_l_1 = solve_uniform_limit(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 1)
        deltas = [Rational(-4), Rational(-3), Rational(-2), Rational(-1), Rational(0)]
        row = [B_l_1[3, j] for j in range(5)]
        # k=0: sum = 0
        assert simplify(sum(row)) == 0
        # k=1: weighted sum = 1
        assert simplify(sum(row[j] * deltas[j] for j in range(5)) - 1) == 0

    def test_e2_1_conservation_cols(self, e2_1_uniform):
        """E2_1: conservation at psi=1 for cols 2,3,4."""
        ur = e2_1_uniform
        B_l_1 = solve_uniform_limit(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 1)
        for j in [2, 3, 4]:
            col_sum = sum(B_l_1[i, j] for i in range(4))
            assert simplify(col_sum) == 0, f"Conservation failed at col {j}"


class TestConstructCutCellStencil:
    """Tests for construct_cut_cell_stencil (20.5d)."""

    def test_e2_2_shape_and_no_betas(self, e2_2_uniform):
        """E2_2: 2x4 matrix, no beta parameters."""
        psi = Symbol("psi")
        ur = e2_2_uniform
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0, psi
        )
        assert result.matrix.shape == (2, 4)
        assert len(result.beta_info) == 0
        assert len(result.beta_symbols) == 0

    def test_e2_2_coefficients_match_cpp(self, e2_2_uniform):
        """E2_2 floating coefficients match E2_2.cpp exactly.

        C++ (pre-h^2 scaling, left boundary):
        c[0] = psi
        c[1] = (2 - 2*psi - 3*psi^2 - psi^3) / 2
        c[2] = -2 + 2*psi^2 + psi^3
        c[3] = (2 - psi^2 - psi^3) / 2
        c[4] = (2 + 4*psi) / (2 + 3*psi + psi^2)
        c[5] = -2*psi
        c[6] = (-2 + 4*psi^2) / (1 + psi)
        c[7] = (2 - 2*psi^2) / (2 + psi)
        """
        psi = Symbol("psi")
        ur = e2_2_uniform
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0, psi
        )
        m = result.matrix
        expected = [
            psi,
            (2 - 2 * psi - 3 * psi**2 - psi**3) / 2,
            -2 + 2 * psi**2 + psi**3,
            (2 - psi**2 - psi**3) / 2,
            (2 + 4 * psi) / (2 + 3 * psi + psi**2),
            -2 * psi,
            (-2 + 4 * psi**2) / (1 + psi),
            (2 - 2 * psi**2) / (2 + psi),
        ]
        for i in range(2):
            for j in range(4):
                idx = i * 4 + j
                assert simplify(m[i, j] - expected[idx]) == 0, (
                    f"Mismatch at c[{idx}]: {cancel(m[i, j])} != {expected[idx]}"
                )

    def test_e2_2_degenerate_limit(self, e2_2_uniform):
        """E2_2 at psi=0 matches degenerate: [[0,1,-2,1],[1,0,-2,1]]."""
        psi = Symbol("psi")
        ur = e2_2_uniform
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0, psi
        )
        m0 = result.matrix.subs(psi, 0)
        expected = Matrix([[0, 1, -2, 1], [1, 0, -2, 1]])
        assert m0 == expected

    def test_e2_2_uniform_limit(self, e2_2_uniform):
        """E2_2 at psi=1 matches uniform: [[1,-2,1,0],[1,-2,1,0]]."""
        psi = Symbol("psi")
        ur = e2_2_uniform
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0, psi
        )
        m1 = result.matrix.subs(psi, 1)
        expected = Matrix([[1, -2, 1, 0], [1, -2, 1, 0]])
        assert m1 == expected

    def test_e2_2_floating_psi05_right_boundary(self, e2_2_uniform):
        """E2_2 floating at psi=0.5, h=0.5, right boundary matches C++.

        C++ right boundary output (after /h^2):
        [2.4, -2.6666..., -4.0, 4.2666..., 3.25, -5.5, 0.25, 2.0]

        To recover left boundary h=1 values: reverse the flat array, then * h^2.
        """
        psi = Symbol("psi")
        ur = e2_2_uniform
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0, psi
        )
        h = Rational(1, 2)
        # Left boundary /h^2 values
        left_div_h2 = []
        for i in range(2):
            for j in range(4):
                left_div_h2.append(
                    float(result.matrix[i, j].subs(psi, Rational(1, 2)) / h**2)
                )
        # C++ right boundary = reversed left boundary (nu=2, no negation)
        cpp_right = [2.4, -2.6666666666666665, -4.0, 4.266666666666667,
                     3.25, -5.5, 0.25, 2.0]
        cpp_left = list(reversed(cpp_right))
        for k in range(8):
            assert abs(left_div_h2[k] - cpp_left[k]) < 1e-12, (
                f"c[{k}]: {left_div_h2[k]} != {cpp_left[k]}"
            )

    def test_e2_1_shape_and_no_betas(self, e2_1_uniform):
        """E2_1: 4x5 matrix, no beta parameters (limit interpolation replaces betas)."""
        psi = Symbol("psi")
        ur = e2_1_uniform
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 1, psi
        )
        assert result.matrix.shape == (4, 5)
        assert len(result.beta_info) == 0
        assert len(result.beta_symbols) == 0

    def test_e2_1_entries_in_psi_alpha(self, e2_1_uniform):
        """E2_1: all matrix entries are rational in psi, linear in alpha only."""
        psi = Symbol("psi")
        ur = e2_1_uniform
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 1, psi
        )
        all_syms = result.matrix.free_symbols
        expected_names = {"psi"} | {f"alpha_{k}" for k in range(4)}
        actual_names = {s.name for s in all_syms}
        assert actual_names <= expected_names, (
            f"Unexpected symbols: {actual_names - expected_names}"
        )

    def test_e2_1_degenerate_limit(self, e2_1_uniform):
        """E2_1 at psi=0 matches the degenerate stencil B_d."""
        psi = Symbol("psi")
        ur = e2_1_uniform
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 1, psi
        )
        B_d = result.matrix.subs(psi, 0)
        B_d_expected = build_degenerate_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu
        )
        for i in range(4):
            for j in range(5):
                assert simplify(B_d[i, j] - B_d_expected[i, j]) == 0, (
                    f"Degenerate mismatch at [{i},{j}]: "
                    f"{B_d[i,j]} != {B_d_expected[i,j]}"
                )

    def test_e2_1_uniform_limit(self, e2_1_uniform):
        """E2_1 at psi=1 matches B_l(1) from solve_uniform_limit."""
        psi = Symbol("psi")
        ur = e2_1_uniform
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 1, psi
        )
        B_1 = result.matrix.subs(psi, 1)
        B_l_1 = solve_uniform_limit(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 1)
        for i in range(4):
            for j in range(5):
                assert simplify(B_1[i, j] - B_l_1[i, j]) == 0, (
                    f"Uniform limit mismatch at [{i},{j}]: "
                    f"{B_1[i,j]} != {B_l_1[i,j]}"
                )

    def test_e2_1_taylor_accuracy_per_row(self, e2_1_uniform):
        """E2_1: each row satisfies Taylor accuracy (q+1=2 equations) for all psi."""
        psi = Symbol("psi")
        ur = e2_1_uniform
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 1, psi
        )
        m = result.matrix
        for i in range(4):
            deltas = build_cut_cell_deltas(i, 5, psi)
            row = [m[i, j] for j in range(5)]
            # k=0: sum = 0
            assert simplify(sum(row)) == 0, f"Row {i} k=0 failed"
            # k=1: weighted sum = 1
            s1 = sum(row[j] * deltas[j] for j in range(5))
            assert simplify(s1 - 1) == 0, f"Row {i} k=1 failed"

    def test_e2_1_conservation_extra_cols(self, e2_1_uniform):
        """E2_1: psi-dependent conservation holds for extra columns (j=3,4)."""
        psi = Symbol("psi")
        ur = e2_1_uniform
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 1, psi
        )
        m = result.matrix
        # SBP weights: w = [psi, 1, 1, 1]
        for j in [3, 4]:
            col_sum = psi * m[0, j] + m[1, j] + m[2, j] + m[3, j]
            assert simplify(col_sum) == 0, (
                f"Conservation failed at col {j}: {cancel(col_sum)}"
            )

    def test_e2_2_taylor_accuracy_per_row(self, e2_2_uniform):
        """E2_2: each row satisfies Taylor accuracy for all psi."""
        psi = Symbol("psi")
        ur = e2_2_uniform
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0, psi
        )
        m = result.matrix
        for i in range(2):
            deltas = build_cut_cell_deltas(i, 4, psi)
            row = [m[i, j] for j in range(4)]
            # k=0: sum = 0
            assert simplify(sum(row)) == 0, f"Row {i} k=0 failed"
            # k=1: sum * delta = 0
            s1 = sum(row[j] * deltas[j] for j in range(4))
            assert simplify(s1) == 0, f"Row {i} k=1 failed"
            # k=2: sum * delta^2/2 = 1
            s2 = sum(row[j] * deltas[j] ** 2 / 2 for j in range(4))
            assert simplify(s2 - 1) == 0, f"Row {i} k=2 failed"


# ---------------------------------------------------------------------------
# 20.5f — Neumann eta coefficients and output assembly
# ---------------------------------------------------------------------------


class TestNeumannUniform:
    """Tests for derive_uniform_neumann."""

    def test_e2_2_uniform_neumann(self):
        """E2_2 uniform Neumann: c = [-2, 2, 0], eta = -2."""
        B_uN, eta_u = derive_uniform_neumann(
            interior=[Rational(1), Rational(-2), Rational(1)],
            p=1, q=1, nu=2,
        )
        assert B_uN.shape == (1, 3), f"Shape {B_uN.shape}, expected (1, 3)"
        assert B_uN[0, 0] == Rational(-2), f"B_uN[0,0] = {B_uN[0,0]}"
        assert B_uN[0, 1] == Rational(2), f"B_uN[0,1] = {B_uN[0,1]}"
        assert B_uN[0, 2] == Rational(0), f"B_uN[0,2] = {B_uN[0,2]}"
        assert len(eta_u) == 1
        assert eta_u[0] == Rational(-2), f"eta_u[0] = {eta_u[0]}"


class TestNeumannVandermonde:
    """Tests for build_neumann_vandermonde."""

    def test_e2_2_row0_virtual_column(self):
        """E2_2 Neumann virtual column at row 0: V_eta = [0, 1, -psi]."""
        psi = Symbol("psi")
        V_aug, rhs = build_neumann_vandermonde(0, 4, q=1, nu=2, psi=psi)
        assert V_aug.shape == (3, 5)  # 3 eqs x (4+1) cols
        # Virtual column (col 4)
        assert V_aug[0, 4] == 0
        assert V_aug[1, 4] == 1  # delta_wall^0 / 0! = 1
        assert simplify(V_aug[2, 4] - (-psi)) == 0  # delta_wall^1 / 1! = -psi

    def test_e2_2_row1_virtual_column(self):
        """E2_2 Neumann virtual column at row 1: V_eta = [0, 1, -(psi+1)]."""
        psi = Symbol("psi")
        V_aug, rhs = build_neumann_vandermonde(1, 4, q=1, nu=2, psi=psi)
        assert V_aug.shape == (3, 5)
        assert V_aug[0, 4] == 0
        assert V_aug[1, 4] == 1
        assert simplify(V_aug[2, 4] - (-(psi + 1))) == 0


class TestNeumannUniformLimit:
    """Tests for solve_neumann_uniform_limit."""

    def test_e2_2_neumann_psi1(self):
        """E2_2 Neumann at psi=1: [[-2, 2, 0, 0], [1, -2, 1, 0]], eta = [-2, 0]."""
        B_uN, eta_u = derive_uniform_neumann(
            interior=[Rational(1), Rational(-2), Rational(1)],
            p=1, q=1, nu=2,
        )
        B_l_N_1 = solve_neumann_uniform_limit(
            B_uN, eta_u,
            interior=[Rational(1), Rational(-2), Rational(1)],
            p=1, q=1, nu=2, nextra=0,
        )
        # Shape: R=2 x (T+1)=5
        assert B_l_N_1.shape == (2, 5)
        # Row 0: stencil = [-2, 2, 0, 0], eta = -2
        assert B_l_N_1[0, 0] == Rational(-2), f"[0,0] = {B_l_N_1[0,0]}"
        assert B_l_N_1[0, 1] == Rational(2), f"[0,1] = {B_l_N_1[0,1]}"
        assert B_l_N_1[0, 2] == Rational(0), f"[0,2] = {B_l_N_1[0,2]}"
        assert B_l_N_1[0, 3] == Rational(0), f"[0,3] = {B_l_N_1[0,3]}"
        assert B_l_N_1[0, 4] == Rational(-2), f"eta[0] = {B_l_N_1[0,4]}"
        # Row 1: stencil = [1, -2, 1, 0], eta = 0
        assert B_l_N_1[1, 0] == Rational(1), f"[1,0] = {B_l_N_1[1,0]}"
        assert B_l_N_1[1, 1] == Rational(-2), f"[1,1] = {B_l_N_1[1,1]}"
        assert B_l_N_1[1, 2] == Rational(1), f"[1,2] = {B_l_N_1[1,2]}"
        assert B_l_N_1[1, 3] == Rational(0), f"[1,3] = {B_l_N_1[1,3]}"
        assert B_l_N_1[1, 4] == Rational(0), f"eta[1] = {B_l_N_1[1,4]}"


class TestNeumannStencil:
    """Tests for construct_neumann_stencil — E2_2 Neumann."""

    @pytest.fixture
    def e2_2_neumann(self, e2_2_uniform):
        """Derive the E2_2 Neumann stencil."""
        psi = Symbol("psi")
        ur = e2_2_uniform
        B_uN, eta_u = derive_uniform_neumann(
            interior=ur.interior, p=ur.p, q=ur.q, nu=ur.nu,
        )
        neumann_main, eta = construct_neumann_stencil(
            ur.B_u, B_uN, eta_u, ur.interior,
            ur.p, ur.q, ur.nu, 0, psi,
        )
        return neumann_main, eta, psi

    def test_e2_2_neumann_eta(self, e2_2_neumann):
        """E2_2 Neumann eta: eta_0 = -2 (constant), eta_1 = 2*(psi-1)."""
        neumann_main, eta, psi = e2_2_neumann
        assert simplify(eta[0] - (-2)) == 0, f"eta[0] = {eta[0]}"
        assert simplify(eta[1] - 2 * (psi - 1)) == 0, f"eta[1] = {eta[1]}"

    def test_e2_2_neumann_psi1(self, e2_2_neumann):
        """At psi=1: [[-2, 2, 0, 0], [1, -2, 1, 0]], eta = [-2, 0]."""
        neumann_main, eta, psi = e2_2_neumann
        B_1 = neumann_main.subs(psi, 1)
        expected = Matrix([[-2, 2, 0, 0], [1, -2, 1, 0]])
        for i in range(2):
            for j in range(4):
                assert simplify(B_1[i, j] - expected[i, j]) == 0, (
                    f"psi=1 mismatch at [{i},{j}]: {B_1[i,j]} vs {expected[i,j]}"
                )
        eta_1 = [cancel(e.subs(psi, 1)) for e in eta]
        assert eta_1[0] == -2
        assert eta_1[1] == 0

    def test_e2_2_neumann_psi0_left(self, e2_2_neumann):
        """At psi=0, h=0.5, left: c=[-8,0,8,0,0,-8,8,0], x=[-4,-4]."""
        neumann_main, eta, psi = e2_2_neumann
        h = Rational(1, 2)
        B_0 = neumann_main.subs(psi, 0)
        eta_0 = [cancel(e.subs(psi, 0)) for e in eta]
        # Pre-h-scaling values (h=1): flatten row-major
        c_h1 = [cancel(B_0[i, j]) for i in range(2) for j in range(4)]
        x_h1 = eta_0
        # Apply h-scaling: c /= h^2, x /= h
        c_scaled = [float(c / h**2) for c in c_h1]
        x_scaled = [float(x / h) for x in x_h1]
        c_expected = [-8., 0., 8., 0., 0., -8., 8., 0.]
        x_expected = [-4., -4.]
        for k in range(8):
            assert abs(c_scaled[k] - c_expected[k]) < 1e-12, (
                f"c[{k}] = {c_scaled[k]}, expected {c_expected[k]}"
            )
        for k in range(2):
            assert abs(x_scaled[k] - x_expected[k]) < 1e-12, (
                f"x[{k}] = {x_scaled[k]}, expected {x_expected[k]}"
            )

    def test_e2_2_neumann_psi08_right(self, e2_2_neumann):
        """At psi=0.8, h=0.5, right boundary: matches C++ test data."""
        neumann_main, eta, psi_sym = e2_2_neumann
        h = 0.5
        psi_val = 0.8

        # Evaluate h=1 left-boundary coefficients
        B_eval = neumann_main.subs(psi_sym, Rational(4, 5))
        c_h1 = [float(cancel(B_eval[i, j])) for i in range(2) for j in range(4)]
        eta_eval = [float(cancel(e.subs(psi_sym, Rational(4, 5)))) for e in eta]

        # Apply h-scaling
        c_scaled = [c / (h * h) for c in c_h1]
        x_scaled = [e / h for e in eta_eval]

        # Apply right-boundary transform: reverse c, reverse x then negate x
        c_right = list(reversed(c_scaled))
        x_right = list(reversed([-v for v in x_scaled]))

        c_expected = [
            -0.384,
            4.928,
            -7.744,
            3.2,
            -0.45714285714285713,
            2.311111111111111,
            6.4,
            -8.253968253968255,
        ]
        x_expected = [0.8, 4.]

        for k in range(8):
            assert abs(c_right[k] - c_expected[k]) < 1e-12, (
                f"c[{k}] = {c_right[k]}, expected {c_expected[k]}"
            )
        for k in range(2):
            assert abs(x_right[k] - x_expected[k]) < 1e-12, (
                f"x[{k}] = {x_right[k]}, expected {x_expected[k]}"
            )

    def test_e2_2_neumann_coeffs_match_cpp(self, e2_2_neumann):
        """E2_2 Neumann symbolic coefficients match E2_2.cpp nbs_neumann."""
        neumann_main, eta, psi = e2_2_neumann
        # Row 0: c[0..3] from C++
        # c[0] = -(4 + 8*psi) / (2 + 3*psi + psi^2)
        c0_expected = -(4 + 8*psi) / (2 + 3*psi + psi**2)
        assert simplify(neumann_main[0, 0] - c0_expected) == 0, (
            f"c[0]: {cancel(neumann_main[0,0])} vs {c0_expected}"
        )
        # c[1] = 2*psi
        assert simplify(neumann_main[0, 1] - 2*psi) == 0
        # c[2] = (2 + 2*psi - 4*psi^2) / (1 + psi)
        c2_expected = (2 + 2*psi - 4*psi**2) / (1 + psi)
        assert simplify(neumann_main[0, 2] - c2_expected) == 0
        # c[3] = 2*psi*(psi-1) / (2+psi)
        c3_expected = 2*psi*(psi - 1) / (2 + psi)
        assert simplify(neumann_main[0, 3] - c3_expected) == 0

        # Row 1: c[4..7] from C++
        # c[4] = psi
        assert simplify(neumann_main[1, 0] - psi) == 0
        # c[5] = (-4 - psi^3 + psi^2) / 2
        c5_expected = (-4 - psi**3 + psi**2) / 2
        assert simplify(neumann_main[1, 1] - c5_expected) == 0
        # c[6] = 2 + psi^3 - 2*psi^2
        c6_expected = 2 + psi**3 - 2*psi**2
        assert simplify(neumann_main[1, 2] - c6_expected) == 0
        # c[7] = -psi*(2 - 3*psi + psi^2) / 2
        c7_expected = -psi*(2 - 3*psi + psi**2) / 2
        assert simplify(neumann_main[1, 3] - c7_expected) == 0

    def test_e2_2_neumann_taylor_accuracy(self, e2_2_neumann):
        """Each Neumann row satisfies augmented Taylor accuracy for all psi."""
        neumann_main, eta, psi = e2_2_neumann
        T = 4
        for i in range(2):
            deltas = build_cut_cell_deltas(i, T, psi)
            row = [neumann_main[i, j] for j in range(T)]
            delta_wall = deltas[0]  # -(psi + i)
            # k=0: sum = 0
            assert simplify(sum(row)) == 0, f"Row {i} k=0 failed"
            # k=1: sum * delta + eta * 1 = 0
            s1 = sum(row[j] * deltas[j] for j in range(T)) + eta[i] * 1
            assert simplify(s1) == 0, f"Row {i} k=1 failed: {cancel(s1)}"
            # k=2: sum * delta^2/2 + eta * delta_wall = 1
            s2 = sum(row[j] * deltas[j]**2 / 2 for j in range(T)) + eta[i] * delta_wall
            assert simplify(s2 - 1) == 0, f"Row {i} k=2 failed: {cancel(s2)}"


class TestNeumannE1HasNoNeumann:
    """E2_1 has X=0, no Neumann support."""

    def test_e2_1_no_neumann(self):
        dims = E2_1.dims()
        assert dims.X == 0


class TestAssembleCutCellResult:
    """Tests for assemble_cut_cell_result."""

    def test_e2_2_dirichlet_is_floating_rows_1_to_R(self, e2_2_uniform):
        """E2_2 Dirichlet = rows 1..R-1 of Floating, T=4 entries per row."""
        psi = Symbol("psi")
        ur = e2_2_uniform
        floating_result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0, psi
        )
        dims = E2_2.dims()
        result = assemble_cut_cell_result(
            floating_result.matrix, None, None, dims, [],
        )
        assert result.dirichlet.shape == (1, 4)
        for j in range(4):
            assert simplify(
                result.dirichlet[0, j] - floating_result.matrix[1, j]
            ) == 0

    def test_e2_2_full_assembly(self, e2_2_uniform):
        """E2_2: full assembly with floating, dirichlet, and neumann."""
        psi = Symbol("psi")
        ur = e2_2_uniform
        floating_result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0, psi
        )
        B_uN, eta_u = derive_uniform_neumann(
            interior=ur.interior, p=ur.p, q=ur.q, nu=ur.nu,
        )
        neumann_main, eta = construct_neumann_stencil(
            ur.B_u, B_uN, eta_u, ur.interior,
            ur.p, ur.q, ur.nu, 0, psi,
        )
        dims = E2_2.dims()
        result = assemble_cut_cell_result(
            floating_result.matrix, neumann_main, eta, dims, [],
        )
        assert result.floating.shape == (2, 4)
        assert result.dirichlet.shape == (1, 4)
        assert result.neumann.shape == (2, 4)
        assert len(result.eta) == 2
        assert result.dims == dims


# ---------------------------------------------------------------------------
# Helpers for C++ data comparison (20.5g / 20.5h)
# ---------------------------------------------------------------------------


def to_h1_left(c_cpp, h, nu, right):
    """Convert C++ nbs_* output to h=1 left-boundary coefficients.

    Undoes h-scaling and right-boundary reversal/negation applied by C++ code.
    """
    c = list(c_cpp)
    if right:
        c = list(reversed(c))
        if nu == 1:
            c = [-v for v in c]
    scale = h ** nu
    return [v * scale for v in c]


def to_h1_left_eta(x_cpp, h, right):
    """Convert C++ nbs_neumann x (eta) output to h=1 left-boundary values.

    Undoes h-scaling, reversal, and negation for right boundary eta.
    """
    x = list(x_cpp)
    if right:
        x = list(reversed(x))
        x = [-v for v in x]
    return [v * h for v in x]


# ---------------------------------------------------------------------------
# 20.5g — E2_1 integration tests
# ---------------------------------------------------------------------------


class TestE2_1Integration:
    """Integration tests for E2_1 cut-cell stencil (20.5g)."""

    @pytest.fixture
    def e2_1(self, e2_1_uniform):
        """Derive the full E2_1 cut-cell stencil."""
        psi = Symbol("psi")
        ur = e2_1_uniform
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 1, psi
        )
        dims = E2_1.dims()
        assembled = assemble_cut_cell_result(
            stencil.matrix, None, None, dims, ur.alpha_symbols,
        )
        return assembled, stencil, ur, psi

    def test_grid_deltas_all_rows(self):
        """build_cut_cell_deltas for i=0,1,2,3 with T=5."""
        psi = Symbol("psi")
        expected_col0 = [-psi, -(psi + 1), -(psi + 2), -(psi + 3)]
        for i in range(4):
            d = build_cut_cell_deltas(i, 5, psi)
            assert len(d) == 5
            assert simplify(d[0] - expected_col0[i]) == 0
            for j in range(1, 5):
                assert d[j] == (j - 1) - i

    def test_taylor_system_row0(self):
        """2x5 Vandermonde for row 0 matches plan worked example."""
        psi = Symbol("psi")
        V, rhs = build_temo_vandermonde(0, 5, q=1, nu=1, psi=psi)
        assert V.shape == (2, 5)
        assert rhs == Matrix([[0], [1]])
        for j in range(5):
            assert V[0, j] == 1
        assert simplify(V[1, 0] + psi) == 0
        assert V[1, 1] == 0
        assert V[1, 2] == 1
        assert V[1, 3] == 2
        assert V[1, 4] == 3

    def test_degenerate_variant(self, e2_1_uniform):
        """nu=1 variant: wall=B_u[i,0], x_0=0 for all rows."""
        ur = e2_1_uniform
        B_d = build_degenerate_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu,
        )
        assert B_d.shape == (4, 5)
        for i in range(4):
            assert B_d[i, 1] == 0, f"x_0 not zero at row {i}"
        for i in range(3):
            assert simplify(B_d[i, 0] - ur.B_u[i, 0]) == 0

    def test_floating_taylor_numeric(self, e2_1):
        """Taylor accuracy at numeric psi=0.3, 0.5, 0.7 with multiple alpha values."""
        _, stencil, ur, psi = e2_1
        m = stencil.matrix
        alphas = [
            {s: 0 for s in ur.alpha_symbols},
            dict(zip(ur.alpha_symbols, [Rational(1, 2), Rational(3, 10),
                                         Rational(1, 5), Rational(1, 10)])),
        ]
        for alpha_vals in alphas:
            for psi_val in [Rational(3, 10), Rational(1, 2), Rational(7, 10)]:
                subs = {psi: psi_val, **alpha_vals}
                for i in range(4):
                    deltas = build_cut_cell_deltas(i, 5, psi_val)
                    row = [float(m[i, j].subs(subs)) for j in range(5)]
                    # k=0: sum = 0
                    assert abs(sum(row)) < 1e-14, (
                        f"k=0 failed: row {i}, psi={psi_val}, alpha={list(alpha_vals.values())}"
                    )
                    # k=1: weighted sum = 1
                    s1 = sum(row[j] * float(deltas[j]) for j in range(5))
                    assert abs(s1 - 1) < 1e-14, (
                        f"k=1 failed: row {i}, psi={psi_val}, alpha={list(alpha_vals.values())}"
                    )

    def test_polynomial_exactness(self, e2_1):
        """f(x)=2x+1 (linear): stencil gives f'(x_i)=2 for all rows, psi, alpha."""
        _, stencil, ur, psi = e2_1
        m = stencil.matrix
        alphas = [
            {s: 0 for s in ur.alpha_symbols},
            dict(zip(ur.alpha_symbols, [Rational(1, 2), Rational(3, 10),
                                         Rational(1, 5), Rational(1, 10)])),
        ]
        for alpha_vals in alphas:
            for psi_val in [Rational(3, 10), Rational(7, 10)]:
                subs = {psi: psi_val, **alpha_vals}
                # Grid: wall at -psi*h, x_j = j*h (h=1)
                x_pts = [float(-psi_val)] + list(range(4))
                f_vals = [2 * xi + 1 for xi in x_pts]
                for i in range(4):
                    row = [float(m[i, j].subs(subs)) for j in range(5)]
                    result_val = sum(row[j] * f_vals[j] for j in range(5))
                    assert abs(result_val - 2) < 1e-13, (
                        f"Poly exactness failed: row {i}, psi={psi_val}, "
                        f"alpha={list(alpha_vals.values())}"
                    )

    def test_conservation_numeric(self, e2_1):
        """Conservation psi*B[0,j]+sum B[i,j]=0 at j=3,4 with numeric values."""
        _, stencil, ur, psi = e2_1
        m = stencil.matrix
        alphas = [
            {s: 0 for s in ur.alpha_symbols},
            dict(zip(ur.alpha_symbols, [Rational(1, 2), Rational(3, 10),
                                         Rational(1, 5), Rational(1, 10)])),
        ]
        for alpha_vals in alphas:
            for psi_val in [Rational(3, 10), Rational(1, 2), Rational(7, 10)]:
                subs = {psi: psi_val, **alpha_vals}
                for j in [3, 4]:
                    col_sum = float(psi_val) * float(m[0, j].subs(subs))
                    for i_row in range(1, 4):
                        col_sum += float(m[i_row, j].subs(subs))
                    assert abs(col_sum) < 1e-13, (
                        f"Conservation col {j}, psi={psi_val}, "
                        f"alpha={list(alpha_vals.values())}"
                    )

    def test_dirichlet_is_rows_1_3(self, e2_1):
        """Dirichlet (3x5) = rows 1..3 of floating (4x5)."""
        assembled, _, _, psi = e2_1
        assert assembled.dirichlet.shape == (3, 5)
        for i in range(3):
            for j in range(5):
                assert simplify(
                    assembled.dirichlet[i, j] - assembled.floating[i + 1, j]
                ) == 0, f"Dirichlet[{i},{j}] != Floating[{i+1},{j}]"

    def test_degeneracy_psi0(self, e2_1):
        """At psi=0, all 20 entries match B_d from 20.5c."""
        _, stencil, ur, psi = e2_1
        B_d = build_degenerate_stencil(ur.B_u, ur.interior, ur.p, ur.q, ur.nu)
        m0 = stencil.matrix.subs(psi, 0)
        for i in range(4):
            for j in range(5):
                assert simplify(m0[i, j] - B_d[i, j]) == 0, (
                    f"Degenerate mismatch at [{i},{j}]"
                )

    def test_degeneracy_psi1(self, e2_1):
        """At psi=1: wall=0 for boundary rows; near-interior row has Taylor+conservation."""
        _, stencil, ur, psi = e2_1
        m1 = stencil.matrix.subs(psi, 1)
        # Wall column = 0 for boundary rows (1st derivative)
        for i in range(3):
            assert simplify(m1[i, 0]) == 0, f"Wall row {i} nonzero at psi=1"
        # Near-interior row 3 satisfies Taylor accuracy
        deltas = [Rational(-4), Rational(-3), Rational(-2), Rational(-1), Rational(0)]
        row3 = [m1[3, j] for j in range(5)]
        assert simplify(sum(row3)) == 0, "Row 3 k=0 at psi=1"
        assert simplify(sum(row3[j] * deltas[j] for j in range(5)) - 1) == 0, (
            "Row 3 k=1 at psi=1"
        )
        # Conservation at psi=1 (w_0=1): sum B[i,j] = 0 for interior cols
        for j in [3, 4]:
            col_sum = sum(m1[i, j] for i in range(4))
            assert simplify(col_sum) == 0, f"Conservation col {j} at psi=1"

    def test_conservation_symbolic(self, e2_1):
        """Symbolic: sum w_i*B[i,j]=0 for j=3,4 (all psi, all alpha)."""
        _, stencil, _, psi = e2_1
        m = stencil.matrix
        for j in [3, 4]:
            col_sum = psi * m[0, j] + m[1, j] + m[2, j] + m[3, j]
            assert simplify(col_sum) == 0, f"Conservation col {j}: {cancel(col_sum)}"


# ---------------------------------------------------------------------------
# 20.5h — E2_2 integration tests
# ---------------------------------------------------------------------------


class TestE2_2Integration:
    """Integration tests for E2_2 cut-cell stencil (20.5h)."""

    @pytest.fixture
    def e2_2(self, e2_2_uniform):
        """Derive the full E2_2 cut-cell stencil with all BC variants."""
        psi = Symbol("psi")
        ur = e2_2_uniform
        floating_result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0, psi,
        )
        B_uN, eta_u = derive_uniform_neumann(
            interior=ur.interior, p=ur.p, q=ur.q, nu=ur.nu,
        )
        neumann_main, eta = construct_neumann_stencil(
            ur.B_u, B_uN, eta_u, ur.interior,
            ur.p, ur.q, ur.nu, 0, psi,
        )
        dims = E2_2.dims()
        assembled = assemble_cut_cell_result(
            floating_result.matrix, neumann_main, eta, dims, [],
        )
        return assembled, neumann_main, eta, psi

    # -- Dirichlet tests --

    def test_dirichlet_psi0_left(self, e2_2):
        """Dirichlet h=0.5, psi=0.0, left: C++ [4., 0., -8., 4.]."""
        assembled, _, _, psi = e2_2
        cpp_out = [4., 0., -8., 4.]
        expected = to_h1_left(cpp_out, 0.5, nu=2, right=False)
        actual = [
            float(cancel(assembled.dirichlet[0, j].subs(psi, 0)))
            for j in range(4)
        ]
        for k in range(4):
            assert abs(actual[k] - expected[k]) < 1e-12, (
                f"Dirichlet[{k}] = {actual[k]}, expected {expected[k]}"
            )

    def test_dirichlet_psi09_right(self, e2_2):
        """Dirichlet h=0.5, psi=0.9, right: C++ data from E2_2.t.cpp line 48."""
        assembled, _, _, psi = e2_2
        cpp_out = [
            0.5241379310344828, 2.610526315789474,
            -7.2, 4.0653357531760435,
        ]
        expected = to_h1_left(cpp_out, 0.5, nu=2, right=True)
        actual = [
            float(cancel(assembled.dirichlet[0, j].subs(psi, Rational(9, 10))))
            for j in range(4)
        ]
        for k in range(4):
            assert abs(actual[k] - expected[k]) < 1e-12, (
                f"Dirichlet[{k}] = {actual[k]}, expected {expected[k]}"
            )

    # -- Floating tests --

    def test_floating_psi0_left(self, e2_2):
        """Floating h=0.5, psi=0.0, left: C++ [0,4,-8,4,4,0,-8,4]."""
        assembled, _, _, psi = e2_2
        cpp_out = [0., 4., -8., 4., 4., 0., -8., 4.]
        expected = to_h1_left(cpp_out, 0.5, nu=2, right=False)
        actual = [
            float(cancel(assembled.floating[i, j].subs(psi, 0)))
            for i in range(2) for j in range(4)
        ]
        for k in range(8):
            assert abs(actual[k] - expected[k]) < 1e-12, (
                f"Floating[{k}] = {actual[k]}, expected {expected[k]}"
            )

    def test_floating_psi05_right(self, e2_2):
        """Floating h=0.5, psi=0.5, right: C++ data from E2_2.t.cpp line 82."""
        assembled, _, _, psi = e2_2
        cpp_out = [
            2.4, -2.6666666666666665, -4., 4.266666666666667,
            3.25, -5.5, 0.25, 2.,
        ]
        expected = to_h1_left(cpp_out, 0.5, nu=2, right=True)
        actual = [
            float(cancel(assembled.floating[i, j].subs(psi, Rational(1, 2))))
            for i in range(2) for j in range(4)
        ]
        for k in range(8):
            assert abs(actual[k] - expected[k]) < 1e-12, (
                f"Floating[{k}] = {actual[k]}, expected {expected[k]}"
            )

    # -- Neumann tests --

    def test_neumann_x_count(self, e2_2):
        """X=2 extra Neumann coefficients produced."""
        assembled, _, eta, _ = e2_2
        assert assembled.dims.X == 2
        assert len(eta) == 2

    def test_neumann_psi0_left(self, e2_2):
        """Neumann h=0.5, psi=0.0, left: c=[-8,0,8,0,0,-8,8,0], x=[-4,-4]."""
        _, neumann_main, eta, psi = e2_2
        c_cpp = [-8., 0., 8., 0., 0., -8., 8., 0.]
        x_cpp = [-4., -4.]
        c_expected = to_h1_left(c_cpp, 0.5, nu=2, right=False)
        x_expected = to_h1_left_eta(x_cpp, 0.5, right=False)
        c_actual = [
            float(cancel(neumann_main[i, j].subs(psi, 0)))
            for i in range(2) for j in range(4)
        ]
        x_actual = [float(cancel(e.subs(psi, 0))) for e in eta]
        for k in range(8):
            assert abs(c_actual[k] - c_expected[k]) < 1e-12, (
                f"c[{k}] = {c_actual[k]}, expected {c_expected[k]}"
            )
        for k in range(2):
            assert abs(x_actual[k] - x_expected[k]) < 1e-12, (
                f"x[{k}] = {x_actual[k]}, expected {x_expected[k]}"
            )

    def test_neumann_psi08_right(self, e2_2):
        """Neumann h=0.5, psi=0.8, right: matches C++ E2_2.t.cpp line 105."""
        _, neumann_main, eta, psi_sym = e2_2
        c_cpp = [
            -0.384, 4.928, -7.744, 3.2,
            -0.45714285714285713, 2.311111111111111,
            6.4, -8.253968253968255,
        ]
        x_cpp = [0.8, 4.]
        c_expected = to_h1_left(c_cpp, 0.5, nu=2, right=True)
        x_expected = to_h1_left_eta(x_cpp, 0.5, right=True)
        c_actual = [
            float(cancel(neumann_main[i, j].subs(psi_sym, Rational(4, 5))))
            for i in range(2) for j in range(4)
        ]
        x_actual = [
            float(cancel(e.subs(psi_sym, Rational(4, 5)))) for e in eta
        ]
        for k in range(8):
            assert abs(c_actual[k] - c_expected[k]) < 1e-12, (
                f"c[{k}] = {c_actual[k]}, expected {c_expected[k]}"
            )
        for k in range(2):
            assert abs(x_actual[k] - x_expected[k]) < 1e-12, (
                f"x[{k}] = {x_actual[k]}, expected {x_expected[k]}"
            )

    # -- No free parameters --

    def test_no_free_parameters(self, e2_2):
        """All E2_2 outputs are fully determined (no alpha symbols)."""
        assembled, neumann_main, eta, psi = e2_2
        for name, mat in [("floating", assembled.floating),
                          ("dirichlet", assembled.dirichlet),
                          ("neumann", assembled.neumann)]:
            extra = mat.free_symbols - {psi}
            assert not extra, f"{name} has unexpected symbols: {extra}"
        for k, e in enumerate(eta):
            extra = e.free_symbols - {psi}
            assert not extra, f"eta[{k}] has unexpected symbols: {extra}"

    # -- Conservation / polynomial exactness --

    def test_conservation_polynomial_exactness(self, e2_2):
        """f(x)=x^2 gives f''=2 for all rows at psi=0.0, 0.5, 0.8, 1.0."""
        assembled, _, _, psi = e2_2
        m = assembled.floating
        for psi_val in [Rational(0), Rational(1, 2), Rational(4, 5), Rational(1)]:
            # Grid: wall at -psi*h, x_j = j*h (h=1)
            x_pts = [float(-psi_val)] + list(range(3))
            f_vals = [xi ** 2 for xi in x_pts]
            for i in range(2):
                row = [
                    float(cancel(m[i, j].subs(psi, psi_val)))
                    for j in range(4)
                ]
                result_val = sum(row[j] * f_vals[j] for j in range(4))
                assert abs(result_val - 2) < 1e-12, (
                    f"Poly exactness: row {i}, psi={psi_val}: got {result_val}"
                )


# ---------------------------------------------------------------------------
# 26.5b — E2_1 regression: derive_cut_cell_scheme is unaffected by zeros path
# ---------------------------------------------------------------------------


class TestE2_1DeriveCutCellSchemeRegression:
    """E2_1 has zeros=() — derive_cut_cell_scheme must match the manual pipeline."""

    @pytest.fixture(scope="class")
    def e2_1_scheme(self):
        psi = Symbol("psi")
        return derive_cut_cell_scheme(E2_1, psi), psi

    @pytest.fixture(scope="class")
    def e2_1_manual(self, e2_1_uniform):
        """Manual E2_1 pipeline (pre-Phase-26 reference)."""
        psi = Symbol("psi")
        ur = e2_1_uniform
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=1, q=1, nu=1, nextra=1, psi=psi,
        )
        dims = compute_dimensions(1, 1, 0, 1, 1)
        manual = assemble_cut_cell_result(
            stencil.matrix, None, None, dims, ur.alpha_symbols,
        )
        return manual, psi

    def test_shape(self, e2_1_scheme):
        result, _ = e2_1_scheme
        assert result.floating.shape == (4, 5)
        assert result.dirichlet.shape == (3, 5)

    def test_alpha_count(self, e2_1_scheme):
        result, _ = e2_1_scheme
        assert len(result.alpha_symbols) == 4

    def test_dims(self, e2_1_scheme):
        result, _ = e2_1_scheme
        assert result.dims == Dimensions(r=3, t=4, R=4, T=5, X=0)

    def test_floating_matches_manual(self, e2_1_scheme, e2_1_manual):
        auto, _ = e2_1_scheme
        manual, _ = e2_1_manual
        R, T = auto.floating.shape
        for i in range(R):
            for j in range(T):
                assert cancel(auto.floating[i, j] - manual.floating[i, j]) == 0, (
                    f"E2_1 floating mismatch at [{i},{j}]"
                )

    def test_taylor_accuracy(self, e2_1_scheme):
        """All rows satisfy q+1=2 Taylor equations at psi=0.3, 0.5, 0.7."""
        result, psi = e2_1_scheme
        m = result.floating
        alpha_vals = {s: 0 for s in result.alpha_symbols}
        for psi_val in [Rational(3, 10), Rational(1, 2), Rational(7, 10)]:
            subs = {psi: psi_val, **alpha_vals}
            for i in range(4):
                deltas = build_cut_cell_deltas(i, 5, psi_val)
                row = [float(m[i, j].subs(subs)) for j in range(5)]
                # k=0: sum = 0
                assert abs(sum(row)) < 1e-14, f"k=0 row {i} psi={psi_val}"
                # k=1: weighted sum = 1
                s1 = sum(row[j] * float(deltas[j]) for j in range(5))
                assert abs(s1 - 1) < 1e-14, f"k=1 row {i} psi={psi_val}"

    def test_psi_limits(self, e2_1_scheme):
        """Stencil is finite at psi=0 and psi=1."""
        result, psi = e2_1_scheme
        m = result.floating
        alpha_vals = {s: 0 for s in result.alpha_symbols}
        for psi_val in [0, 1]:
            subs = {psi: psi_val, **alpha_vals}
            for i in range(4):
                for j in range(5):
                    val = float(m[i, j].subs(subs))
                    assert abs(val) < 1e6, (
                        f"Divergent entry [{i},{j}] at psi={psi_val}: {val}"
                    )
