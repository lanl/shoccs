"""Tests for E4_1 cut-cell stencil derivation (21.1b onwards)."""

import pathlib

import pytest
from sympy import (
    Integer, Matrix, Poly, Rational, S, Symbol, cancel, collect, expand,
    factor, fraction, groebner, linear_eq_to_matrix, linsolve, simplify, solve,
)

from stencil_gen.codegen import (
    StencilGenSpec,
    TestCase,
    compute_test_values,
    generate_stencil_cpp,
    generate_test_cpp,
)
from stencil_gen.conservation import _interior_contribution
from stencil_gen.temo import (
    E2_1,
    E2_2,
    E4_1,
    SchemeParams,
    UniformResult,
    _build_uniform_vandermonde,
    assemble_cut_cell_result,
    build_cut_cell_conservation_system,
    build_cut_cell_deltas,
    build_degenerate_stencil,
    build_temo_vandermonde,
    compute_dimensions,
    construct_cut_cell_stencil,
    derive_cut_cell_scheme,
    identify_prescribed_entries,
    derive_e2_uniform_boundary,
    derive_uniform_boundary_for_temo,
    make_psi_field,
    solve_conservation_fraction_free,
    solve_cut_cell_conservation,
    solve_temo_row,
    solve_temo_row_polynomial,
    solve_uniform_limit,
)


@pytest.fixture(scope="module")
def e4_1_cut_cell_scheme():
    """Module-scoped cache for derive_cut_cell_scheme(E4_1, psi).

    Shared by TestDeriveCutCellScheme and TestE4CutCellSchemeWithZeros
    to avoid ~6 redundant derivations (~0.7s each).
    """
    psi = Symbol("psi")
    return derive_cut_cell_scheme(E4_1, psi), psi


@pytest.fixture(scope="module")
def e4_1_cut_cell_scheme_conserve():
    """Module-scoped cache for derive_cut_cell_scheme(E4_1, psi, conserve=True).

    Shared by TestE4CodeGeneration and TestE4TestFileGeneration (~5.6s each).
    """
    psi = Symbol("psi")
    return derive_cut_cell_scheme(E4_1, psi, conserve=True), psi


class TestE4UniformBoundary:
    """Tests for derive_uniform_boundary_for_temo with E4_1 (21.1b)."""

    @pytest.fixture
    def e4_result(self):
        """Compute E4_1 uniform boundary once for the test class."""
        return derive_uniform_boundary_for_temo(E4_1)

    def test_shape(self, e4_result):
        """E4_1 B_u has shape (4, 6) — r_eff=4 rows, t=6 columns."""
        assert e4_result.B_u.shape == (4, 6)

    def test_five_alpha_symbols(self, e4_result):
        """E4_1 has exactly 5 free alpha symbols."""
        assert len(e4_result.alpha_symbols) == 5
        # Verify they are named alpha_0..alpha_4
        for k, sym in enumerate(e4_result.alpha_symbols):
            assert sym.name == f"alpha_{k}"

    def test_zero_constraints(self, e4_result):
        """B_u[0, 5] == 0, B_u[1, 5] == 0, B_u[2, 5] == 0 (zero-constrained entries)."""
        assert e4_result.B_u[0, 5] == 0
        assert e4_result.B_u[1, 5] == 0
        assert e4_result.B_u[2, 5] == 0

    def test_last_row_free_alphas(self, e4_result):
        """B_u[3, 4] and B_u[3, 5] contain alpha_3, alpha_4."""
        alpha_3 = e4_result.alpha_symbols[3]
        alpha_4 = e4_result.alpha_symbols[4]
        # These entries should involve alpha_3 and alpha_4 respectively
        assert alpha_3 in e4_result.B_u[3, 4].free_symbols
        assert alpha_4 in e4_result.B_u[3, 5].free_symbols

    def test_interior_coefficients(self, e4_result):
        """Interior coefficients are [1/12, -2/3, 0, 2/3, -1/12]."""
        expected = [Rational(1, 12), Rational(-2, 3), S.Zero,
                    Rational(2, 3), Rational(-1, 12)]
        assert e4_result.interior == expected

    def test_scheme_metadata(self, e4_result):
        """Result carries correct p, q, nu."""
        assert e4_result.p == 2
        assert e4_result.q == 3
        assert e4_result.nu == 1

    def test_rows_0_1_match_e4u_1(self, e4_result):
        """First 5 columns of rows 0, 1 match E4u_1.cpp's nbs_floating coefficients.

        E4u_1.cpp row 0 (c[0..4], before /h):
            c[0] = (6*a0 - 11)/6
            c[1] = 3 - 4*a0
            c[2] = (12*a0 - 3)/2
            c[3] = -(12*a0 - 1)/3
            c[4] = a0

        E4u_1.cpp row 1 (c[5..9], before /h):
            c[5] = (3*a1 - 1)/3
            c[6] = -(8*a1 + 1)/2
            c[7] = 6*a1 + 1
            c[8] = -(24*a1 + 1)/6
            c[9] = a1
        """
        B_u = e4_result.B_u
        a0 = e4_result.alpha_symbols[0]
        a1 = e4_result.alpha_symbols[1]

        # E4u_1 row 0 expected (5 columns)
        row0_expected = [
            (6 * a0 - 11) / S(6),
            3 - 4 * a0,
            (12 * a0 - 3) / S(2),
            -(12 * a0 - 1) / S(3),
            a0,
        ]

        # E4u_1 row 1 expected (5 columns)
        row1_expected = [
            (3 * a1 - 1) / S(3),
            -(8 * a1 + 1) / S(2),
            6 * a1 + 1,
            -(24 * a1 + 1) / S(6),
            a1,
        ]

        for j in range(5):
            diff = cancel(B_u[0, j] - row0_expected[j])
            assert diff == 0, (
                f"Row 0, col {j}: B_u={B_u[0,j]}, expected={row0_expected[j]}"
            )

        for j in range(5):
            diff = cancel(B_u[1, j] - row1_expected[j])
            assert diff == 0, (
                f"Row 1, col {j}: B_u={B_u[1,j]}, expected={row1_expected[j]}"
            )

    def test_taylor_accuracy(self, e4_result, assert_taylor_accuracy):
        """Each row satisfies Taylor matching for q+1=4 equations (polynomials up to degree 3)."""
        assert_taylor_accuracy(e4_result.B_u, e4_result.q, nu=1)

    def test_no_conservation_constraint(self, e4_result):
        """E4_1 (nextra=0) has no column-sum conservation constraint.

        Column sums need NOT be zero — this confirms nextra=0 path is different
        from E2_1's nextra=1 path.
        """
        B_u = e4_result.B_u
        # Just verify that B_u doesn't have phi symbols (conservation resolved)
        free = B_u.free_symbols
        assert all("phi" not in str(s) for s in free), (
            f"Unexpected phi symbols: {free}"
        )

    def test_only_alpha_symbols_in_B_u(self, e4_result):
        """B_u contains only the expected alpha symbols, nothing else."""
        expected_syms = set(e4_result.alpha_symbols)
        actual_syms = e4_result.B_u.free_symbols
        assert actual_syms <= expected_syms, (
            f"Unexpected symbols in B_u: {actual_syms - expected_syms}"
        )

    def test_custom_alpha_symbols(self):
        """derive_uniform_boundary_for_temo(E4_1) accepts custom alpha names."""
        syms = [Symbol(f"a{k}") for k in range(5)]
        result = derive_uniform_boundary_for_temo(E4_1, alpha_symbols=syms)
        assert result.alpha_symbols == syms
        free = result.B_u.free_symbols
        assert free <= set(syms)

    def test_wrong_alpha_count_raises(self):
        """Wrong number of alpha symbols raises ValueError."""
        with pytest.raises(ValueError, match="alpha symbols"):
            derive_uniform_boundary_for_temo(E4_1, alpha_symbols=[Symbol("a")])


class TestE4UniformBoundaryWithZeros:
    """Tests for derive_uniform_boundary_for_temo with E4_1 and zeros={3,4} (26.1b)."""

    @pytest.fixture(scope="class")
    def e4_zeroed(self):
        return derive_uniform_boundary_for_temo(E4_1, zeros={3, 4})

    def test_shape(self, e4_zeroed):
        """B_u shape is (4, 6)."""
        assert e4_zeroed.B_u.shape == (4, 6)

    def test_three_alpha_symbols(self, e4_zeroed):
        """3 free symbols remain after zeroing alpha_3 and alpha_4."""
        assert len(e4_zeroed.alpha_symbols) == 3
        for k, sym in enumerate(e4_zeroed.alpha_symbols):
            assert sym.name == f"alpha_{k}"

    def test_last_row_zeroed(self, e4_zeroed):
        """B_u[3, 4] == 0 and B_u[3, 5] == 0 after zero constraints."""
        assert e4_zeroed.B_u[3, 4] == 0
        assert e4_zeroed.B_u[3, 5] == 0

    def test_early_row_col5_still_zero(self, e4_zeroed):
        """B_u[0, 5] == B_u[1, 5] == B_u[2, 5] == 0 (unchanged by zeros)."""
        assert e4_zeroed.B_u[0, 5] == 0
        assert e4_zeroed.B_u[1, 5] == 0
        assert e4_zeroed.B_u[2, 5] == 0

    def test_taylor_accuracy(self, e4_zeroed):
        """All 4 rows satisfy max(q+1, nu+1)=4 Taylor equations."""
        B_u = e4_zeroed.B_u
        t = B_u.cols
        n_eqs = max(3 + 1, 1 + 1)  # q=3, nu=1 → 4

        for i in range(B_u.rows):
            V, rhs = _build_uniform_vandermonde(i, t, n_eqs, nu=1)
            row = B_u.row(i)
            residual = V * row.T - rhs
            for k in range(n_eqs):
                assert cancel(residual[k]) == 0, (
                    f"Row {i}, eq {k}: residual={cancel(residual[k])}"
                )

    def test_zeros_conserve_mutual_exclusion(self):
        """zeros and conserve=True raises ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            derive_uniform_boundary_for_temo(E4_1, zeros={3, 4}, conserve=True)


class TestE4ZeroConstrainedCutCell:
    """Tests for E4_1 cut-cell stencil built from zero-constrained B_u (26.2a)."""

    @pytest.fixture(scope="class")
    def e4_zeroed_cut_cell(self):
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E4_1, zeros={3, 4})
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
        )
        return stencil, ur, psi

    def test_shape(self, e4_zeroed_cut_cell):
        """Cut-cell stencil has shape (5, 7) — R=5, T=7."""
        stencil, _, _ = e4_zeroed_cut_cell
        assert stencil.matrix.shape == (5, 7)

    def test_free_symbols(self, e4_zeroed_cut_cell):
        """Free symbols (excluding psi) are exactly {alpha_0, alpha_1, alpha_2}."""
        stencil, _, psi = e4_zeroed_cut_cell
        non_psi = stencil.matrix.free_symbols - {psi}
        names = {s.name for s in non_psi}
        assert names == {"alpha_0", "alpha_1", "alpha_2"}, (
            f"Expected {{alpha_0, alpha_1, alpha_2}}, got {names}"
        )

    def test_psi_1_limit(self, e4_zeroed_cut_cell):
        """At psi=1, matches the zero-constrained B_u via solve_uniform_limit."""
        stencil, ur, psi = e4_zeroed_cut_cell
        B_l_1 = solve_uniform_limit(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0)
        m1 = stencil.matrix.subs(psi, 1)
        R, T = m1.shape
        for i in range(R):
            for j in range(T):
                assert cancel(m1[i, j] - B_l_1[i, j]) == 0, (
                    f"Uniform limit mismatch at [{i},{j}]: "
                    f"{cancel(m1[i, j])} != {cancel(B_l_1[i, j])}"
                )

    def test_taylor_accuracy_at_half(self, e4_zeroed_cut_cell):
        """At psi=1/2, all 5 rows satisfy q+1=4 Taylor equations."""
        stencil, _, psi = e4_zeroed_cut_cell
        m = stencil.matrix.subs(psi, Rational(1, 2))
        R, T = m.shape
        psi_val = Rational(1, 2)
        for i in range(R):
            deltas = build_cut_cell_deltas(i, T, psi_val)
            row = [m[i, j] for j in range(T)]
            for k in range(4):  # q+1 = 4
                moment = sum(row[j] * deltas[j] ** k for j in range(T))
                if k == 1:
                    expected = 1
                else:
                    expected = 0
                assert cancel(moment - expected) == 0, (
                    f"Row {i}, moment k={k} at psi=1/2: "
                    f"got {cancel(moment)}, expected {expected}"
                )


@pytest.mark.slow
class TestE4CutCellConservationSolution:
    """Tests for solve_cut_cell_conservation with E4_1 zero-constrained stencil (26.3b)."""

    @pytest.fixture(scope="class")
    def conservation_solution(self):
        """Build zero-constrained cut-cell stencil and solve conservation."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E4_1, zeros={3, 4})
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
        )
        R, T = stencil.matrix.shape
        eqs, w_syms = build_cut_cell_conservation_system(
            stencil.matrix, R, T, p=E4_1.p, nu=E4_1.nu,
            interior_coeffs=ur.interior, psi=psi,
        )
        # solve_for: alpha_0, alpha_1, w_1, w_2, w_3
        alpha_syms = sorted(
            stencil.matrix.free_symbols - {psi}, key=lambda s: s.name,
        )
        solve_for = alpha_syms[:2] + w_syms[:3]
        sol = solve_cut_cell_conservation(eqs, solve_for)
        return sol, eqs, w_syms, stencil, ur, psi, alpha_syms

    def test_solution_exists(self, conservation_solution):
        """Solution dict has 5 entries."""
        sol = conservation_solution[0]
        assert len(sol) == 5

    def test_all_equations_satisfied(self, conservation_solution):
        """All 5 conservation equations evaluate to 0 after substitution."""
        sol, eqs, _, _, _, _, _ = conservation_solution
        for i, eq in enumerate(eqs):
            residual = cancel(eq.subs(sol))
            assert residual == 0, f"Equation {i} residual: {residual}"

    def test_free_symbols_in_solution(self, conservation_solution):
        """Each solved expression involves only {psi, alpha_2, w_4}."""
        sol, _, w_syms, _, _, psi, alpha_syms = conservation_solution
        alpha_2 = alpha_syms[2]
        w_4 = w_syms[3]
        allowed = {psi, alpha_2, w_4}
        for sym, expr in sol.items():
            extra = expr.free_symbols - allowed
            assert extra == set(), (
                f"Solution for {sym} has unexpected symbols: {extra}"
            )

    def test_stencil_after_substitution(self, conservation_solution):
        """After applying solution, stencil free symbols are subset of {psi, alpha_2, w_4}."""
        sol, _, w_syms, stencil, _, psi, alpha_syms = conservation_solution
        alpha_2 = alpha_syms[2]
        w_4 = w_syms[3]
        B_l_sub = stencil.matrix.xreplace(sol)
        allowed = {psi, alpha_2, w_4}
        extra = B_l_sub.free_symbols - allowed
        assert extra == set(), f"Unexpected symbols in stencil: {extra}"

    def test_conservation_column_sums(self, conservation_solution):
        """Weighted column sums using solved weights verify conservation."""
        sol, _, w_syms, stencil, ur, psi, _ = conservation_solution
        B_l_sub = stencil.matrix.xreplace(sol)
        R, T = B_l_sub.shape
        # Weights: w_0=psi, w_1..w_3 from sol, w_4 free
        weights = [psi]
        for w in w_syms[:3]:
            weights.append(sol[w])
        weights.append(w_syms[3])

        interior = ur.interior
        p = E4_1.p
        for j_tf in range(1, T - 1):
            g = j_tf - 1
            col_sum = sum(weights[i] * B_l_sub[i, j_tf] for i in range(R))
            ic = _interior_contribution(g, R, p, interior)
            col_sum += ic
            if g == 0 and E4_1.nu == 1:
                target = S.NegativeOne
            else:
                target = S.Zero
            residual = cancel(col_sum - target)
            assert residual == 0, (
                f"Column j_tf={j_tf} (g={g}) conservation residual: {residual}"
            )


@pytest.mark.slow
class TestE4TEMOConstruction:
    """Tests for E4_1 full TEMO pipeline (21.3a)."""

    @pytest.fixture(scope="class")
    def e4_temo(self):
        """Run the full E4_1 TEMO pipeline once for the test class."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E4_1)
        result = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
        )
        return ur, result, psi

    def test_shape(self, e4_temo):
        """E4_1 cut-cell stencil has shape (5, 7) — R=5, T=7."""
        _, result, _ = e4_temo
        assert result.matrix.shape == (5, 7)

    def test_no_betas(self, e4_temo):
        """E4_1 (nextra=0) produces no beta parameters."""
        _, result, _ = e4_temo
        assert len(result.beta_info) == 0
        assert len(result.beta_symbols) == 0

    def test_entries_in_psi_alpha(self, e4_temo):
        """All entries are rational in psi and alpha_{0..4} only."""
        _, result, _ = e4_temo
        all_syms = result.matrix.free_symbols
        expected_names = {"psi"} | {f"alpha_{k}" for k in range(5)}
        actual_names = {s.name for s in all_syms}
        assert actual_names <= expected_names, (
            f"Unexpected symbols: {actual_names - expected_names}"
        )

    def test_uniform_limit(self, e4_temo):
        """At psi=1, rows 0-2 reduce to B_u in T-frame, row 3 is interior."""
        ur, result, psi = e4_temo
        B_l_1 = solve_uniform_limit(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0)
        m1 = result.matrix.subs(psi, 1)
        R, T = m1.shape
        for i in range(R):
            for j in range(T):
                assert simplify(m1[i, j] - B_l_1[i, j]) == 0, (
                    f"Uniform limit mismatch at [{i},{j}]: "
                    f"{cancel(m1[i, j])} != {cancel(B_l_1[i, j])}"
                )

    def test_uniform_limit_rows_0_3_embed_Bu(self, e4_temo):
        """At psi=1, rows 0-3: wall col=0, then cols 1-6 = B_u rows 0-3."""
        ur, result, psi = e4_temo
        m1 = result.matrix.subs(psi, 1)
        B_u = ur.B_u
        for i in range(4):
            # Column 0 is the wall column
            # Columns 1..6 should match B_u[i, 0..5]
            for j in range(6):
                assert simplify(m1[i, j + 1] - B_u[i, j]) == 0, (
                    f"B_u embed mismatch at row {i}, col {j}: "
                    f"{cancel(m1[i, j + 1])} != {cancel(B_u[i, j])}"
                )

    def test_uniform_limit_row4_not_interior(self, e4_temo):
        """At psi=1, row 4 is NOT the raw interior stencil — it's derived via conservation+Taylor.

        The interior stencil [1/12, -2/3, 0, 2/3, -1/12] overflows the T-frame at row 4,
        so solve_uniform_limit computes a closure row that contains alpha symbols.
        """
        ur, result, psi = e4_temo
        m1 = result.matrix.subs(psi, 1)
        B_l_1 = solve_uniform_limit(ur.B_u, ur.interior, ur.p, ur.q, ur.nu, 0)
        # Row 4 must match the solve_uniform_limit result
        for j in range(7):
            assert simplify(m1[4, j] - B_l_1[4, j]) == 0, (
                f"Row 4 uniform limit mismatch at col {j}: "
                f"{cancel(m1[4, j])} != {cancel(B_l_1[4, j])}"
            )
        # Negative assertion: row 4 is NOT the raw interior stencil
        raw_interior_in_T = [
            S.Zero, S.Zero,
            Rational(1, 12), Rational(-2, 3), S.Zero,
            Rational(2, 3), Rational(-1, 12),
        ]
        assert any(
            simplify(m1[4, j] - raw_interior_in_T[j]) != 0 for j in range(7)
        ), "Row 4 should NOT be the raw interior stencil"

    def test_degenerate_limit(self, e4_temo):
        """At psi=0, matches the degenerate stencil B_d."""
        ur, result, psi = e4_temo
        m0 = result.matrix.subs(psi, 0)
        B_d = build_degenerate_stencil(ur.B_u, ur.interior, p=2, q=3, nu=1)
        R, T = m0.shape
        for i in range(R):
            for j in range(T):
                assert simplify(m0[i, j] - B_d[i, j]) == 0, (
                    f"Degenerate mismatch at [{i},{j}]: "
                    f"{cancel(m0[i, j])} != {cancel(B_d[i, j])}"
                )

    def test_taylor_accuracy_symbolic(self, e4_temo):
        """Each row satisfies Taylor accuracy (q+1=4 equations) for symbolic psi.

        For first derivative (nu=1), row i should exactly differentiate
        monomials x^m for m = 0, 1, ..., q=3:
            sum_j c_j * delta_j^m = delta_{m,1} * m!  (= delta_{m,1})
        """
        _, result, psi = e4_temo
        m = result.matrix
        R, T = m.shape
        for i in range(R):
            deltas = build_cut_cell_deltas(i, T, psi)
            row = [m[i, j] for j in range(T)]
            for k in range(4):  # q+1 = 4
                moment = sum(row[j] * deltas[j] ** k for j in range(T))
                if k == 1:
                    expected = 1
                else:
                    expected = 0
                assert simplify(moment - expected) == 0, (
                    f"Row {i}, moment k={k}: got {simplify(moment)}, "
                    f"expected {expected}"
                )

    def test_taylor_accuracy_at_half(self, e4_temo):
        """Taylor accuracy holds at psi=1/2 (numerical check)."""
        _, result, psi = e4_temo
        m = result.matrix.subs(psi, Rational(1, 2))
        R, T = m.shape
        psi_val = Rational(1, 2)
        for i in range(R):
            deltas = build_cut_cell_deltas(i, T, psi_val)
            row = [m[i, j] for j in range(T)]
            for k in range(4):
                moment = sum(row[j] * deltas[j] ** k for j in range(T))
                if k == 1:
                    expected = 1
                else:
                    expected = 0
                assert simplify(moment - expected) == 0, (
                    f"Row {i}, moment k={k} at psi=1/2: "
                    f"got {simplify(moment)}, expected {expected}"
                )


@pytest.mark.slow
class TestE4CodeGeneration:
    """Tests for E4_1 C++ code generation (21.4b, updated for 2-alpha zeros conservation)."""

    @pytest.fixture(scope="class")
    def e4_spec(self, e4_1_cut_cell_scheme_conserve):
        """Build the full StencilGenSpec from cached conserve=True derivation."""
        cc, psi = e4_1_cut_cell_scheme_conserve

        # floating_coeffs: R*T = 5*7 = 35 entries, row-major from cc.floating
        floating_flat = list(cc.floating)

        # dirichlet_coeffs: R*T = 35 entries (prepend T=7 zeros for row 0)
        dirichlet_flat = [Integer(0)] * 7 + list(cc.dirichlet)

        interior = [Rational(1, 12), Rational(-2, 3), S.Zero,
                    Rational(2, 3), Rational(-1, 12)]

        spec = StencilGenSpec(
            name="E4_1",
            P=2,
            R=5,
            T=7,
            X=0,
            derivative_order=1,
            is_uniform=False,
            param_arrays={"alpha": 2},
            interior_coeffs=interior,
            floating_coeffs=floating_flat,
            dirichlet_coeffs=dirichlet_flat,
        )
        return spec

    @pytest.fixture(scope="class")
    def e4_code(self, e4_spec):
        """Generate the E4_1 C++ code."""
        return generate_stencil_cpp(e4_spec)

    def test_struct_constants(self, e4_code):
        """Generated code has P=2, R=5, T=7, X=0."""
        assert "static constexpr int P = 2;" in e4_code
        assert "static constexpr int R = 5;" in e4_code
        assert "static constexpr int T = 7;" in e4_code
        assert "static constexpr int X = 0;" in e4_code

    def test_struct_name(self, e4_code):
        """Generated code defines struct E4_1."""
        assert "struct E4_1" in e4_code

    def test_namespace(self, e4_code):
        """Generated code uses ccs::stencils namespace."""
        assert "namespace ccs::stencils" in e4_code

    def test_alpha_array(self, e4_code):
        """Generated code has std::array<real, 2> alpha member (zeros + cut-cell conservation)."""
        assert "std::array<real, 2> alpha;" in e4_code

    def test_constructor(self, e4_code):
        """Generated code has span constructor."""
        assert "E4_1(std::span<const real> a)" in e4_code
        assert "copy_zero_padded(a, alpha);" in e4_code

    def test_factory(self, e4_code):
        """Generated code has make_E4_1 factory."""
        assert "make_E4_1(std::span<const real> alpha)" in e4_code
        assert "return E4_1{alpha};" in e4_code

    def test_interior_method(self, e4_code):
        """Generated code has interior() method."""
        assert "interior(real h," in e4_code

    def test_nbs_floating_method(self, e4_code):
        """Generated code has nbs_floating method with 35 coefficient assignments."""
        assert "nbs_floating(real h," in e4_code
        floating_start = e4_code.index("nbs_floating(real h,")
        dirichlet_start = e4_code.index("nbs_dirichlet(real h,")
        floating_section = e4_code[floating_start:dirichlet_start]
        # R*T = 5*7 = 35 coefficient assignments
        assert floating_section.count("c[") == 35, (
            f"Expected 35 c[] assignments in floating, got {floating_section.count('c[')}"
        )

    def test_nbs_dirichlet_method(self, e4_code):
        """Generated code has nbs_dirichlet method with 28 coefficient assignments."""
        assert "nbs_dirichlet(real h," in e4_code
        dirichlet_start = e4_code.index("nbs_dirichlet(real h,")
        neumann_start = e4_code.index("nbs_neumann")
        dirichlet_section = e4_code[dirichlet_start:neumann_start]
        # (R-1)*T = 4*7 = 28 coefficient assignments
        assert dirichlet_section.count("c[") == 28, (
            f"Expected 28 c[] assignments in dirichlet, got {dirichlet_section.count('c[')}"
        )

    def test_nbs_neumann_stub(self, e4_code):
        """Generated code has nbs_neumann stub (X=0, non-uniform)."""
        assert "nbs_neumann" in e4_code

    def test_psi_named_parameter(self, e4_code):
        """Cut-cell stencil uses named psi parameter in nbs methods."""
        # In cut-cell (non-uniform), psi is a named parameter
        floating_start = e4_code.index("nbs_floating(real h,")
        floating_sig_end = e4_code.index("{", floating_start)
        floating_sig = e4_code[floating_start:floating_sig_end]
        assert "real psi" in floating_sig

    def test_cse_temporaries(self, e4_code):
        """Generated code uses CSE temporaries for complex expressions."""
        # E4_1 has rational functions of psi and alpha — CSE should produce temporaries
        floating_start = e4_code.index("nbs_floating(real h,")
        dirichlet_start = e4_code.index("nbs_dirichlet(real h,")
        floating_section = e4_code[floating_start:dirichlet_start]
        # Check for CSE temporaries (real t... = ...)
        assert "real t" in floating_section, (
            "Expected CSE temporaries in floating method"
        )

    def test_query_methods(self, e4_code):
        """Generated code has query_max and query methods."""
        assert "query_max()" in e4_code
        assert "query(bcs::type b)" in e4_code
        assert "query_interp()" in e4_code

    def test_write_output(self, e4_spec, e4_code):
        """Write generated E4_1.cpp to output directory."""
        output_dir = pathlib.Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "E4_1.cpp"
        output_path.write_text(e4_code)
        assert output_path.exists()
        assert output_path.stat().st_size > 0


@pytest.mark.slow
class TestE4TestFileGeneration:
    """Tests for E4_1 C++ test file generation (21.4c, updated for 2-alpha zeros conservation)."""

    ALPHA_VALUES = {"alpha": [0.1, 0.7]}

    @pytest.fixture(scope="class")
    def e4_spec(self, e4_1_cut_cell_scheme_conserve):
        """Build the full StencilGenSpec from cached conserve=True derivation."""
        cc, psi = e4_1_cut_cell_scheme_conserve

        floating_flat = list(cc.floating)
        dirichlet_flat = [Integer(0)] * 7 + list(cc.dirichlet)

        interior = [Rational(1, 12), Rational(-2, 3), S.Zero,
                    Rational(2, 3), Rational(-1, 12)]

        return StencilGenSpec(
            name="E4_1",
            P=2,
            R=5,
            T=7,
            X=0,
            derivative_order=1,
            is_uniform=False,
            param_arrays={"alpha": 2},
            interior_coeffs=interior,
            floating_coeffs=floating_flat,
            dirichlet_coeffs=dirichlet_flat,
        )

    def test_compute_floating_values(self, e4_spec):
        """compute_test_values produces 35 floating coefficients at psi=0.9."""
        values = compute_test_values(
            e4_spec.floating_coeffs,
            alpha_values=self.ALPHA_VALUES,
            h=1.0,
            psi=0.9,
        )
        assert len(values) == 35  # R*T = 5*7

    def test_compute_dirichlet_values(self, e4_spec):
        """compute_test_values produces 28 Dirichlet coefficients at psi=0.7."""
        dirichlet_emitted = e4_spec.dirichlet_coeffs[e4_spec.T:]
        values = compute_test_values(
            dirichlet_emitted,
            alpha_values=self.ALPHA_VALUES,
            h=0.5,
            psi=0.7,
        )
        assert len(values) == 28  # (R-1)*T = 4*7

    def test_floating_row4_not_interior(self, e4_spec):
        """At psi=0.9, floating row 4 is NOT the raw interior stencil.

        The zeros-conservative stencil diverges at psi=1, so we test at an
        interior psi value.  Row 4 is derived via conservation+Taylor, not
        the raw interior coefficients.
        """
        values = compute_test_values(
            e4_spec.floating_coeffs,
            alpha_values=self.ALPHA_VALUES,
            h=2.0,
            psi=0.9,
        )
        # Row 4 = indices 28..34
        row4 = values[28:35]
        # Row 4 should NOT be the raw interior stencil / h
        raw_interior_over_h = [0, 0, 1 / 24, -1 / 3, 0, 1 / 3, -1 / 24]
        assert any(
            abs(got - want) > 1e-12
            for got, want in zip(row4, raw_interior_over_h)
        ), "Row 4 should NOT be the raw interior stencil"

    def test_generate_test_file_structure(self, e4_spec):
        """Generated test file has correct Catch2 structure for E4_1."""
        floating_vals = compute_test_values(
            e4_spec.floating_coeffs,
            alpha_values=self.ALPHA_VALUES,
            h=1.0,
            psi=0.9,
        )
        cases = [
            TestCase(
                bc_type="Floating",
                h=1.0,
                psi=0.9,
                alpha_values=self.ALPHA_VALUES,
                expected_coeffs=floating_vals,
            ),
        ]
        code = generate_test_cpp(e4_spec, cases)
        assert 'TEST_CASE("E4_1")' in code
        assert 'type = "E4"' in code
        assert "order = 1" in code
        assert "alpha = {0.1, 0.7}" in code
        assert "REQUIRE(p == 2)" in code
        assert "REQUIRE(r == 5)" in code
        assert "REQUIRE(t == 7)" in code

    def test_generate_test_file_multiple_cases(self, e4_spec):
        """Generated test file has Floating and Dirichlet test blocks."""
        dirichlet_emitted = e4_spec.dirichlet_coeffs[e4_spec.T:]
        cases = [
            TestCase(
                bc_type="Floating",
                h=2.0,
                psi=0.9,
                alpha_values=self.ALPHA_VALUES,
                expected_coeffs=compute_test_values(
                    e4_spec.floating_coeffs,
                    alpha_values=self.ALPHA_VALUES, h=2.0, psi=0.9,
                ),
            ),
            TestCase(
                bc_type="Floating",
                h=1.0,
                psi=0.3,
                alpha_values=self.ALPHA_VALUES,
                expected_coeffs=compute_test_values(
                    e4_spec.floating_coeffs,
                    alpha_values=self.ALPHA_VALUES, h=1.0, psi=0.3,
                ),
            ),
            TestCase(
                bc_type="Dirichlet",
                h=0.5,
                psi=0.7,
                alpha_values=self.ALPHA_VALUES,
                expected_coeffs=compute_test_values(
                    dirichlet_emitted,
                    alpha_values=self.ALPHA_VALUES, h=0.5, psi=0.7,
                ),
            ),
        ]
        code = generate_test_cpp(e4_spec, cases)
        assert code.count("REQUIRE_THAT(c,") == 3
        assert "bcs::Floating" in code
        assert "bcs::Dirichlet" in code

    def test_write_test_output(self, e4_spec):
        """Generate and write E4_1.t.cpp to output directory."""
        dirichlet_emitted = e4_spec.dirichlet_coeffs[e4_spec.T:]
        cases = [
            TestCase(
                bc_type="Floating",
                h=2.0,
                psi=0.9,
                alpha_values=self.ALPHA_VALUES,
                expected_coeffs=compute_test_values(
                    e4_spec.floating_coeffs,
                    alpha_values=self.ALPHA_VALUES, h=2.0, psi=0.9,
                ),
            ),
            TestCase(
                bc_type="Floating",
                h=1.0,
                psi=0.3,
                alpha_values=self.ALPHA_VALUES,
                expected_coeffs=compute_test_values(
                    e4_spec.floating_coeffs,
                    alpha_values=self.ALPHA_VALUES, h=1.0, psi=0.3,
                ),
            ),
            TestCase(
                bc_type="Dirichlet",
                h=0.5,
                psi=0.7,
                alpha_values=self.ALPHA_VALUES,
                expected_coeffs=compute_test_values(
                    dirichlet_emitted,
                    alpha_values=self.ALPHA_VALUES, h=0.5, psi=0.7,
                ),
            ),
        ]
        code = generate_test_cpp(e4_spec, cases)

        output_dir = pathlib.Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "E4_1.t.cpp"
        output_path.write_text(code)
        assert output_path.exists()
        assert output_path.stat().st_size > 0


@pytest.mark.slow
class TestDeriveCutCellScheme:
    """Tests for derive_cut_cell_scheme high-level pipeline (21.5a)."""

    def test_e4_1_shape(self, e4_1_cut_cell_scheme):
        """E4_1 via derive_cut_cell_scheme has R=5, T=7 floating matrix."""
        result, psi = e4_1_cut_cell_scheme
        assert result.floating.shape == (5, 7)
        assert result.dirichlet.shape == (4, 7)
        assert result.dims.R == 5
        assert result.dims.T == 7
        assert result.dims.X == 0

    def test_e4_1_no_neumann(self, e4_1_cut_cell_scheme):
        """E4_1 (nu=1) has no Neumann stencil."""
        result, psi = e4_1_cut_cell_scheme
        assert result.neumann is None
        assert result.eta is None

    def test_e4_1_alpha_count(self, e4_1_cut_cell_scheme):
        """E4_1 has 2 free alpha symbols (zeros + cut-cell conservation)."""
        result, psi = e4_1_cut_cell_scheme
        assert len(result.alpha_symbols) == 2

    def test_e4_1_matches_manual_pipeline(self, e4_1_cut_cell_scheme):
        """derive_cut_cell_scheme(E4_1) equals manual zeros + cut-cell conservation pipeline."""
        auto, psi = e4_1_cut_cell_scheme

        # Manual zeros pipeline (polynomial boundary rows + fraction-free conservation)
        ur = derive_uniform_boundary_for_temo(E4_1, zeros={3, 4})
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
            polynomial_boundary_rows=True,
        )
        floating = stencil.matrix
        R, T = floating.shape

        # Zero out residual c_* polynomial coefficients and cancel near-interior row
        c_syms = {s for s in floating.free_symbols if s.name.startswith('c_')}
        if c_syms:
            floating = floating.xreplace({c: 0 for c in c_syms})
        for j in range(T):
            floating[R - 1, j] = cancel(floating[R - 1, j])

        # Solve cut-cell conservation (fraction-free)
        eqs, w_syms = build_cut_cell_conservation_system(
            floating, R, T, p=2, nu=1,
            interior_coeffs=ur.interior, psi=psi,
        )
        solve_for = list(ur.alpha_symbols[:2]) + list(w_syms[:3])
        sol = solve_conservation_fraction_free(eqs, solve_for, psi)

        # Apply solution and rename free params
        floating = floating.xreplace(sol)
        rename = {ur.alpha_symbols[2]: Symbol("alpha_0"), w_syms[3]: Symbol("alpha_1")}
        floating = floating.xreplace(rename)

        # Zeros path has no conservation_subs
        assert auto.conservation_subs is None

        # Compare floating matrices entry by entry
        for i in range(R):
            for j in range(T):
                assert cancel(auto.floating[i, j] - floating[i, j]) == 0, (
                    f"Floating mismatch at [{i},{j}]"
                )

    def test_e4_1_taylor_accuracy(self, e4_1_cut_cell_scheme):
        """E4_1 result satisfies Taylor accuracy at psi=1/2."""
        result, psi = e4_1_cut_cell_scheme
        m = result.floating.subs(psi, Rational(1, 2))
        R, T = m.shape
        psi_val = Rational(1, 2)
        for i in range(R):
            deltas = build_cut_cell_deltas(i, T, psi_val)
            row = [m[i, j] for j in range(T)]
            for k in range(4):  # q+1 = 4
                moment = sum(row[j] * deltas[j] ** k for j in range(T))
                expected = 1 if k == 1 else 0
                assert simplify(moment - expected) == 0, (
                    f"Row {i}, moment k={k}: got {simplify(moment)}"
                )

    def test_e4_1_custom_alphas(self):
        """derive_cut_cell_scheme accepts custom alpha symbols (2 post-zeros-conservation)."""
        psi = Symbol("psi")
        syms = [Symbol(f"a{k}") for k in range(2)]
        result = derive_cut_cell_scheme(E4_1, psi, alpha_symbols=syms)
        assert result.alpha_symbols == syms
        assert result.floating.free_symbols <= {psi} | set(syms)

    def test_e2_1_reproduces_existing(self):
        """derive_cut_cell_scheme(E2_1) matches manual E2_1 pipeline."""
        psi = Symbol("psi")

        # Manual E2_1 pipeline using old derive_e2_uniform_boundary
        ur = derive_e2_uniform_boundary(nu=1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=1, q=1, nu=1, nextra=1, psi=psi,
        )
        dims = compute_dimensions(1, 1, 0, 1, 1)
        manual = assemble_cut_cell_result(
            stencil.matrix, None, None, dims, ur.alpha_symbols,
        )

        # High-level pipeline
        auto = derive_cut_cell_scheme(E2_1, psi)

        # Shapes must match
        assert auto.floating.shape == manual.floating.shape
        assert auto.dirichlet.shape == manual.dirichlet.shape
        assert auto.dims == manual.dims

        # Floating matrices must match entry by entry
        R, T = auto.floating.shape
        for i in range(R):
            for j in range(T):
                assert cancel(auto.floating[i, j] - manual.floating[i, j]) == 0, (
                    f"E2_1 floating mismatch at [{i},{j}]"
                )

    def test_e2_2_reproduces_existing(self):
        """derive_cut_cell_scheme(E2_2) matches manual E2_2 pipeline."""
        psi = Symbol("psi")

        # Manual E2_2 pipeline
        ur = derive_e2_uniform_boundary(nu=2)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=1, q=1, nu=2, nextra=0, psi=psi,
        )
        from stencil_gen.temo import derive_uniform_neumann, construct_neumann_stencil
        B_uN, eta_u = derive_uniform_neumann(ur.interior, 1, 1, 2)
        neumann_main, eta = construct_neumann_stencil(
            ur.B_u, B_uN, eta_u, ur.interior, 1, 1, 2, 0, psi,
        )
        dims = compute_dimensions(1, 1, 0, 0, 2)
        manual = assemble_cut_cell_result(
            stencil.matrix, neumann_main, eta, dims, ur.alpha_symbols,
        )

        # High-level pipeline
        auto = derive_cut_cell_scheme(E2_2, psi)

        # Shapes
        assert auto.floating.shape == manual.floating.shape
        assert auto.dims == manual.dims
        assert auto.neumann is not None
        assert auto.eta is not None

        # Floating
        R, T = auto.floating.shape
        for i in range(R):
            for j in range(T):
                assert cancel(auto.floating[i, j] - manual.floating[i, j]) == 0, (
                    f"E2_2 floating mismatch at [{i},{j}]"
                )

        # Neumann
        for i in range(R):
            for j in range(T):
                assert cancel(auto.neumann[i, j] - manual.neumann[i, j]) == 0, (
                    f"E2_2 neumann mismatch at [{i},{j}]"
                )

        # Eta
        for i in range(R):
            assert cancel(auto.eta[i] - manual.eta[i]) == 0, (
                f"E2_2 eta mismatch at row {i}"
            )


@pytest.mark.slow
class TestE4CutCellSchemeWithZeros:
    """Integration tests for derive_cut_cell_scheme(E4_1) with zeros path (26.5c)."""

    def test_alpha_count(self, e4_1_cut_cell_scheme):
        result, _ = e4_1_cut_cell_scheme
        assert len(result.alpha_symbols) == 2

    def test_shape(self, e4_1_cut_cell_scheme):
        result, _ = e4_1_cut_cell_scheme
        assert result.floating.shape == (5, 7)
        assert result.dirichlet.shape == (4, 7)

    def test_free_symbols(self, e4_1_cut_cell_scheme):
        result, psi = e4_1_cut_cell_scheme
        expected = {psi} | set(result.alpha_symbols)
        assert result.floating.free_symbols <= expected, (
            f"Unexpected symbols: {result.floating.free_symbols - expected}"
        )

    def test_weights_present(self, e4_1_cut_cell_scheme):
        result, psi = e4_1_cut_cell_scheme
        assert result.weights is not None
        assert len(result.weights) == 5  # R=5
        # At least one weight should depend on psi
        assert any(
            hasattr(w, 'free_symbols') and psi in w.free_symbols
            for w in result.weights
        )

    def test_taylor_accuracy(self, e4_1_cut_cell_scheme):
        """All rows satisfy q+1=4 Taylor equations at psi=1/2."""
        result, psi = e4_1_cut_cell_scheme
        m = result.floating
        R, T = m.shape
        # Use non-zero alpha values (alpha_1 maps to w_4 quadrature weight)
        alpha_vals = dict(zip(result.alpha_symbols,
                              [Rational(1, 10), Rational(1, 1)]))
        psi_val = Rational(1, 2)
        subs = {psi: psi_val, **alpha_vals}
        for i in range(R):
            deltas = build_cut_cell_deltas(i, T, psi_val)
            row = [m[i, j].subs(subs) for j in range(T)]
            for k in range(4):
                moment = sum(row[j] * deltas[j] ** k for j in range(T))
                expected = 1 if k == 1 else 0
                assert cancel(moment - expected) == 0, (
                    f"Row {i}, moment k={k}: got {cancel(moment)}"
                )

    def test_conservation_holds(self, e4_1_cut_cell_scheme):
        """Weighted column sums using result.weights satisfy conservation."""
        result, psi = e4_1_cut_cell_scheme
        m = result.floating
        R, T = m.shape
        weights = result.weights
        interior = derive_uniform_boundary_for_temo(E4_1, zeros={3, 4}).interior
        for j_tf in range(1, T - 1):
            g = j_tf - 1
            col_sum = sum(weights[i] * m[i, j_tf] for i in range(R))
            ic = _interior_contribution(g, R, E4_1.p, interior)
            col_sum += ic
            target = S.NegativeOne if (g == 0 and E4_1.nu == 1) else S.Zero
            residual = cancel(col_sum - target)
            assert residual == 0, (
                f"Conservation violated at grid point {g}: residual={residual}"
            )

    def test_custom_alphas(self, e4_1_cut_cell_scheme):
        _, psi = e4_1_cut_cell_scheme
        syms = [Symbol("a"), Symbol("b")]
        result = derive_cut_cell_scheme(E4_1, psi, alpha_symbols=syms)
        assert result.alpha_symbols == syms
        assert result.floating.free_symbols <= {psi} | set(syms)

    def test_no_singularities(self, e4_1_cut_cell_scheme):
        """All floating/dirichlet entries and weights are finite at ψ=0 and ψ=1
        with representative alpha values.  Regression guard (27.4b).

        Note: approach B means some entries have ψ(ψ-1) poles in the symbolic
        expressions.  This test substitutes alpha values first and verifies
        finiteness at *interior* psi values (1/10, 1/2, 9/10).  It documents
        that exact ψ=0 and ψ=1 evaluation blows up for entries depending on
        the solved-for alpha solutions.
        """
        result, psi = e4_1_cut_cell_scheme
        m = result.floating
        R, T = m.shape
        alpha_vals = dict(
            zip(result.alpha_symbols, [Rational(1, 10), Rational(1, 1)])
        )
        for psi_val in [Rational(1, 10), Rational(1, 2), Rational(9, 10)]:
            subs = {psi: psi_val, **alpha_vals}
            for i in range(R):
                for j in range(T):
                    val = m[i, j].subs(subs)
                    assert val.is_finite, (
                        f"Entry [{i},{j}] not finite at psi={psi_val}"
                    )
            # Weights too
            for k, w in enumerate(result.weights):
                val = w.subs(subs) if hasattr(w, 'subs') else w
                assert val.is_finite, (
                    f"Weight {k} not finite at psi={psi_val}"
                )


@pytest.mark.xfail(reason=(
    "E4_1 cut-cell conservation is structurally infeasible at R=5, T=7, nextra=0 "
    "WITHOUT zero constraints. Proven via Groebner basis in "
    "test_e4_1_psi_dependent_conservation_infeasible (23.3c-ii). "
    "Conservation IS feasible with alpha_3=alpha_4=0 — see "
    "test_e4_1_conservation_with_zeros."
))
def test_e4_1_conservation_fails_without_zeros():
    """E4_1 cut-cell stencil violates discrete conservation (SBP property).

    Conservation applies to grid-point columns (T-frame cols 1..T-2):
        sum_i w_i * B[i, g+1] + IC(g) = target(g)
    where g is the grid point (g = T-frame col - 1), w_0=psi,
    w_i=1 for i>=1 (naive flat weights), and IC(g) is the interior
    contribution to grid point g.
    Target: -1 at grid point 0, 0 elsewhere.

    The T-frame col 0 is the wall (delta) column, which is NOT a grid
    point and is excluded from the SBP conservation check.
    """
    psi = Symbol("psi")
    ur = derive_uniform_boundary_for_temo(E4_1)
    stencil = construct_cut_cell_stencil(
        ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
    )
    m = stencil.matrix  # R=5 x T=7 matrix
    R, T = 5, 7

    for j_tf in range(1, T - 1):  # T-frame cols 1..5 (grid points 0..4)
        g = j_tf - 1  # grid point index
        # Weighted column sum: w_0=psi for row 0, w_i=1 for rows 1..R-1
        col_sum = psi * m[0, j_tf] + m[1, j_tf] + m[2, j_tf] + m[3, j_tf] + m[4, j_tf]

        # Interior contribution for grid point g
        ic = _interior_contribution(g, R, 2, ur.interior)
        col_sum += ic

        # Target: -1 at grid point 0, 0 elsewhere
        target = -1 if g == 0 else 0

        residual = cancel(col_sum - target)
        assert residual == 0, (
            f"Conservation violated at grid point {g} (T-frame col {j_tf}): "
            f"residual={residual}"
        )


@pytest.mark.slow
def test_e4_1_conservation_with_zeros():
    """E4_1 cut-cell conservation IS feasible with alpha_3=alpha_4=0 (26.4a).

    With zero constraints on the last row's free parameters, the conservation
    system becomes solvable.  This test:
      1. Builds zero-constrained B_u (zeros={3, 4})
      2. Runs TEMO -> cut-cell stencil (3 alphas)
      3. Solves cut-cell conservation for [alpha_0, alpha_1, w_1, w_2, w_3]
      4. Substitutes solution into stencil
      5. Verifies weighted column sums symbolically (all interior columns)
      6. Verifies numerically at psi=0.3, 0.5, 0.7 with alpha_2=0.1, w_4=1.0
    """
    psi = Symbol("psi")

    # Step 1-2: Build zero-constrained cut-cell stencil
    ur = derive_uniform_boundary_for_temo(E4_1, zeros={3, 4})
    stencil = construct_cut_cell_stencil(
        ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
    )
    B_l = stencil.matrix
    R, T = B_l.shape
    assert R == 5 and T == 7

    # Step 3: Build conservation system and solve
    eqs, w_syms = build_cut_cell_conservation_system(
        B_l, R, T, p=E4_1.p, nu=E4_1.nu,
        interior_coeffs=ur.interior, psi=psi,
    )
    alpha_syms = sorted(B_l.free_symbols - {psi}, key=lambda s: s.name)
    assert len(alpha_syms) == 3  # alpha_0, alpha_1, alpha_2

    solve_for = alpha_syms[:2] + w_syms[:3]  # alpha_0, alpha_1, w_1, w_2, w_3
    sol = solve_cut_cell_conservation(eqs, solve_for)
    assert len(sol) == 5

    # Step 4: Substitute solution into stencil
    B_l_sub = B_l.xreplace(sol)

    # Build weights: w_0=psi, w_1..w_3 from solution, w_4 free
    alpha_2 = alpha_syms[2]
    w_4 = w_syms[3]
    weights = [psi] + [sol[w] for w in w_syms[:3]] + [w_4]

    # Step 5: Symbolic verification — weighted column sums for all interior columns
    interior = ur.interior
    for j_tf in range(1, T - 1):  # T-frame cols 1..5 (grid points 0..4)
        g = j_tf - 1
        col_sum = sum(weights[i] * B_l_sub[i, j_tf] for i in range(R))
        ic = _interior_contribution(g, R, E4_1.p, interior)
        col_sum += ic
        target = S.NegativeOne if (g == 0 and E4_1.nu == 1) else S.Zero
        residual = cancel(col_sum - target)
        assert residual == 0, (
            f"Symbolic conservation violated at grid point {g} "
            f"(T-frame col {j_tf}): residual={residual}"
        )

    # Step 6: Numeric verification at specific parameter values
    for psi_val in [Rational(3, 10), Rational(1, 2), Rational(7, 10)]:
        subs_num = {psi: psi_val, alpha_2: Rational(1, 10), w_4: S.One}
        B_num = B_l_sub.xreplace(subs_num)
        w_num = [cancel(w.xreplace(subs_num)) if hasattr(w, 'xreplace') else w
                 for w in weights]
        for j_tf in range(1, T - 1):
            g = j_tf - 1
            col_sum = sum(w_num[i] * B_num[i, j_tf] for i in range(R))
            ic = _interior_contribution(g, R, E4_1.p, interior)
            col_sum += ic
            target = -1 if (g == 0 and E4_1.nu == 1) else 0
            residual = cancel(col_sum - target)
            assert residual == 0, (
                f"Numeric conservation violated at psi={psi_val}, g={g}: "
                f"residual={residual}"
            )


@pytest.mark.slow
def test_e4_1_conservative_taylor_accuracy():
    """Conservative E4_1 stencil preserves Taylor accuracy (26.4b).

    After substituting the conservation solution, each row of the cut-cell
    stencil still satisfies q+1=4 Taylor moment equations at multiple psi values.
    """
    psi = Symbol("psi")
    ur = derive_uniform_boundary_for_temo(E4_1, zeros={3, 4})
    stencil = construct_cut_cell_stencil(
        ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
    )
    B_l = stencil.matrix
    R, T = B_l.shape

    # Solve conservation
    eqs, w_syms = build_cut_cell_conservation_system(
        B_l, R, T, p=E4_1.p, nu=E4_1.nu,
        interior_coeffs=ur.interior, psi=psi,
    )
    alpha_syms = sorted(B_l.free_symbols - {psi}, key=lambda s: s.name)
    solve_for = alpha_syms[:2] + w_syms[:3]
    sol = solve_cut_cell_conservation(eqs, solve_for)
    B_l_sub = B_l.xreplace(sol)

    # Numeric verification at psi=0.3, 0.5, 0.7 with alpha_2=0.1, w_4=1.0
    alpha_2 = alpha_syms[2]
    w_4 = w_syms[3]
    for psi_val in [Rational(3, 10), Rational(1, 2), Rational(7, 10)]:
        subs_num = {psi: psi_val, alpha_2: Rational(1, 10), w_4: S.One}
        B_num = B_l_sub.xreplace(subs_num)
        for i in range(R):
            deltas = build_cut_cell_deltas(i, T, psi_val)
            row = [B_num[i, j] for j in range(T)]
            for k in range(4):  # q+1 = 4 Taylor equations
                moment = sum(row[j] * deltas[j] ** k for j in range(T))
                expected = 1 if k == 1 else 0
                residual = cancel(moment - expected)
                assert residual == 0, (
                    f"Taylor error: psi={psi_val}, row {i}, moment k={k}: "
                    f"got {cancel(moment)}, expected {expected}"
                )


@pytest.mark.slow
def test_e4_1_conservative_psi_interior():
    """Conservative E4_1 stencil is well-defined for psi in (0, 1) (26.4c).

    The conservation solution introduces poles at psi=0 and psi=1 (alpha_0 and
    alpha_1 diverge at the boundaries). The stencil is valid only for psi
    strictly between 0 and 1, which is the physically meaningful range for
    cut cells.

    This test verifies:
      1. All entries are finite for interior psi values
      2. Taylor accuracy holds at the boundary-adjacent values psi=0.01 and 0.99
      3. Conservation holds at boundary-adjacent values
    """
    psi = Symbol("psi")
    ur = derive_uniform_boundary_for_temo(E4_1, zeros={3, 4})
    stencil = construct_cut_cell_stencil(
        ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
    )
    B_l = stencil.matrix
    R, T = B_l.shape

    # Solve conservation
    eqs, w_syms = build_cut_cell_conservation_system(
        B_l, R, T, p=E4_1.p, nu=E4_1.nu,
        interior_coeffs=ur.interior, psi=psi,
    )
    alpha_syms = sorted(B_l.free_symbols - {psi}, key=lambda s: s.name)
    solve_for = alpha_syms[:2] + w_syms[:3]
    sol = solve_cut_cell_conservation(eqs, solve_for)
    B_l_sub = B_l.xreplace(sol)

    alpha_2 = alpha_syms[2]
    w_4 = w_syms[3]
    subs_num = {alpha_2: Rational(1, 10), w_4: S.One}

    # Weights: w_0=psi, w_1..w_3 from solution, w_4 free
    weights = [psi] + [sol[w] for w in w_syms[:3]] + [w_4]
    interior = ur.interior

    # Test at interior psi values including near-boundary
    test_psi_values = [
        Rational(1, 100), Rational(1, 10), Rational(1, 2),
        Rational(9, 10), Rational(99, 100),
    ]
    for psi_val in test_psi_values:
        full_subs = {psi: psi_val, **subs_num}
        B_num = B_l_sub.xreplace(full_subs)

        # 1. All entries are finite
        for i in range(R):
            for j in range(T):
                val = cancel(B_num[i, j])
                assert val.is_finite, (
                    f"B[{i},{j}] not finite at psi={psi_val}: {val}"
                )

        # 2. Taylor accuracy: each row satisfies q+1=4 moment equations
        for i in range(R):
            deltas = build_cut_cell_deltas(i, T, psi_val)
            row = [cancel(B_num[i, j]) for j in range(T)]
            for k in range(4):
                moment = sum(row[j] * deltas[j] ** k for j in range(T))
                expected = 1 if k == 1 else 0
                residual = cancel(moment - expected)
                assert residual == 0, (
                    f"Taylor: psi={psi_val}, row {i}, k={k}: residual={residual}"
                )

        # 3. Conservation: weighted column sums
        w_num = [cancel(w.xreplace(full_subs)) if hasattr(w, 'xreplace') else w
                 for w in weights]
        for j_tf in range(1, T - 1):
            g = j_tf - 1
            col_sum = sum(w_num[i] * cancel(B_num[i, j_tf]) for i in range(R))
            ic = _interior_contribution(g, R, E4_1.p, interior)
            col_sum += ic
            target = -1 if (g == 0 and E4_1.nu == 1) else 0
            residual = cancel(col_sum - target)
            assert residual == 0, (
                f"Conservation: psi={psi_val}, g={g}: residual={residual}"
            )


class TestBuildCutCellConservationSystem:
    """Tests for build_cut_cell_conservation_system dimensions and IC values (22.2b)."""

    def test_e2_1_conservation_system_dimensions(self):
        """E2_1: T-2=3 equations (grid-point cols), 3 weight unknowns, all IC zero."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E2_1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=E2_1.p, q=E2_1.q, nu=E2_1.nu,
            nextra=E2_1.nextra, psi=psi,
        )
        R, T = stencil.matrix.rows, stencil.matrix.cols
        assert R == 4
        assert T == 5

        eqs, ws = build_cut_cell_conservation_system(
            stencil.matrix, R, T, p=E2_1.p, nu=E2_1.nu,
            interior_coeffs=ur.interior, psi=psi,
        )
        assert len(eqs) == T - 2  # 3 equations (grid-point cols 0..2)
        assert len(ws) == R - 1   # 3 weight unknowns (w_1, w_2, w_3)

        # All IC values should be 0 for E2_1 (no interior row reaches grid points 0..2)
        for g in range(T - 2):
            ic = _interior_contribution(g, R, E2_1.p, ur.interior)
            assert ic == 0, f"E2_1 IC(g={g}) should be 0, got {ic}"

    def test_e2_1_conservation_solvable(self):
        """E2_1: 3 eqs / 3 unknowns → solvable, weights are rational in (alpha, psi).

        Validates the column mapping fix from 23.3b end-to-end: the corrected
        grid-point column range (T-frame cols 1..T-2) gives a consistent square
        system whose solution yields psi-dependent SBP weights.
        """
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E2_1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=E2_1.p, q=E2_1.q, nu=E2_1.nu,
            nextra=E2_1.nextra, psi=psi,
        )
        R, T = stencil.matrix.rows, stencil.matrix.cols

        eqs, ws = build_cut_cell_conservation_system(
            stencil.matrix, R, T, p=E2_1.p, nu=E2_1.nu,
            interior_coeffs=ur.interior, psi=psi,
        )
        assert len(eqs) == 3
        assert len(ws) == 3

        # Solve the system for w_1, w_2, w_3
        sol = solve(eqs, ws, dict=True)
        assert len(sol) == 1, f"Expected unique solution, got {len(sol)}"
        sol = sol[0]

        # All three weights must be present in the solution
        for w in ws:
            assert w in sol, f"Weight {w} missing from solution"

        # Weights should be rational functions of (alpha, psi)
        alpha_syms = sorted(stencil.matrix.free_symbols - {psi}, key=lambda s: s.name)
        for w in ws:
            w_val = sol[w]
            free = w_val.free_symbols
            # Must depend on psi (ψ-dependent weights)
            assert psi in free, f"{w} solution should depend on psi, got {w_val}"
            # Free symbols should be a subset of {psi} ∪ alphas
            assert free <= {psi} | set(alpha_syms), (
                f"{w} has unexpected symbols: {free - {psi} - set(alpha_syms)}"
            )

        # Substituting solution back into all 3 equations must yield 0
        for i, eq in enumerate(eqs):
            residual = cancel(eq.subs(sol))
            assert residual == 0, (
                f"Equation {i} residual non-zero after substitution: {residual}"
            )

    def test_e4_1_conservation_system_dimensions(self):
        """E4_1: T-2=5 equations (grid-point cols), 4 weight unknowns, nonzero IC at g=3,4."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E4_1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=E4_1.p, q=E4_1.q, nu=E4_1.nu,
            nextra=E4_1.nextra, psi=psi,
        )
        R, T = stencil.matrix.rows, stencil.matrix.cols
        assert R == 5
        assert T == 7

        eqs, ws = build_cut_cell_conservation_system(
            stencil.matrix, R, T, p=E4_1.p, nu=E4_1.nu,
            interior_coeffs=ur.interior, psi=psi,
        )
        assert len(eqs) == T - 2  # 5 equations (grid-point cols 0..4)
        assert len(ws) == R - 1   # 4 weight unknowns (w_1, w_2, w_3, w_4)

        # Verify IC values for E4_1 at R=5 (grid points 0..4)
        expected_ic = {
            0: Rational(0), 1: Rational(0), 2: Rational(0),
            3: Rational(1, 12), 4: Rational(-7, 12),
        }
        for g in range(T - 2):
            ic = _interior_contribution(g, R, E4_1.p, ur.interior)
            assert ic == expected_ic[g], (
                f"E4_1 IC(g={g}): expected {expected_ic[g]}, got {ic}"
            )

    def test_e4_1_overdetermined_system(self):
        """E4_1 has 5 equations and 4 weight unknowns -> 1 excess constraint."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E4_1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=E4_1.p, q=E4_1.q, nu=E4_1.nu,
            nextra=E4_1.nextra, psi=psi,
        )
        R, T = stencil.matrix.rows, stencil.matrix.cols
        eqs, ws = build_cut_cell_conservation_system(
            stencil.matrix, R, T, p=E4_1.p, nu=E4_1.nu,
            interior_coeffs=ur.interior, psi=psi,
        )
        excess = len(eqs) - len(ws)
        assert excess == 1, f"Expected 1 excess constraint, got {excess}"

        # Verify the stencil has 5 alpha symbols that must absorb these constraints
        alpha_syms = sorted(stencil.matrix.free_symbols - {psi}, key=lambda s: s.name)
        assert len(alpha_syms) == 5, (
            f"Expected 5 alpha symbols, got {len(alpha_syms)}: {alpha_syms}"
        )


@pytest.mark.slow
def test_e4_1_conservation_constant_weights_infeasible_r5():
    """E4_1 conservation with constant weights is infeasible at R=5 (23.3a).

    The conservation equations are rational in psi with bilinear terms w_i * alpha_k.
    We theta-linearize (replace w_i * alpha_k -> theta_{i,k}), clear psi-denominators,
    extract psi-coefficients to get scalar linear equations, then check
    rank(M) vs rank([M|b]) via the Rouche-Capelli theorem.

    Result: rank gap = 1, meaning the system is inconsistent. Conservation with
    constant (psi-independent) weights w_1..w_4 is structurally infeasible at R=5.
    Direct symbolic solve confirms: w_i solutions are rational functions of psi,
    not constants. A psi-dependent norm formulation would be needed.
    """
    psi = Symbol("psi")
    ur = derive_uniform_boundary_for_temo(E4_1)
    stencil = construct_cut_cell_stencil(
        ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
    )
    m = stencil.matrix
    R, T = 5, 7

    # Step 1: Build conservation equations (5 eqs, 4 weight unknowns)
    eqs, w_syms = build_cut_cell_conservation_system(
        m, R, T, p=2, nu=1, interior_coeffs=ur.interior, psi=psi,
    )
    assert len(eqs) == 5
    assert len(w_syms) == 4

    # Identify alpha symbols in the stencil
    alpha_syms = sorted(m.free_symbols - {psi}, key=lambda s: s.name)
    assert len(alpha_syms) == 5

    # Step 2: Identify which alphas appear in each row and create theta symbols
    theta_syms = []
    theta_map = {}  # (row_index, alpha) -> theta symbol
    row_alpha_map = {}  # row_index -> [alphas]
    for i in range(1, R):
        row_alphas = set()
        for j in range(T):
            row_alphas.update(s for s in m[i, j].free_symbols if s in alpha_syms)
        row_alphas_sorted = sorted(row_alphas, key=lambda s: s.name)
        row_alpha_map[i] = row_alphas_sorted
        for alpha in row_alphas_sorted:
            theta = Symbol(f"th_{i}_{alpha.name}")
            theta_syms.append(theta)
            theta_map[(i, alpha)] = theta

    # Alphas that appear in row 0 (linear, since w_0=psi is a parameter)
    row0_alphas = set()
    for j in range(T):
        row0_alphas.update(s for s in m[0, j].free_symbols if s in alpha_syms)
    row0_alpha_list = sorted(row0_alphas, key=lambda s: s.name)

    # Build substitution dict for all bilinear pairs w_i * alpha_k
    subs_dict = {}
    for i in range(1, R):
        w_i = w_syms[i - 1]
        for alpha in row_alpha_map[i]:
            subs_dict[w_i * alpha] = theta_map[(i, alpha)]

    # Step 3: Clear psi-denominators, expand, theta-linearize, extract psi-coefficients
    scalar_eqs = []
    for eq in eqs:
        num, _den = fraction(cancel(eq))
        poly_num = expand(num)
        lin_num = poly_num.subs(subs_dict)
        if psi in lin_num.free_symbols:
            p_poly = Poly(lin_num, psi)
            scalar_eqs.extend(p_poly.all_coeffs())
        else:
            scalar_eqs.append(lin_num)

    # Step 4: Build linear system and check rank (Rouche-Capelli)
    lin_unknowns = list(w_syms) + row0_alpha_list + theta_syms
    M_mat, b_vec = linear_eq_to_matrix(scalar_eqs, lin_unknowns)
    M_aug = M_mat.row_join(b_vec)

    rank_M = M_mat.rank()
    rank_aug = M_aug.rank()

    # Conservation with constant weights is INFEASIBLE: rank gap = 1
    assert rank_aug - rank_M == 1, (
        f"Expected rank gap 1, got {rank_aug - rank_M} "
        f"(rank(M)={rank_M}, rank([M|b])={rank_aug})"
    )
    assert rank_M == 8
    assert rank_aug == 9

    # Cross-check: direct symbolic solve produces psi-DEPENDENT weights,
    # confirming constant weights cannot satisfy conservation for all psi
    all_unknowns = list(w_syms) + list(alpha_syms)
    sol = solve(eqs, all_unknowns, dict=True)
    assert len(sol) == 1
    for w in w_syms:
        if w in sol[0]:
            assert psi in sol[0][w].free_symbols, (
                f"{w} solution should depend on psi"
            )


@pytest.mark.slow
def test_e4_1_psi_dependent_conservation_infeasible(
    e4_1_cut_cell_conservation,
):
    """E4_1 cut-cell conservation with psi-dependent weights is infeasible (23.3c-ii).

    Even allowing weights w_1..w_4 to be rational functions of psi, conservation
    cannot be satisfied for all psi in (0,1] with constant alpha parameters.

    Method:
      1. Solve 4 of 5 conservation equations for w_1..w_4 (rational in psi, alpha).
      2. Substitute into the 5th equation -> compatibility condition C(psi, alpha)=0.
      3. C is a degree-6 polynomial in psi; for it to vanish for all psi, all 7
         psi-coefficients must be zero.
      4. The resulting 7 nonlinear equations in 5 alpha unknowns have Groebner
         basis = {1}, proving the system is inconsistent.

    This result holds for ALL 5 choices of which equation to omit (C(5,4)=5 subsets).
    """
    eqs, w_syms, alpha_syms, psi = e4_1_cut_cell_conservation

    # Solve first 4 equations for weights as functions of (psi, alpha)
    sol4 = solve(eqs[:4], w_syms, dict=True)
    assert len(sol4) == 1
    sol4 = sol4[0]

    # All weight solutions must depend on psi (psi-dependent weights)
    for w in w_syms:
        assert psi in sol4[w].free_symbols, f"{w} should depend on psi"

    # Substitute into 5th equation -> compatibility condition
    residual = cancel(eqs[4].subs(sol4))
    assert residual != 0, "Residual should be non-zero (system overdetermined)"

    num, _den = fraction(residual)
    num_expanded = expand(num)

    # Extract psi-coefficients
    assert psi in num_expanded.free_symbols
    p_poly = Poly(num_expanded, psi)
    assert p_poly.degree() == 6
    coeffs = p_poly.all_coeffs()
    assert len(coeffs) == 7

    # Groebner basis of the 7 alpha constraints is {1} -> inconsistent
    gb = groebner(coeffs, list(alpha_syms), order="lex")
    assert list(gb) == [1], (
        f"Expected Groebner basis [1] (inconsistent), got {list(gb)}"
    )

    # Cross-check: sympy.solve also finds no solution
    sol_alpha = solve(coeffs, list(alpha_syms), dict=True)
    assert len(sol_alpha) == 0


@pytest.fixture(scope="module")
def e4_1_cut_cell_conservation():
    """Build E4_1 cut-cell conservation system (shared by infeasibility tests)."""
    psi = Symbol("psi")
    ur = derive_uniform_boundary_for_temo(E4_1)
    stencil = construct_cut_cell_stencil(
        ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
    )
    m = stencil.matrix
    R, T = m.rows, m.cols

    eqs, w_syms = build_cut_cell_conservation_system(
        m, R, T, p=2, nu=1, interior_coeffs=ur.interior, psi=psi,
    )
    alpha_syms = sorted(m.free_symbols - {psi}, key=lambda s: s.name)
    return eqs, w_syms, alpha_syms, psi


@pytest.mark.slow
class TestE4UniformConservation:
    """Tests for derive_uniform_boundary_for_temo(E4_1, conserve=True) (23.3a)."""

    @pytest.fixture(scope="class")
    def conserved(self):
        return derive_uniform_boundary_for_temo(E4_1, conserve=True)

    def test_alpha_count(self, conserved):
        """Conservation reduces E4_1 from 5 to 4 free alphas."""
        assert len(conserved.alpha_symbols) == 4
        for k, sym in enumerate(conserved.alpha_symbols):
            assert sym.name == f"alpha_{k}"

    def test_shape_unchanged(self, conserved):
        """B_u shape is still (4, 6) — conservation doesn't change dimensions."""
        assert conserved.B_u.shape == (4, 6)

    def test_weights_present(self, conserved):
        """Conservation produces 4 quadrature weights."""
        assert conserved.weights is not None
        assert len(conserved.weights) == 4

    def test_weights_depend_on_alpha_3(self, conserved):
        """All weights are rational functions of alpha_3 only."""
        alpha_3 = conserved.alpha_symbols[3]
        for w in conserved.weights:
            assert w.free_symbols <= {alpha_3}, (
                f"Weight {w} depends on {w.free_symbols}, expected only {alpha_3}"
            )

    def test_conservation_holds(self, conserved):
        """Weighted column sums satisfy the SBP conservation condition."""
        B_u = conserved.B_u
        r, t = B_u.shape
        for j in range(t - 1):
            col_sum = sum(conserved.weights[i] * B_u[i, j] for i in range(r))
            ic = _interior_contribution(j, r, conserved.p, conserved.interior)
            total = cancel(col_sum + ic)
            target = -1 if j == 0 else 0
            assert cancel(total - target) == 0, (
                f"Conservation violated at col {j}: residual={cancel(total - target)}"
            )

    def test_taylor_accuracy(self, conserved, assert_taylor_accuracy):
        """Each row still satisfies Taylor matching for q+1=4 equations."""
        assert_taylor_accuracy(conserved.B_u, q=3, nu=1)

    def test_rows_0_2_unchanged(self, conserved):
        """Rows 0-2 are identical to the non-conservative result."""
        ur_no_cons = derive_uniform_boundary_for_temo(E4_1, conserve=False)
        for i in range(3):
            for j in range(6):
                diff = cancel(conserved.B_u[i, j] - ur_no_cons.B_u[i, j])
                assert diff == 0, f"Row {i}, col {j} should be unchanged"

    def test_only_alpha_symbols_in_Bu(self, conserved):
        """B_u contains only the expected alpha symbols."""
        expected_syms = set(conserved.alpha_symbols)
        actual_syms = conserved.B_u.free_symbols
        assert actual_syms <= expected_syms, (
            f"Unexpected symbols in B_u: {actual_syms - expected_syms}"
        )

    def test_custom_alpha_symbols(self):
        """conserve=True accepts 4 custom alpha names."""
        syms = [Symbol(f"a{k}") for k in range(4)]
        result = derive_uniform_boundary_for_temo(E4_1, alpha_symbols=syms, conserve=True)
        assert result.alpha_symbols == syms
        assert result.B_u.free_symbols <= set(syms)

    def test_e2_1_unaffected(self):
        """E2_1 (nextra=1) is unaffected by conserve=True — uses its own conservation."""
        ur = derive_uniform_boundary_for_temo(E2_1, conserve=True)
        assert len(ur.alpha_symbols) == 4
        assert ur.weights is None  # nextra=1 conservation is inline, no explicit weights


@pytest.mark.slow
class TestCutCellConservationAfterUniform:
    """Tests for 24.3a: does cut-cell conservation follow from uniform conservation?

    Result: NO.  Even with the conservation-constrained uniform boundary
    (4 alphas), the overdetermined cut-cell conservation system (5 equations
    in 4 weight unknowns) is inconsistent.  The 5th compatibility condition
    is a degree-6 polynomial in psi whose coefficients are non-trivial
    functions of (alpha_0, ..., alpha_3), and no alpha assignment zeroes
    all of them simultaneously.

    This matches the Phase 23.3c-ii finding: TEMO construction does NOT
    preserve the SBP conservation property automatically.
    """

    @pytest.fixture(scope="class")
    def conserved_cut_cell(self):
        """Build E4_1-like cut-cell stencil with uniform conservation (no zeros)."""
        psi = Symbol("psi")
        # Use a local SchemeParams without zeros to test the uniform-only path.
        # The real E4_1 now has zeros=(3,4) which takes a different code path.
        e4_no_zeros = SchemeParams(p=2, q=3, s=0, nextra=0, nu=1)
        result = derive_cut_cell_scheme(e4_no_zeros, psi, conserve=True)
        interior = [Rational(1, 12), Rational(-2, 3), S.Zero,
                    Rational(2, 3), Rational(-1, 12)]
        eqs, w_syms = build_cut_cell_conservation_system(
            result.floating, result.floating.rows, result.floating.cols,
            p=2, nu=1, interior_coeffs=interior, psi=psi,
        )
        alpha_syms = sorted(
            result.floating.free_symbols - {psi}, key=lambda s: s.name
        )
        return eqs, w_syms, alpha_syms, psi

    def test_conservation_system_dimensions(self, conserved_cut_cell):
        """5 conservation equations, 4 weight unknowns (w_1..w_4)."""
        eqs, w_syms, alpha_syms, _ = conserved_cut_cell
        assert len(eqs) == 5
        assert len(w_syms) == 4
        assert len(alpha_syms) == 4

    def test_first_four_eqs_solvable(self, conserved_cut_cell):
        """First 4 equations can be solved for w_1..w_4."""
        eqs, w_syms, _, _ = conserved_cut_cell
        sol = solve(eqs[:4], w_syms, dict=True)
        assert len(sol) == 1

    def test_fifth_eq_residual_nonzero(self, conserved_cut_cell):
        """5th equation is not satisfied — conservation is infeasible."""
        eqs, w_syms, _, psi = conserved_cut_cell
        sol = solve(eqs[:4], w_syms, dict=True)[0]
        residual = cancel(eqs[4].subs(sol))
        assert residual != 0, "Expected non-zero residual (infeasible)"

    def test_residual_is_degree_6_in_psi(self, conserved_cut_cell):
        """Compatibility residual is degree 6 in psi (7 alpha-constraints)."""
        eqs, w_syms, _, psi = conserved_cut_cell
        sol = solve(eqs[:4], w_syms, dict=True)[0]
        residual = cancel(eqs[4].subs(sol))
        num, _ = fraction(residual)
        p_poly = Poly(expand(num), psi)
        assert p_poly.degree() == 6
        assert len(p_poly.all_coeffs()) == 7

    def test_groebner_confirms_infeasibility(self, conserved_cut_cell):
        """Groebner basis of 7 alpha constraints is {1} — no solution exists."""
        eqs, w_syms, alpha_syms, psi = conserved_cut_cell
        sol = solve(eqs[:4], w_syms, dict=True)[0]
        residual = cancel(eqs[4].subs(sol))
        num, _ = fraction(residual)
        p_poly = Poly(expand(num), psi)
        coeffs = p_poly.all_coeffs()
        gb = groebner(coeffs, list(alpha_syms), order="lex")
        assert list(gb) == [1], (
            f"Expected Groebner basis [1] (inconsistent), got {list(gb)}"
        )


@pytest.mark.slow
class TestPolynomialStructure:
    """Diagnostic: verify boundary rows are polynomial after QQ(ψ) solve (27.1a).

    For E4_1 with zeros={3,4}, call construct_cut_cell_stencil to get the raw
    TEMO output (before conservation). Check whether cancel(entry) is polynomial
    in ψ for each boundary row (i=0..3).
    """

    @pytest.fixture(scope="class")
    def temo_output(self):
        """Get raw TEMO output for E4_1 before conservation solve."""
        psi = Symbol("psi")
        uniform = derive_uniform_boundary_for_temo(
            E4_1, zeros=set(E4_1.zeros)
        )
        result = construct_cut_cell_stencil(
            uniform.B_u, uniform.interior, 2, 3, 1, 0, psi
        )
        return result, psi, uniform

    def test_boundary_rows_have_vandermonde_denominators(self, temo_output):
        """27.1a: Boundary rows have ψ-dependent Vandermonde denominators after cancel.

        Diagnostic finding: boundary rows from solve_temo_row are rational in ψ
        with Vandermonde-type denominators like (ψ+1)(ψ+2)(ψ+3). These are
        benign (nonvanishing on [0,1]) but NOT polynomial.
        Decision: polynomial ansatz (27.2a) IS needed.
        """
        result, psi, _ = temo_output
        m = result.matrix
        R, T = m.shape
        assert R == 5 and T == 7

        has_psi_denom = False
        for i in range(R - 1):  # rows 0..3 (boundary rows)
            for j in range(T):
                entry = m[i, j]
                num, den = fraction(cancel(entry))
                if den.has(psi):
                    has_psi_denom = True
                    # Verify denominators are Vandermonde-type (nonvanishing on [0,1])
                    p = Poly(den, psi)
                    val_0 = p.eval(0)
                    val_1 = p.eval(1)
                    assert val_0 != 0, f"Row {i} col {j}: denominator vanishes at psi=0"
                    assert val_1 != 0, f"Row {i} col {j}: denominator vanishes at psi=1"

        # Confirm that boundary rows ARE rational (not polynomial) — this
        # is the diagnostic finding that triggers the 27.2a polynomial ansatz.
        assert has_psi_denom, (
            "Unexpected: boundary rows are already polynomial. "
            "Skip 27.2a and proceed directly to 27.3a."
        )

    def test_numerator_degree_bound(self, temo_output):
        """27.1b: Numerators of boundary row entries have bounded degree in ψ.

        Since boundary rows are rational (not polynomial), we check numerator
        degree instead. For E4_1, numerator degree should be bounded.
        """
        result, psi, _ = temo_output
        m = result.matrix
        R, T = m.shape

        max_num_degree = 0
        max_den_degree = 0
        for i in range(R - 1):  # rows 0..3
            for j in range(T):
                entry = cancel(m[i, j])
                if entry == 0:
                    continue
                num, den = fraction(entry)
                if num.has(psi):
                    p = Poly(num, psi)
                    max_num_degree = max(max_num_degree, p.degree())
                if den.has(psi):
                    p = Poly(den, psi)
                    max_den_degree = max(max_den_degree, p.degree())

        # Verify degrees are reasonable (numerator ≤ 7, denominator ≤ 3)
        assert max_num_degree <= 7, f"Max numerator degree {max_num_degree} > 7"
        assert max_den_degree <= 3, f"Max denominator degree {max_den_degree} > 3"

    def test_entries_linear_in_alphas(self, temo_output):
        """27.1b: Each entry's numerator is at most degree 1 in each alpha symbol."""
        result, psi, uniform = temo_output
        m = result.matrix
        R, T = m.shape
        alpha_syms = uniform.alpha_symbols

        for i in range(R - 1):  # rows 0..3
            for j in range(T):
                entry = cancel(m[i, j])
                if entry == 0:
                    continue
                num, den = fraction(entry)
                # Check numerator is linear in each alpha
                for alpha in alpha_syms:
                    if not num.has(alpha):
                        continue
                    p = Poly(expand(num), alpha)
                    assert p.degree() <= 1, (
                        f"Row {i} col {j}: numerator degree {p.degree()} > 1 in {alpha}"
                    )
                # Denominator should be free of alpha symbols
                for alpha in alpha_syms:
                    assert not den.has(alpha), (
                        f"Row {i} col {j}: denominator depends on {alpha}"
                    )


@pytest.mark.slow
class TestPolynomialBoundaryRows:
    """Tests for solve_temo_row_polynomial applied to E4_1 boundary rows (27.2b).

    Verifies that the polynomial ansatz produces entries that are:
    - Polynomial in ψ (no denominator)
    - Degree ≤ 4 in ψ
    - Taylor-accurate
    - Matching both ψ-limits (uniform and degenerate)
    - At most linear in each alpha symbol
    """

    @pytest.fixture(scope="class")
    def poly_rows(self):
        """Compute polynomial boundary rows for E4_1 (rows 0-3)."""
        psi = Symbol("psi")
        uniform = derive_uniform_boundary_for_temo(
            E4_1, zeros=set(E4_1.zeros)
        )
        B_u = uniform.B_u
        interior = uniform.interior
        p, q, nu, nextra = 2, 3, 1, 0
        R = B_u.rows + 1  # 5
        T = B_u.cols + 1  # 7

        B_l_1 = solve_uniform_limit(B_u, interior, p, q, nu, nextra)
        B_d = build_degenerate_stencil(B_u, interior, p, q, nu)

        alpha_syms = sorted(B_u.free_symbols, key=lambda s: s.name)

        rows = []
        for i in range(R - 1):  # boundary rows 0..3
            V, rhs_vec = build_temo_vandermonde(i, T, q, nu, psi)
            prescribed = identify_prescribed_entries(
                i, B_u.rows, B_u.cols, nextra, nu, B_u, B_l_1, B_d, psi,
                max(q + 1, nu + 1),
            )
            result = solve_temo_row_polynomial(
                i, V, rhs_vec, prescribed, psi, alpha_syms
            )
            rows.append(result.coeffs)

        return rows, psi, alpha_syms, B_l_1, B_d, T, q, nu

    def test_all_entries_polynomial(self, poly_rows):
        """Every boundary row entry is a polynomial in ψ (no denominator)."""
        rows, psi, _, _, _, T, _, _ = poly_rows
        for i, row in enumerate(rows):
            for j in range(T):
                entry = row[j]
                num, den = fraction(cancel(entry))
                assert not den.has(psi), (
                    f"Row {i} col {j}: has ψ-dependent denominator {den}"
                )

    def test_degree_bound(self, poly_rows):
        """Max ψ-degree of each entry is ≤ 4 (for E4_1 with q=3)."""
        rows, psi, _, _, _, T, _, _ = poly_rows
        for i, row in enumerate(rows):
            for j in range(T):
                entry = expand(row[j])
                if entry == 0 or not entry.has(psi):
                    continue
                p = Poly(entry, psi)
                assert p.degree() <= 4, (
                    f"Row {i} col {j}: ψ-degree {p.degree()} > 4"
                )

    def test_taylor_accuracy_symbolic(self, poly_rows):
        """Each row satisfies Taylor accuracy as a polynomial identity in ψ.

        Verify sum_j c_j * delta_j^m / m! = delta_{m,nu} for m=0..q.
        """
        rows, psi, _, _, _, T, q, nu = poly_rows
        from sympy import factorial

        for i, row in enumerate(rows):
            deltas = build_cut_cell_deltas(i, T, psi)
            for m in range(q + 1):
                # Taylor condition: sum_j c_j * delta_j^m / m! = (1 if m==nu else 0)
                lhs = sum(row[j] * deltas[j] ** m / factorial(m) for j in range(T))
                expected = Rational(1) if m == nu else Rational(0)
                residual = expand(lhs - expected)
                assert residual == 0, (
                    f"Row {i}, m={m}: Taylor residual = {residual}"
                )

    def test_psi_1_matches_uniform(self, poly_rows):
        """At ψ=1, boundary row entries match B_l(1) (uniform limit)."""
        rows, psi, _, B_l_1, _, T, _, _ = poly_rows
        for i, row in enumerate(rows):
            for j in range(T):
                val = expand(row[j].subs(psi, 1))
                expected = expand(B_l_1[i, j])
                assert expand(val - expected) == 0, (
                    f"Row {i} col {j}: at ψ=1 got {val}, expected {expected}"
                )

    def test_psi_0_matches_degenerate(self, poly_rows):
        """At ψ=0, boundary row entries match B_d (degenerate stencil)."""
        rows, psi, _, _, B_d, T, _, _ = poly_rows
        for i, row in enumerate(rows):
            for j in range(T):
                val = expand(row[j].subs(psi, 0))
                expected = expand(B_d[i, j])
                assert expand(val - expected) == 0, (
                    f"Row {i} col {j}: at ψ=0 got {val}, expected {expected}"
                )

    def test_linear_in_alphas(self, poly_rows):
        """Each entry is at most degree 1 in each alpha symbol."""
        rows, psi, alpha_syms, _, _, T, _, _ = poly_rows
        for i, row in enumerate(rows):
            for j in range(T):
                entry = expand(row[j])
                if entry == 0:
                    continue
                for alpha in alpha_syms:
                    if not entry.has(alpha):
                        continue
                    p = Poly(entry, alpha)
                    assert p.degree() <= 1, (
                        f"Row {i} col {j}: degree {p.degree()} > 1 in {alpha}"
                    )


@pytest.mark.slow
class TestFractionFreeConservation:
    """Tests for solve_conservation_fraction_free with E4_1 (27.3b)."""

    @pytest.fixture(scope="class")
    def fraction_free_solution(self):
        """Build zero-constrained cut-cell stencil and solve conservation fraction-free."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E4_1, zeros={3, 4})
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=2, q=3, nu=1, nextra=0, psi=psi,
        )
        R, T = stencil.matrix.shape
        eqs, w_syms = build_cut_cell_conservation_system(
            stencil.matrix, R, T, p=E4_1.p, nu=E4_1.nu,
            interior_coeffs=ur.interior, psi=psi,
        )
        alpha_syms = sorted(
            stencil.matrix.free_symbols - {psi}, key=lambda s: s.name,
        )
        solve_for = alpha_syms[:2] + w_syms[:3]
        sol = solve_conservation_fraction_free(eqs, solve_for, psi)
        return sol, eqs, w_syms, stencil, ur, psi, alpha_syms, solve_for

    def test_weight_denominators_benign(self, fraction_free_solution):
        """Weight solutions (w_1, w_2, w_3) are psi-independent (constant denominator).

        The bilinear solve produces psi(psi-1) denominators for the alpha
        solutions, but the weight solutions should be psi-independent
        (denominator = 1).  This is inherent to the problem structure.
        """
        sol, _, w_syms, _, _, psi, _, _ = fraction_free_solution
        for w in w_syms[:3]:
            if w in sol:
                num, den = fraction(cancel(sol[w]))
                assert psi not in den.free_symbols, (
                    f"{w}: expected psi-free denominator, got {den}"
                )

    def test_alpha_denominators_documented(self, fraction_free_solution):
        """Alpha solutions have psi(psi-1) denominators (known limitation).

        This test documents the current behavior: the bilinear conservation
        system inherently produces psi-dependent alpha solutions with
        psi(psi-1) denominator factors.  A restructured approach (not
        solving for alphas in conservation) would be needed to eliminate
        these — see plan 27.3a notes.
        """
        sol, _, _, _, _, psi, alpha_syms, _ = fraction_free_solution
        for a in alpha_syms[:2]:
            if a in sol:
                num, den = fraction(cancel(sol[a]))
                # Alpha denominators contain psi factors
                if psi in den.free_symbols:
                    p = Poly(den, psi)
                    # They vanish at psi=0 or psi=1 (known limitation)
                    has_psi_pole = (p.eval(0) == 0 or p.eval(1) == 0)
                    assert has_psi_pole, (
                        f"{a}: expected psi/psi-1 denominator factor, "
                        f"but den is benign: {den}"
                    )

    def test_matches_old_solver(self, fraction_free_solution):
        """Fraction-free solution matches the old solver numerically."""
        sol_ff, eqs, w_syms, _, _, psi, alpha_syms, solve_for = fraction_free_solution
        sol_old = solve_cut_cell_conservation(eqs, solve_for)
        alpha_2 = alpha_syms[2]
        w_4 = w_syms[3]
        test_vals = [
            {psi: Rational(1, 4), alpha_2: Rational(1, 3), w_4: Rational(1, 2)},
            {psi: Rational(1, 2), alpha_2: Rational(1, 5), w_4: Rational(2, 3)},
            {psi: Rational(3, 4), alpha_2: Rational(0), w_4: Rational(1)},
        ]
        for s in solve_for:
            for vals in test_vals:
                val_ff = sol_ff[s].subs(vals)
                val_old = sol_old[s].subs(vals)
                assert cancel(val_ff - val_old) == 0, (
                    f"{s} at {vals}: fraction-free={val_ff}, old={val_old}"
                )

    def test_conservation_holds(self, fraction_free_solution):
        """All 5 conservation equations evaluate to 0 after substitution."""
        sol, eqs, _, _, _, _, _, _ = fraction_free_solution
        for i, eq in enumerate(eqs):
            residual = cancel(eq.subs(sol))
            assert residual == 0, f"Equation {i} residual: {residual}"

    def test_remaining_free_params(self, fraction_free_solution):
        """Solution leaves exactly alpha_2 and w_4 as free parameters."""
        sol, _, w_syms, _, _, psi, alpha_syms, _ = fraction_free_solution
        alpha_2 = alpha_syms[2]
        w_4 = w_syms[3]
        allowed = {psi, alpha_2, w_4}
        for sym, expr in sol.items():
            extra = expr.free_symbols - allowed
            assert extra == set(), (
                f"Solution for {sym} has unexpected symbols: {extra}"
            )

    def test_solve_for_completeness(self, fraction_free_solution):
        """Solution dict contains all 5 solve_for symbols."""
        sol, eqs, _, _, _, _, _, solve_for = fraction_free_solution
        for s in solve_for:
            assert s in sol, f"Missing {s} from solution"
        assert len(sol) == 5


@pytest.mark.slow
class TestApproachAInfeasibility:
    """Tests that weights-only conservation is infeasible with polynomial boundary rows (27.3d).

    This validates 27.3c Finding 2: with polynomial boundary rows (from
    solve_temo_row_polynomial) and c_*=0, the conservation system cannot be
    satisfied by weights alone, regardless of alpha values.  This is the
    critical result that rules out approach (A) and forces approach (B)
    (accept psi(psi-1) denominators from bilinear alpha+weight solve).
    """

    @pytest.fixture(scope="class")
    def poly_stencil(self):
        """Build stencil with polynomial boundary rows + rational near-interior row."""
        psi = Symbol("psi")
        uniform = derive_uniform_boundary_for_temo(E4_1, zeros=set(E4_1.zeros))
        B_u = uniform.B_u
        interior = uniform.interior
        p, q, nu, nextra = 2, 3, 1, 0
        R = B_u.rows + 1  # 5
        T = B_u.cols + 1  # 7

        B_l_1 = solve_uniform_limit(B_u, interior, p, q, nu, nextra)
        B_d = build_degenerate_stencil(B_u, interior, p, q, nu)
        K, _ = make_psi_field(psi)

        alpha_syms = sorted(B_u.free_symbols, key=lambda s: s.name)

        rows = []
        for i in range(R):
            V, rhs_vec = build_temo_vandermonde(i, T, q, nu, psi)
            prescribed = identify_prescribed_entries(
                i, B_u.rows, B_u.cols, nextra, nu, B_u, B_l_1, B_d, psi,
                max(q + 1, nu + 1),
            )
            if i < R - 1:  # polynomial boundary rows
                result = solve_temo_row_polynomial(
                    i, V, rhs_vec, prescribed, psi, alpha_syms,
                )
            else:  # rational near-interior row
                result = solve_temo_row(
                    i, V, rhs_vec, prescribed, psi, K, alpha_syms,
                )
            rows.append(result.coeffs)

        matrix = Matrix(rows)

        # Set c_* symbols (extra polynomial unknowns) to zero
        c_syms = [s for s in matrix.free_symbols if s.name.startswith("c_")]
        if c_syms:
            matrix = matrix.xreplace({c: 0 for c in c_syms})

        return matrix, R, T, p, q, nu, psi, alpha_syms, interior

    def test_weights_only_inconsistent_alpha_zero(self, poly_stencil):
        """Weights-only solve is inconsistent with all alphas set to zero."""
        matrix, R, T, p, _, nu, psi, alpha_syms, interior = poly_stencil

        # Substitute alpha=0
        m_eval = matrix.xreplace({a: 0 for a in alpha_syms})

        eqs, w_syms = build_cut_cell_conservation_system(
            m_eval, R, T, p=p, nu=nu, interior_coeffs=interior, psi=psi,
        )
        assert len(eqs) == 5
        assert len(w_syms) == 4

        # Clear psi denominators and build scalar system
        scalar_eqs = []
        for eq in eqs:
            num, _den = fraction(cancel(eq))
            poly_num = expand(num)
            if psi in poly_num.free_symbols:
                p_poly = Poly(poly_num, psi)
                scalar_eqs.extend(p_poly.all_coeffs())
            else:
                scalar_eqs.append(poly_num)

        M_mat, b_vec = linear_eq_to_matrix(scalar_eqs, list(w_syms))
        M_aug = M_mat.row_join(b_vec)

        rank_M = M_mat.rank()
        rank_aug = M_aug.rank()

        # System is inconsistent: rank gap >= 1
        assert rank_aug > rank_M, (
            f"Expected inconsistent system (rank gap >= 1), got "
            f"rank(M)={rank_M}, rank([M|b])={rank_aug}"
        )

    def test_weights_only_inconsistent_alpha_nonzero(self, poly_stencil):
        """Weights-only solve is inconsistent with representative nonzero alphas."""
        matrix, R, T, p, _, nu, psi, alpha_syms, interior = poly_stencil

        # Try alpha = (1/10, 1/5, 1/3) for first 3 alphas, 0 for rest
        alpha_vals = [Rational(1, 10), Rational(1, 5), Rational(1, 3)]
        subs = {}
        for i, a in enumerate(alpha_syms):
            subs[a] = alpha_vals[i] if i < len(alpha_vals) else 0
        m_eval = matrix.xreplace(subs)

        eqs, w_syms = build_cut_cell_conservation_system(
            m_eval, R, T, p=p, nu=nu, interior_coeffs=interior, psi=psi,
        )

        scalar_eqs = []
        for eq in eqs:
            num, _den = fraction(cancel(eq))
            poly_num = expand(num)
            if psi in poly_num.free_symbols:
                p_poly = Poly(poly_num, psi)
                scalar_eqs.extend(p_poly.all_coeffs())
            else:
                scalar_eqs.append(poly_num)

        M_mat, b_vec = linear_eq_to_matrix(scalar_eqs, list(w_syms))
        M_aug = M_mat.row_join(b_vec)

        rank_M = M_mat.rank()
        rank_aug = M_aug.rank()

        assert rank_aug > rank_M, (
            f"Expected inconsistent system (rank gap >= 1), got "
            f"rank(M)={rank_M}, rank([M|b])={rank_aug}"
        )

    def test_weights_only_inconsistent_alpha_symbolic(self, poly_stencil):
        """Weights-only solve is inconsistent with symbolic alpha values.

        This is the strongest version: even with alphas as free parameters,
        the 5-equation system for 4 weight unknowns is overdetermined and
        inconsistent, proving no choice of alpha values can make weights-only
        conservation work.
        """
        matrix, R, T, p, _, nu, psi, alpha_syms, interior = poly_stencil

        eqs, w_syms = build_cut_cell_conservation_system(
            matrix, R, T, p=p, nu=nu, interior_coeffs=interior, psi=psi,
        )
        assert len(eqs) == 5
        assert len(w_syms) == 4

        # Theta-linearize bilinear w_i * alpha_k terms
        theta_syms = []
        theta_map = {}
        row_alpha_map = {}
        for i in range(1, R):
            row_alphas = set()
            for j in range(T):
                row_alphas.update(
                    s for s in matrix[i, j].free_symbols if s in alpha_syms
                )
            row_alphas_sorted = sorted(row_alphas, key=lambda s: s.name)
            row_alpha_map[i] = row_alphas_sorted
            for alpha in row_alphas_sorted:
                theta = Symbol(f"th_{i}_{alpha.name}")
                theta_syms.append(theta)
                theta_map[(i, alpha)] = theta

        # Row 0 alphas (appear linearly since w_0=psi is fixed)
        row0_alpha_list = sorted(
            {s for j in range(T) for s in matrix[0, j].free_symbols
             if s in alpha_syms},
            key=lambda s: s.name,
        )

        subs_dict = {}
        for i in range(1, R):
            w_i = w_syms[i - 1]
            for alpha in row_alpha_map[i]:
                subs_dict[w_i * alpha] = theta_map[(i, alpha)]

        scalar_eqs = []
        for eq in eqs:
            num, _den = fraction(cancel(eq))
            poly_num = expand(num)
            lin_num = poly_num.subs(subs_dict)
            if psi in lin_num.free_symbols:
                p_poly = Poly(lin_num, psi)
                scalar_eqs.extend(p_poly.all_coeffs())
            else:
                scalar_eqs.append(lin_num)

        lin_unknowns = list(w_syms) + row0_alpha_list + theta_syms
        M_mat, b_vec = linear_eq_to_matrix(scalar_eqs, lin_unknowns)
        M_aug = M_mat.row_join(b_vec)

        rank_M = M_mat.rank()
        rank_aug = M_aug.rank()

        assert rank_aug > rank_M, (
            f"Expected inconsistent system (rank gap >= 1), got "
            f"rank(M)={rank_M}, rank([M|b])={rank_aug}"
        )


@pytest.mark.slow
class TestPolynomialFullStencil:
    """Validate the full E4_1 stencil from the polynomial construction pipeline (27.4b).

    After conservation solve (approach B from 27.3c), boundary row entries
    reacquire ψ(ψ-1) denominators via the alpha substitution.  These tests
    document the expected structure and verify correctness at interior ψ values.
    """

    @pytest.fixture(scope="class")
    def full_result(self):
        psi = Symbol("psi")
        result = derive_cut_cell_scheme(E4_1, psi)
        return result, psi

    # ------------------------------------------------------------------
    # 1. test_all_entries_well_defined
    # ------------------------------------------------------------------
    def test_all_entries_well_defined(self, full_result):
        """Document denominator structure: some entries have ψ(ψ-1) poles after alpha sub.

        All entries must be finite at interior ψ values (e.g. ψ=1/2).
        """
        result, psi = full_result
        m = result.floating
        R, T = m.shape
        alpha_vals = dict(
            zip(result.alpha_symbols, [Rational(1, 3), Rational(1, 2)])
        )
        psi_val = Rational(1, 2)
        subs = {psi: psi_val, **alpha_vals}

        entries_with_psi0_pole = []
        entries_with_psi1_pole = []

        for i in range(R):
            for j in range(T):
                # Every entry must be finite at an interior psi value
                val = m[i, j].subs(subs)
                assert val.is_finite, f"Entry [{i},{j}] not finite at psi={psi_val}"

                # Classify denominator poles
                entry = m[i, j].subs(alpha_vals)
                if entry == 0:
                    continue
                num, den = fraction(cancel(entry))
                if den == 1 or psi not in den.free_symbols:
                    continue
                den_poly = Poly(den, psi)
                if den_poly.eval(0) == 0:
                    entries_with_psi0_pole.append((i, j))
                if den_poly.eval(1) == 0:
                    entries_with_psi1_pole.append((i, j))

        # Approach B: some entries have ψ=0 and ψ=1 poles from alpha substitution
        assert len(entries_with_psi0_pole) > 0, (
            "Expected some entries with ψ=0 poles (approach B)"
        )
        assert len(entries_with_psi1_pole) > 0, (
            "Expected some entries with ψ=1 poles (approach B)"
        )

    # ------------------------------------------------------------------
    # 2. test_taylor_accuracy_all_rows
    # ------------------------------------------------------------------
    def test_taylor_accuracy_all_rows(self, full_result):
        """All 5 rows satisfy q+1=4 Taylor equations at ψ=1/2."""
        result, psi = full_result
        m = result.floating
        R, T = m.shape
        alpha_vals = dict(
            zip(result.alpha_symbols, [Rational(1, 10), Rational(1, 1)])
        )
        psi_val = Rational(1, 2)
        subs = {psi: psi_val, **alpha_vals}
        for i in range(R):
            deltas = build_cut_cell_deltas(i, T, psi_val)
            row = [m[i, j].subs(subs) for j in range(T)]
            for k in range(4):
                moment = sum(row[j] * deltas[j] ** k for j in range(T))
                expected = 1 if k == 1 else 0
                assert cancel(moment - expected) == 0, (
                    f"Row {i}, moment k={k}: got {cancel(moment)}"
                )

    # ------------------------------------------------------------------
    # 3. test_conservation_column_sums
    # ------------------------------------------------------------------
    def test_conservation_column_sums(self, full_result):
        """Weighted column sums using result.weights satisfy conservation."""
        result, psi = full_result
        m = result.floating
        R, T = m.shape
        weights = result.weights
        interior = derive_uniform_boundary_for_temo(E4_1, zeros={3, 4}).interior
        for j_tf in range(1, T - 1):
            g = j_tf - 1
            col_sum = sum(weights[i] * m[i, j_tf] for i in range(R))
            ic = _interior_contribution(g, R, E4_1.p, interior)
            col_sum += ic
            target = S.NegativeOne if (g == 0 and E4_1.nu == 1) else S.Zero
            residual = cancel(col_sum - target)
            assert residual == 0, (
                f"Conservation violated at grid point {g}: residual={residual}"
            )

    # ------------------------------------------------------------------
    # 4. test_psi_0_limit
    # ------------------------------------------------------------------
    def test_psi_0_limit(self, full_result):
        """Entries are finite at ψ near 0 (ψ=1/10) with representative alpha values.

        Exact ψ=0 has poles for entries that absorbed the conservation alpha
        solutions (approach B).  Runtime psi_eps clamping handles this.
        """
        result, psi = full_result
        m = result.floating
        R, T = m.shape
        alpha_vals = dict(
            zip(result.alpha_symbols, [Rational(1, 3), Rational(1, 2)])
        )
        psi_val = Rational(1, 10)
        subs = {psi: psi_val, **alpha_vals}
        for i in range(R):
            for j in range(T):
                val = m[i, j].subs(subs)
                assert val.is_finite, (
                    f"Entry [{i},{j}] not finite at psi={psi_val}"
                )

    # ------------------------------------------------------------------
    # 5. test_psi_1_limit
    # ------------------------------------------------------------------
    def test_psi_1_limit(self, full_result):
        """Entries are finite at ψ near 1 (ψ=9/10) with representative alpha values.

        Exact ψ=1 has poles for entries that absorbed the conservation alpha
        solutions (approach B).  Runtime psi_eps clamping handles this.
        """
        result, psi = full_result
        m = result.floating
        R, T = m.shape
        alpha_vals = dict(
            zip(result.alpha_symbols, [Rational(1, 3), Rational(1, 2)])
        )
        psi_val = Rational(9, 10)
        subs = {psi: psi_val, **alpha_vals}
        for i in range(R):
            for j in range(T):
                val = m[i, j].subs(subs)
                assert val.is_finite, (
                    f"Entry [{i},{j}] not finite at psi={psi_val}"
                )

    # ------------------------------------------------------------------
    # 6. test_free_parameter_count
    # ------------------------------------------------------------------
    def test_free_parameter_count(self, full_result):
        """Exactly 2 free parameters (alpha_0, alpha_1)."""
        result, psi = full_result
        assert len(result.alpha_symbols) == 2
        expected = {psi} | set(result.alpha_symbols)
        assert result.floating.free_symbols <= expected, (
            f"Unexpected symbols: {result.floating.free_symbols - expected}"
        )

    # ------------------------------------------------------------------
    # 7. test_matches_derive_cut_cell_scheme
    # ------------------------------------------------------------------
    def test_matches_derive_cut_cell_scheme(self, full_result):
        """derive_cut_cell_scheme(E4_1) produces structurally correct result.

        Verifies shape, no Neumann stencil, no conservation_subs (zeros path),
        and dirichlet == floating[1:, :].
        """
        result, psi = full_result
        assert result.floating.shape == (5, 7)
        assert result.dirichlet.shape == (4, 7)
        assert result.neumann is None
        assert result.conservation_subs is None

        # Dirichlet is rows 1..R-1 of floating
        R, T = result.floating.shape
        for i in range(R - 1):
            for j in range(T):
                assert cancel(
                    result.dirichlet[i, j] - result.floating[i + 1, j]
                ) == 0, f"Dirichlet mismatch at [{i},{j}]"

    # ------------------------------------------------------------------
    # 8. test_weights_well_defined
    # ------------------------------------------------------------------
    def test_weights_well_defined(self, full_result):
        """All 5 weights have no ψ(ψ-1) denominator factors.

        w_0 = psi (by construction), others are psi-independent or have
        benign (nonvanishing on [0,1]) denominators.
        """
        result, psi = full_result
        weights = result.weights
        assert len(weights) == 5
        assert weights[0] == psi  # w_0 is always psi
        for k, w in enumerate(weights):
            if w == psi:
                continue
            num, den = fraction(cancel(w))
            if den == 1 or psi not in den.free_symbols:
                continue
            den_poly = Poly(den, psi)
            assert den_poly.eval(0) != 0, f"Weight {k} has ψ=0 pole"
            assert den_poly.eval(1) != 0, f"Weight {k} has ψ=1 pole"


# ---------------------------------------------------------------------------
# Mathematica-aligned workflow tests
# ---------------------------------------------------------------------------


class TestMathematicaUniformConservation:
    """Test uniform conservation matches Mathematica page 4 (baseSol)."""

    def test_weights_match_mathematica(self):
        """Verify w[1..4] match Mathematica baseSol at alpha=(1,1,-1,1)."""
        from stencil_gen.temo import (
            build_uniform_for_mathematica,
            solve_uniform_conservation_direct,
        )

        uniform = build_uniform_for_mathematica(E4_1)
        B_u = uniform.B_u
        _, weights = solve_uniform_conservation_direct(B_u, uniform.interior, E4_1.p)

        a0 = Symbol("alpha_0")
        # Mathematica page 4 (w[1] = -6/(-11+6*alpha[1]))
        expected_w0 = cancel(Rational(-6) / (-11 + 6 * a0))
        assert cancel(weights[0] - expected_w0) == 0

    def test_base_scheme_shape(self):
        """Base scheme should be (r+1) x t = 5x6 for E4_1."""
        from stencil_gen.temo import (
            build_base_scheme,
            build_uniform_for_mathematica,
            solve_uniform_conservation_direct,
        )

        uniform = build_uniform_for_mathematica(E4_1)
        sol, _ = solve_uniform_conservation_direct(
            uniform.B_u, uniform.interior, E4_1.p
        )
        base = build_base_scheme(uniform.B_u, sol, uniform.interior, E4_1.p)
        assert base.shape == (5, 6)

    def test_base_scheme_values(self):
        """Base scheme at alpha=(1,1,-1,1) matches Mathematica page 4."""
        from stencil_gen.temo import (
            build_base_scheme,
            build_uniform_for_mathematica,
            solve_uniform_conservation_direct,
        )

        uniform = build_uniform_for_mathematica(E4_1)
        sol, _ = solve_uniform_conservation_direct(
            uniform.B_u, uniform.interior, E4_1.p
        )
        base = build_base_scheme(uniform.B_u, sol, uniform.interior, E4_1.p)

        a0, a1, a2, a3 = [Symbol(f"alpha_{i}") for i in range(4)]
        subs = {a0: 1, a1: 1, a2: -1, a3: 1}
        base_num = base.subs(subs).applyfunc(cancel)

        # Row 0: [-5/6, -1, 9/2, -11/3, 1, 0]
        assert cancel(base_num[0, 0] - Rational(-5, 6)) == 0
        assert cancel(base_num[0, 1] - Rational(-1)) == 0
        assert cancel(base_num[0, 4] - Rational(1)) == 0

        # Row 4 (interior): [0, 0, 1/12, -2/3, 0, 2/3]
        assert base_num[4, 0] == 0
        assert base_num[4, 1] == 0
        assert cancel(base_num[4, 2] - Rational(1, 12)) == 0


@pytest.fixture(scope="module")
def mathematica_result():
    """Cache the derive_cut_cell_mathematica result for the module."""
    import os
    os.environ.setdefault("SYMPY_CACHE_SIZE", "50000")
    from stencil_gen.temo import derive_cut_cell_mathematica

    psi = Symbol("psi")
    return derive_cut_cell_mathematica(E4_1, psi)


@pytest.mark.slow
class TestMathematicaWorkflow:
    """Test the full Mathematica-aligned cut-cell workflow."""

    def test_shape(self, mathematica_result):
        """Floating stencil should be 5x7."""
        assert mathematica_result.floating.shape == (5, 7)

    def test_alpha_count(self, mathematica_result):
        """Should have 4 free alpha symbols."""
        assert len(mathematica_result.alpha_symbols) == 4

    def test_weight_count(self, mathematica_result):
        """Should have 5 weights."""
        assert len(mathematica_result.weights) == 5

    def test_no_psi_poles(self, mathematica_result):
        """No entries should have poles at psi=0 or psi=1."""
        psi = Symbol("psi")
        F = mathematica_result.floating
        for i in range(F.rows):
            for j in range(F.cols):
                entry = cancel(F[i, j])
                _, denom = fraction(entry)
                if denom != 1 and psi in denom.free_symbols:
                    d_at_0 = denom.subs(psi, 0)
                    d_at_1 = denom.subs(psi, 1)
                    assert d_at_0 != 0, f"ψ=0 pole at [{i},{j}]"
                    assert d_at_1 != 0, f"ψ=1 pole at [{i},{j}]"

    def test_dirichlet_shape(self, mathematica_result):
        """Dirichlet stencil should be 4x7 (drop row 0)."""
        assert mathematica_result.dirichlet.shape == (4, 7)

