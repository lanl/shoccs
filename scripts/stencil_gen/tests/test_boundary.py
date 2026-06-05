"""Tests for boundary stencil derivation pipeline (20.3a–20.3g).

Test naming convention:
  - test_taylor_*       : 20.3a (Taylor system builder)
  - test_solve_row_*    : 20.3b (single-row boundary solver)
  - test_conservation_* : 20.3d (conservation constraint solver)
  - test_E4u_*          : 20.3e (E4u end-to-end validation)
  - test_E6u_*          : 20.3f (E6u end-to-end validation)
  - test_E8u_*          : 20.3g (E8u end-to-end validation)
"""

import pytest
from sympy import Matrix, Rational, S, Symbol, cancel, symbols

from stencil_gen.taylor_system import build_taylor_system
from stencil_gen.boundary import solve_boundary_row, BoundaryRow
from stencil_gen.conservation import _interior_contribution


# ---------------------------------------------------------------------------
# 20.3a -- Taylor system tests
# ---------------------------------------------------------------------------


def test_taylor_E4u_row0_shape():
    """V shape is (4, 5) for E4u_1 row 0 (i=0, t=5, q=3, nu=1)."""
    V, rhs = build_taylor_system(0, 5, 3, 1)
    assert V.shape == (4, 5)
    assert rhs.shape == (4, 1)


def test_taylor_E4u_row0_entries():
    """V entries match the worked example for E4u_1 row 0."""
    V, rhs = build_taylor_system(0, 5, 3, 1)

    # k=0 row: (j-0)^0 / 0! = 1 for all j
    for j in range(5):
        assert V[0, j] == 1

    # k=1 row: (j-0)^1 / 1! = j
    for j in range(5):
        assert V[1, j] == j

    # k=2 row: j^2 / 2
    assert V[2, 0] == 0
    assert V[2, 1] == Rational(1, 2)
    assert V[2, 2] == 2
    assert V[2, 3] == Rational(9, 2)
    assert V[2, 4] == 8

    # k=3 row: j^3 / 6
    assert V[3, 0] == 0
    assert V[3, 1] == Rational(1, 6)
    assert V[3, 2] == Rational(4, 3)
    assert V[3, 3] == Rational(9, 2)
    assert V[3, 4] == Rational(32, 3)


def test_taylor_E4u_row0_rhs():
    """rhs = [0, 1, 0, 0]^T for nu=1."""
    V, rhs = build_taylor_system(0, 5, 3, 1)
    assert rhs == Matrix([0, 1, 0, 0])


def test_taylor_E4u_row1_spot():
    """Spot-check V for E4u_1 row 1 (i=1)."""
    V, rhs = build_taylor_system(1, 5, 3, 1)
    assert V.shape == (4, 5)
    # k=1: (j - 1)^1 / 1!
    assert V[1, 0] == -1
    assert V[1, 1] == 0
    assert V[1, 2] == 1
    assert V[1, 3] == 2
    assert V[1, 4] == 3


def test_taylor_E8u_row0_shape():
    """V shape is (8, 11) for E8u_1 row 0 (i=0, t=11, q=7, nu=1)."""
    V, rhs = build_taylor_system(0, 11, 7, 1)
    assert V.shape == (8, 11)
    assert rhs.shape == (8, 1)


# ---------------------------------------------------------------------------
# 20.3b -- Single-row boundary solver tests
# ---------------------------------------------------------------------------

a0, a1, a2, a3, a4, a5, a6 = symbols("alpha_0 alpha_1 alpha_2 alpha_3 alpha_4 alpha_5 alpha_6")


def test_solve_row_E4u_row0():
    """E4u row 0: 1 free param (alpha_0), 5 coefficients."""
    result = solve_boundary_row(i=0, t=5, q=3, nu=1, free_symbols=[a0])
    assert result.row_index == 0
    assert len(result.coefficients) == 5
    assert result.free_params == [a0]
    expected = [
        (6 * a0 - 11) / 6,
        3 - 4 * a0,
        (12 * a0 - 3) / 2,
        -(12 * a0 - 1) / 3,
        a0,
    ]
    for got, exp in zip(result.coefficients, expected):
        assert cancel(got - exp) == 0, f"{got} != {exp}"


def test_solve_row_E4u_row1():
    """E4u row 1: 1 free param (alpha_1), 5 coefficients."""
    result = solve_boundary_row(i=1, t=5, q=3, nu=1, free_symbols=[a1])
    assert result.row_index == 1
    assert len(result.coefficients) == 5
    assert result.free_params == [a1]
    expected = [
        (3 * a1 - 1) / 3,
        -(8 * a1 + 1) / 2,
        6 * a1 + 1,
        -(24 * a1 + 1) / 6,
        a1,
    ]
    for got, exp in zip(result.coefficients, expected):
        assert cancel(got - exp) == 0, f"{got} != {exp}"


def test_solve_row_zero_padded():
    """E6u row 0: 2 free slots, second is zero-padded."""
    result = solve_boundary_row(i=0, t=8, q=5, nu=1, free_symbols=[a0, S.Zero])
    assert len(result.coefficients) == 8
    assert result.coefficients[7] == S.Zero
    assert result.coefficients[6] == a0


def test_solve_row_two_free():
    """E6u row 3 (penultimate): 2 active free params."""
    result = solve_boundary_row(
        i=3, t=8, q=5, nu=1, free_symbols=[a3, a4]
    )
    assert len(result.coefficients) == 8
    assert result.coefficients[6] == a3
    assert result.coefficients[7] == a4


# ---------------------------------------------------------------------------
# 20.3d -- Conservation constraint solver tests
# ---------------------------------------------------------------------------


def test_conservation_weight_count_E4u(e4u_pipeline):
    """Verify 3 weight symbols exist and are solved."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    assert len(w_syms) == 3
    for w in w_syms:
        assert w in solution_dict


def test_conservation_placeholders_resolved_E4u(e4u_pipeline):
    """Verify last row has no phi_* symbols remaining."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    last_row = updated_rows[2]
    allowed = set(result.all_free_params)
    for coeff in last_row.coefficients:
        assert coeff.free_symbols <= allowed, (
            f"Unexpected symbols in last row: {coeff.free_symbols - allowed}"
        )


def test_conservation_redundant_column_E4u(e4u_pipeline):
    """Verify the redundant column (t-1=4) sums to zero."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    w_exprs = [solution_dict[w] for w in w_syms]
    col_sum = sum(w * row.coefficients[4]
                  for w, row in zip(w_exprs, updated_rows))
    col_sum += _interior_contribution(4, result.r, 2, result.interior_coeffs)
    assert cancel(col_sum) == 0


# ---------------------------------------------------------------------------
# 20.3e -- E4u_1 end-to-end validation tests
# ---------------------------------------------------------------------------

# Alpha symbols for E4u
_a0, _a1 = symbols("alpha_0 alpha_1")

# Alpha values from E4u_1.t.cpp
_alpha_vals_e4 = {
    _a0: -0.7733323791884821,
    _a1:  0.1623961700641681,
}


def test_E4u_taylor_shape_and_entries(e4u_pipeline):
    """Taylor system shape and specific entries for E4u row 0."""
    V, rhs = build_taylor_system(0, 5, 3, 1)
    assert V.shape == (4, 5)
    assert V[0, 0] == 1
    assert V[1, 1] == 1
    assert V[2, 3] == Rational(9, 2)
    assert V[3, 4] == Rational(32, 3)


def test_E4u_row0_symbolic(e4u_pipeline):
    """Row 0 symbolic coefficients match E4u_1.cpp lines 80-84."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    row = updated_rows[0]
    expected = [
        (6 * _a0 - 11) / 6,
        3 - 4 * _a0,
        (12 * _a0 - 3) / 2,
        -(12 * _a0 - 1) / 3,
        _a0,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 0 coeff {i}: {got} != {exp}"


def test_E4u_row1_symbolic(e4u_pipeline):
    """Row 1 symbolic coefficients match E4u_1.cpp lines 85-89."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    row = updated_rows[1]
    expected = [
        (3 * _a1 - 1) / 3,
        -(8 * _a1 + 1) / 2,
        6 * _a1 + 1,
        -(24 * _a1 + 1) / 6,
        _a1,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 1 coeff {i}: {got} != {exp}"


def test_E4u_row2_symbolic(e4u_pipeline):
    """Row 2 (conservation-constrained) symbolic coefficients match E4u_1.cpp lines 90-94."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    row = updated_rows[2]
    expected = [
        -(168 * _a1 + 54 * _a0 - 11) / 138,
        (112 * _a1 + 36 * _a0 - 15) / 23,
        -(336 * _a1 + 108 * _a0 + 1) / 46,
        (336 * _a1 + 108 * _a0 + 47) / 69,
        -(28 * _a1 + 9 * _a0 + 2) / 23,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 2 coeff {i}: {got} != {exp}"


def test_E4u_numerical_floating(e4u_pipeline):
    """Numerical evaluation (floating, h=2) matches E4u_1.t.cpp."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    h = 2
    expected_float = [
        -1.3033328562609077, 3.046664758376964, -3.069997137565446,
        1.713331425043631, -0.38666618959424104,
        -0.08546858163458262, -0.5747923401283361, 0.9871885101925043,
        -0.4081256734616695, 0.08119808503208405,
        0.0923093909615862, -0.5359042305130115, 0.3038563457695172,
        0.13076243615365518, 0.00897605762825287,
    ]
    computed = []
    for row in updated_rows:
        for coeff in row.coefficients:
            val = float(coeff.xreplace(_alpha_vals_e4)) / h
            computed.append(val)
    assert computed == pytest.approx(expected_float, abs=1e-12)


def test_E4u_numerical_dirichlet(e4u_pipeline):
    """Numerical evaluation (Dirichlet, h=0.5) matches E4u_1.t.cpp."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    h = 0.5
    # Dirichlet drops row 0, uses rows 1 and 2
    expected_dirichlet = [
        -0.3418743265383305, -2.2991693605133445, 3.9487540407700172,
        -1.632502693846678, 0.3247923401283362,
        0.3692375638463448, -2.143616922052046, 1.2154253830780688,
        0.5230497446146207, 0.03590423051301148,
    ]
    computed = []
    for row in updated_rows[1:]:  # skip row 0
        for coeff in row.coefficients:
            val = float(coeff.xreplace(_alpha_vals_e4)) / h
            computed.append(val)
    assert computed == pytest.approx(expected_dirichlet, abs=1e-12)


def test_E4u_conservation_column_sums(e4u_pipeline):
    """Conservation verification: weighted column sums satisfy SBP."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    w_exprs = [solution_dict[w] for w in w_syms]
    t = result.t  # 5
    r = result.r  # 3
    p = 2

    for j in range(t):
        col_sum = sum(
            w * row.coefficients[j]
            for w, row in zip(w_exprs, updated_rows)
        )
        col_sum += _interior_contribution(j, r, p, result.interior_coeffs)
        if j == 0:
            # Column 0 sums to -1
            assert cancel(col_sum + 1) == 0, f"Column {j} SBP failed"
        else:
            # All other columns sum to 0
            assert cancel(col_sum) == 0, f"Column {j} SBP failed"


def test_E4u_polynomial_exactness(e4u_pipeline):
    """Polynomial exactness up to degree q=3."""
    updated_rows, solution_dict, w_syms, result = e4u_pipeline
    t = result.t  # 5

    for d in range(4):  # degrees 0, 1, 2, 3
        # Grid values f(j) = j^d for j = 0..t-1
        grid_vals = [Rational(j) ** d for j in range(t)]
        for row in updated_rows:
            i = row.row_index
            # Apply stencil: sum_j coeff_j * f(j)
            stencil_result = sum(
                c * fj for c, fj in zip(row.coefficients, grid_vals)
            )
            # Expected: d-th derivative of x^d at x=i
            if d == 0:
                expected = 0
            elif d == 1:
                expected = 1
            else:
                # d-th derivative of x^d w.r.t. first derivative = d * i^(d-1)
                expected = d * Rational(i) ** (d - 1)
            assert cancel(stencil_result - expected) == 0, (
                f"Poly exactness failed: d={d}, row={i}, "
                f"got {stencil_result}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# 20.3f -- E6u_1 end-to-end validation tests
# ---------------------------------------------------------------------------

# Alpha symbols for E6u (reuse a0..a4 already defined at module level)
_alpha_vals_e6 = {
    a0: 0.1,
    a1: 0.2,
    a2: 0.3,
    a3: 0.4,
    a4: 0.5,
}


def test_E6u_row0_symbolic(e6u_pipeline):
    """Row 0 symbolic coefficients match E6u_1.cpp lines 81-88."""
    updated_rows, solution_dict, w_syms, result = e6u_pipeline
    row = updated_rows[0]
    expected = [
        (60 * a0 - 137) / 60,
        5 - 6 * a0,
        15 * a0 - 5,
        -(60 * a0 - 10) / 3,
        (60 * a0 - 5) / 4,
        -(30 * a0 - 1) / 5,
        a0,
        S.Zero,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 0 coeff {i}: {got} != {exp}"


def test_E6u_row1_symbolic(e6u_pipeline):
    """Row 1 symbolic coefficients match E6u_1.cpp lines 89-96."""
    updated_rows, solution_dict, w_syms, result = e6u_pipeline
    row = updated_rows[1]
    expected = [
        (5 * a1 - 1) / 5,
        -(72 * a1 + 13) / 12,
        15 * a1 + 2,
        -20 * a1 - 1,
        (45 * a1 + 1) / 3,
        -(120 * a1 + 1) / 20,
        a1,
        S.Zero,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 1 coeff {i}: {got} != {exp}"


def test_E6u_row2_symbolic(e6u_pipeline):
    """Row 2 symbolic coefficients match E6u_1.cpp lines 97-104."""
    updated_rows, solution_dict, w_syms, result = e6u_pipeline
    row = updated_rows[2]
    expected = [
        (20 * a2 + 1) / 20,
        -(12 * a2 + 1) / 2,
        (45 * a2 - 1) / 3,
        1 - 20 * a2,
        (60 * a2 - 1) / 4,
        -(180 * a2 - 1) / 30,
        a2,
        S.Zero,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 2 coeff {i}: {got} != {exp}"


def test_E6u_row3_symbolic(e6u_pipeline):
    """Row 3 symbolic coefficients match E6u_1.cpp lines 105-112."""
    updated_rows, solution_dict, w_syms, result = e6u_pipeline
    row = updated_rows[3]
    expected = [
        (180 * a4 + 30 * a3 - 1) / 30,
        -(140 * a4 + 24 * a3 - 1) / 4,
        84 * a4 + 15 * a3 - 1,
        -(315 * a4 + 60 * a3 - 1) / 3,
        (140 * a4 + 30 * a3 + 1) / 2,
        -(420 * a4 + 120 * a3 + 1) / 20,
        a3,
        a4,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 3 coeff {i}: {got} != {exp}"


def test_E6u_row4_symbolic(e6u_pipeline):
    """Row 4 (conservation-constrained) symbolic coefficients match E6u_1.cpp lines 113-134."""
    updated_rows, solution_dict, w_syms, result = e6u_pipeline
    row = updated_rows[4]
    expected = [
        -(190320 * a4 + 31720 * a3 + 22080 * a2 + 38040 * a1 + 9500 * a0 - 453) / 28260,
        (55510 * a4 + 9516 * a3 + 6624 * a2 + 11412 * a1 + 2850 * a0 - 159) / 1413,
        -(44408 * a4 + 7930 * a3 + 5520 * a2 + 9510 * a1 + 2375 * a0 - 183) / 471,
        (166530 * a4 + 31720 * a3 + 22080 * a2 + 38040 * a1 + 9500 * a0 - 1506) / 1413,
        -(444080 * a4 + 95160 * a3 + 66240 * a2 + 114120 * a1 + 28500 * a0 - 1323) / 5652,
        (55510 * a4 + 15860 * a3 + 11040 * a2 + 19020 * a1 + 4750 * a0 + 1551) / 2355,
        -(1586 * a3 + 1104 * a2 + 1902 * a1 + 475 * a0 + 192) / 1413,
        -(1586 * a4 - 24) / 1413,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 4 coeff {i}: {got} != {exp}"


def test_E6u_numerical_floating(e6u_pipeline):
    """Numerical evaluation (floating, h=2) matches E6u_1.t.cpp."""
    updated_rows, solution_dict, w_syms, result = e6u_pipeline
    h = 2
    expected_float = [
        -1.0916666666666666, 2.2, -1.75, 0.6666666666666666,
        0.12500000000000006, -0.2, 0.05, 0.0,
        5.551115123125783e-18, -1.1416666666666666, 2.5, -2.5,
        1.6666666666666667, -0.625, 0.1, 0.0,
        0.175, -1.15, 2.083333333333333, -2.5,
        2.125, -0.8833333333333333, 0.15, 0.0,
        1.6833333333333333, -9.825, 23.5, -30.083333333333332,
        20.75, -6.475, 0.2, 0.25,
        -2.1687367303609344, 12.723637650389243, -30.773354564755838,
        38.79299363057325, -26.92206298655343, 9.18067940552017,
        -0.5610403397027601, -0.27211606510969566,
    ]
    computed = []
    for row in updated_rows:
        for coeff in row.coefficients:
            val = float(coeff.xreplace(_alpha_vals_e6)) / h
            computed.append(val)
    assert computed == pytest.approx(expected_float, abs=1e-10)


def test_E6u_numerical_dirichlet(e6u_pipeline):
    """Numerical evaluation (Dirichlet, h=0.5) matches E6u_1.t.cpp."""
    updated_rows, solution_dict, w_syms, result = e6u_pipeline
    h = 0.5
    # Dirichlet drops row 0 => rows 1-4
    expected_dirichlet = [
        2.2204460492503132e-17, -4.566666666666666, 10.0, -10.0,
        6.666666666666667, -2.5, 0.4, 0.0,
        0.7, -4.6, 8.333333333333332, -10.0,
        8.5, -3.533333333333333, 0.6, 0.0,
        6.733333333333333, -39.3, 94.0, -120.33333333333333,
        83.0, -25.9, 0.8, 1.0,
        -8.674946921443738, 50.89455060155697, -123.09341825902335,
        155.171974522293, -107.68825194621373, 36.72271762208068,
        -2.2441613588110405, -1.0884642604387826,
    ]
    computed = []
    for row in updated_rows[1:]:  # skip row 0
        for coeff in row.coefficients:
            val = float(coeff.xreplace(_alpha_vals_e6)) / h
            computed.append(val)
    assert computed == pytest.approx(expected_dirichlet, abs=1e-10)


def test_E6u_polynomial_exactness(e6u_pipeline):
    """Polynomial exactness up to degree q=5."""
    updated_rows, solution_dict, w_syms, result = e6u_pipeline
    t = result.t  # 8

    for d in range(6):  # degrees 0, 1, 2, 3, 4, 5
        # Grid values f(j) = j^d for j = 0..t-1
        grid_vals = [Rational(j) ** d for j in range(t)]
        for row in updated_rows:
            i = row.row_index
            # Apply stencil: sum_j coeff_j * f(j)
            stencil_result = sum(
                c * fj for c, fj in zip(row.coefficients, grid_vals)
            )
            # Expected: first derivative of x^d at x=i
            if d == 0:
                expected = 0
            elif d == 1:
                expected = 1
            else:
                expected = d * Rational(i) ** (d - 1)
            assert cancel(stencil_result - expected) == 0, (
                f"Poly exactness failed: d={d}, row={i}, "
                f"got {stencil_result}, expected {expected}"
            )


def test_E6u_conservation_column_sums(e6u_pipeline):
    """Conservation verification: weighted column sums satisfy SBP."""
    updated_rows, solution_dict, w_syms, result = e6u_pipeline
    w_exprs = [solution_dict[w] for w in w_syms]
    t = result.t  # 8
    r = result.r  # 5
    p = 3

    for j in range(t):
        col_sum = sum(
            w * row.coefficients[j]
            for w, row in zip(w_exprs, updated_rows)
        )
        col_sum += _interior_contribution(j, r, p, result.interior_coeffs)
        if j == 0:
            assert cancel(col_sum + 1) == 0, f"Column {j} SBP failed"
        else:
            assert cancel(col_sum) == 0, f"Column {j} SBP failed"


# ---------------------------------------------------------------------------
# 20.3g -- E8u_1 end-to-end validation tests
# ---------------------------------------------------------------------------

# Alpha values from E8u_1.t.cpp lines 25-31
_alpha_vals_e8 = {
    a0: -0.6484343281044554,
    a1:  0.1546307245964576,
    a2: -0.0024361150534464,
    a3: -0.0767677234359209,
    a4:  0.0395245547501803,
    a5: -0.25779890216368,
    a6:  0.0527307049447768,
}


def test_E8u_row0_symbolic(e8u_pipeline):
    """Row 0 symbolic coefficients match E8u_1.cpp lines 83-93."""
    updated_rows, solution_dict, w_syms, result = e8u_pipeline
    row = updated_rows[0]
    expected = [
        (140 * a0 - 363) / 140,
        7 - 8 * a0,
        (56 * a0 - 21) / 2,
        -(168 * a0 - 35) / 3,
        (280 * a0 - 35) / 4,
        -(280 * a0 - 21) / 5,
        (168 * a0 - 7) / 6,
        -(56 * a0 - 1) / 7,
        a0,
        S.Zero,
        S.Zero,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 0 coeff {i}: {got} != {exp}"


def test_E8u_row1_symbolic(e8u_pipeline):
    """Row 1 symbolic coefficients match E8u_1.cpp lines 94-104."""
    updated_rows, solution_dict, w_syms, result = e8u_pipeline
    row = updated_rows[1]
    expected = [
        (7 * a1 - 1) / 7,
        -(160 * a1 + 29) / 20,
        28 * a1 + 3,
        -(112 * a1 + 5) / 2,
        (210 * a1 + 5) / 3,
        -(224 * a1 + 3) / 4,
        (140 * a1 + 1) / 5,
        -(336 * a1 + 1) / 42,
        a1,
        S.Zero,
        S.Zero,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 1 coeff {i}: {got} != {exp}"


def test_E8u_row2_symbolic(e8u_pipeline):
    """Row 2 symbolic coefficients match E8u_1.cpp lines 105-115."""
    updated_rows, solution_dict, w_syms, result = e8u_pipeline
    row = updated_rows[2]
    expected = [
        (42 * a2 + 1) / 42,
        -(24 * a2 + 1) / 3,
        (1680 * a2 - 47) / 60,
        -(168 * a2 - 5) / 3,
        (420 * a2 - 5) / 6,
        -(168 * a2 - 1) / 3,
        (336 * a2 - 1) / 12,
        -(840 * a2 - 1) / 105,
        a2,
        S.Zero,
        S.Zero,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 2 coeff {i}: {got} != {exp}"


def test_E8u_row3_symbolic(e8u_pipeline):
    """Row 3 symbolic coefficients match E8u_1.cpp lines 116-126."""
    updated_rows, solution_dict, w_syms, result = e8u_pipeline
    row = updated_rows[3]
    expected = [
        (105 * a3 - 1) / 105,
        -(80 * a3 - 1) / 10,
        (140 * a3 - 3) / 5,
        -(224 * a3 + 1) / 4,
        70 * a3 + 1,
        -(560 * a3 + 3) / 10,
        (420 * a3 + 1) / 15,
        -(1120 * a3 + 1) / 140,
        a3,
        S.Zero,
        S.Zero,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 3 coeff {i}: {got} != {exp}"


def test_E8u_row4_symbolic(e8u_pipeline):
    """Row 4 symbolic coefficients match E8u_1.cpp lines 127-137."""
    updated_rows, solution_dict, w_syms, result = e8u_pipeline
    row = updated_rows[4]
    expected = [
        (140 * a4 + 1) / 140,
        -(120 * a4 + 1) / 15,
        (280 * a4 + 3) / 10,
        -56 * a4 - 1,
        (280 * a4 + 1) / 4,
        -(280 * a4 - 3) / 5,
        (280 * a4 - 1) / 10,
        -(840 * a4 - 1) / 105,
        a4,
        S.Zero,
        S.Zero,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 4 coeff {i}: {got} != {exp}"


def test_E8u_row5_symbolic(e8u_pipeline):
    """Row 5 symbolic coefficients match E8u_1.cpp lines 138-148."""
    updated_rows, solution_dict, w_syms, result = e8u_pipeline
    row = updated_rows[5]
    expected = [
        (840 * a6 + 105 * a5 - 1) / 105,
        -(756 * a6 + 96 * a5 - 1) / 12,
        (648 * a6 + 84 * a5 - 1) / 3,
        -(2520 * a6 + 336 * a5 - 5) / 6,
        (1512 * a6 + 210 * a5 - 5) / 3,
        -(22680 * a6 + 3360 * a5 - 47) / 60,
        (504 * a6 + 84 * a5 + 1) / 3,
        -(1512 * a6 + 336 * a5 + 1) / 42,
        a5,
        a6,
        S.Zero,
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 5 coeff {i}: {got} != {exp}"


def test_E8u_row6_symbolic(e8u_pipeline):
    """Row 6 (conservation-constrained) symbolic coefficients match E8u_1.cpp lines 149-185."""
    updated_rows, solution_dict, w_syms, result = e8u_pipeline
    row = updated_rows[6]
    expected = [
        -(43994496 * a6 + 5499312 * a5 + 3756354 * a4 + 7475328 * a3
          + 2303742 * a2 + 7419216 * a1 + 1545558 * a0 - 28865) / 5022570,
        (8248968 * a6 + 1047488 * a5 + 715496 * a4 + 1423872 * a3
         + 438808 * a2 + 1413184 * a1 + 294392 * a0 - 5917) / 119585,
        -(113128704 * a6 + 14664832 * a5 + 10016944 * a4 + 19934208 * a3
          + 6143312 * a2 + 19784576 * a1 + 4121488 * a0 - 92067) / 478340,
        (164979360 * a6 + 21997248 * a5 + 15025416 * a4 + 29901312 * a3
         + 9214968 * a2 + 29676864 * a1 + 6182232 * a0 - 164197) / 358755,
        -(131983488 * a6 + 18331040 * a5 + 12521180 * a4 + 24917760 * a3
          + 7679140 * a2 + 24730720 * a1 + 5151860 * a0 - 190693) / 239170,
        (49493808 * a6 + 7332416 * a5 + 5008472 * a4 + 9967104 * a3
         + 3071656 * a2 + 9892288 * a1 + 2060744 * a0 - 163203) / 119585,
        -(87988992 * a6 + 14664832 * a5 + 10016944 * a4 + 19934208 * a3
          + 6143312 * a2 + 19784576 * a1 + 4121488 * a0 - 169433) / 478340,
        (32995872 * a6 + 7332416 * a5 + 5008472 * a4 + 9967104 * a3
         + 3071656 * a2 + 9892288 * a1 + 2060744 * a0 + 551009) / 837095,
        -(130936 * a5 + 89437 * a4 + 177984 * a3
          + 54851 * a2 + 176648 * a1 + 36799 * a0 + 20016) / 119585,
        -(130936 * a6 - 4176) / 119585,
        -Rational(432, 119585),
    ]
    for i, (got, exp) in enumerate(zip(row.coefficients, expected)):
        assert cancel(got - exp) == 0, f"Row 6 coeff {i}: {got} != {exp}"


def test_E8u_numerical_floating(e8u_pipeline):
    """Numerical evaluation (floating, h=2) matches E8u_1.t.cpp."""
    updated_rows, solution_dict, w_syms, result = e8u_pipeline
    h = 2
    expected_float = [
        -1.620645735480799, 6.093737312417822, -14.328080593462376,
        23.989494520258084, -27.07020148365594, 20.256161186924754,
        -9.66141392679571, 2.6651658838463934, -0.3242171640522277, 0.0, 0.0,
        0.005886790869657372, -1.3435228983858305, 3.6648301443504065,
        -5.579660288700813, 6.24540869420935, -4.704660288700813,
        2.2648301443504066, -0.6304276602905923, 0.0773153622982288, 0.0, 0.0,
        0.010686704378038705, -0.15692220645288107, -0.42577227741491624,
        0.9015445548298325, -0.5019306935372907, 0.23487788816316588,
        -0.07577227741491627, 0.014506364975690363, -0.0012180575267232, 0.0, 0.0,
        -0.04314576647986521, 0.3570708937436836, -1.3747481281028926,
        2.0244962562057855, -2.1868703202572317, 1.9994962562057854,
        -1.0414147947695593, 0.30349946517225507, -0.03838386171796045, 0.0, 0.0,
        0.023333705946518724, -0.19143155233405454, 0.7033437665025243,
        -1.6066875330050485, 1.5083594162563105, -0.8066875330050485,
        0.5033437665025242, -0.15333631423881644, 0.01976227737509015, 0.0, 0.0,
        0.07726146393536244, -0.5881549304390825, 1.9190648370777077,
        -3.4384121111534216, 3.4318427370216202, -2.3560673073131086,
        0.9868612517363978, 0.07013815774397571, -0.12889945108184,
        0.0263653524723884, 0.0,
        -0.058467796419719775, 0.4371229276793953, -1.3903775683566566,
        2.397968815999491, -2.278631808271436, 0.7319853517776528,
        0.07615328403773296, 0.01154775812234574, 0.085912848127809,
        -0.01140756609377972, -0.0018062466028348036,
    ]
    computed = []
    for row in updated_rows:
        for coeff in row.coefficients:
            val = float(coeff.xreplace(_alpha_vals_e8)) / h
            computed.append(val)
    assert computed == pytest.approx(expected_float, abs=1e-10)


def test_E8u_numerical_dirichlet(e8u_pipeline):
    """Numerical evaluation (Dirichlet, h=0.5) matches E8u_1.t.cpp."""
    updated_rows, solution_dict, w_syms, result = e8u_pipeline
    h = 0.5
    # Dirichlet drops row 0 => rows 1-6
    expected_dirichlet = [
        0.02354716347862949, -5.374091593543322, 14.659320577401626,
        -22.318641154803252, 24.9816347768374, -18.818641154803252,
        9.059320577401627, -2.521710641162369, 0.3092614491929152, 0.0, 0.0,
        0.04274681751215482, -0.6276888258115243, -1.703089109659665,
        3.60617821931933, -2.007722774149163, 0.9395115526526635,
        -0.30308910965966507, 0.05802545990276145, -0.0048722301068928, 0.0, 0.0,
        -0.17258306591946085, 1.4282835749747345, -5.49899251241157,
        8.097985024823142, -8.747481281028927, 7.997985024823142,
        -4.165659179078237, 1.2139978606890203, -0.1535354468718418, 0.0, 0.0,
        0.0933348237860749, -0.7657262093362182, 2.813375066010097,
        -6.426750132020194, 6.033437665025242, -3.226750132020194,
        2.013375066010097, -0.6133452569552658, 0.0790491095003606, 0.0, 0.0,
        0.30904585574144977, -2.35261972175633, 7.676259348310831,
        -13.753648444613686, 13.727370948086481, -9.424269229252435,
        3.9474450069455913, 0.28055263097590283, -0.51559780432736,
        0.1054614098895536, 0.0,
        -0.2338711856788791, 1.7484917107175812, -5.561510273426626,
        9.591875263997965, -9.114527233085743, 2.927941407110611,
        0.30461313615093183, 0.04619103248938296, 0.343651392511236,
        -0.04563026437511888, -0.0072249864113392145,
    ]
    computed = []
    for row in updated_rows[1:]:  # skip row 0
        for coeff in row.coefficients:
            val = float(coeff.xreplace(_alpha_vals_e8)) / h
            computed.append(val)
    assert computed == pytest.approx(expected_dirichlet, abs=1e-10)


def test_E8u_conservation_column_sums(e8u_pipeline):
    """Conservation verification: weighted column sums satisfy SBP."""
    updated_rows, solution_dict, w_syms, result = e8u_pipeline
    w_exprs = [solution_dict[w] for w in w_syms]
    t = result.t  # 11
    r = result.r  # 7
    p = 4

    for j in range(t):
        col_sum = sum(
            w * row.coefficients[j]
            for w, row in zip(w_exprs, updated_rows)
        )
        col_sum += _interior_contribution(j, r, p, result.interior_coeffs)
        if j == 0:
            assert cancel(col_sum + 1) == 0, f"Column {j} SBP failed"
        else:
            assert cancel(col_sum) == 0, f"Column {j} SBP failed"


def test_E8u_polynomial_exactness(e8u_pipeline):
    """Polynomial exactness up to degree q=7."""
    updated_rows, solution_dict, w_syms, result = e8u_pipeline
    t = result.t  # 11

    for d in range(8):  # degrees 0, 1, ..., 7
        # Grid values f(j) = j^d for j = 0..t-1
        grid_vals = [Rational(j) ** d for j in range(t)]
        for row in updated_rows:
            i = row.row_index
            # Apply stencil: sum_j coeff_j * f(j)
            stencil_result = sum(
                c * fj for c, fj in zip(row.coefficients, grid_vals)
            )
            # Expected: first derivative of x^d at x=i
            if d == 0:
                expected = 0
            elif d == 1:
                expected = 1
            else:
                expected = d * Rational(i) ** (d - 1)
            assert cancel(stencil_result - expected) == 0, (
                f"Poly exactness failed: d={d}, row={i}, "
                f"got {stencil_result}, expected {expected}"
            )


def test_E8u_performance():
    """E8u derivation completes within 2 seconds."""
    import time
    from stencil_gen.boundary import derive_boundary
    start = time.perf_counter()
    derive_boundary(p=4, nu=1, s=0)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"E8u derivation took {elapsed:.2f}s (budget: 2.0s)"
