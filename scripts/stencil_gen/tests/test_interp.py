"""Symbolic golden + derivation-correctness tests for the poly-interp module.

Covers design §4.1: interior interp golden, the shipped 2-point runtime form,
the interp→derivative duality (central difference and wall rows vs nbs_floating),
ia/fa decoupling under d/dy, and degree-1 interpolation exactness.
"""

import os
import sys

from sympy import Rational, Symbol, cancel, diff, factorial, solve

from stencil_gen.interior import derive_interior, full_gamma_array
from stencil_gen.interp import (
    build_cut_cell_interp_deltas,
    derivative_from_interp,
    derive_dirichlet_coeffs,
    derive_floating_coeffs,
    derive_interior_interp,
    derive_poly_interp,
    dirichlet_coeffs_symbolic,
)
from stencil_gen.taylor_system import build_interp_system

_tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
sys.path.insert(0, _tools_dir)

y = Symbol("y")
psi = Symbol("psi")


# ── taylor_system: interp RHS ────────────────────────────────────────────
def test_build_interp_system_rhs():
    """build_interp_system swaps the RHS to the eval-at-y functional."""
    _, rhs = build_interp_system(0, 3, 2, y)
    assert cancel(rhs[0] - 1) == 0
    assert cancel(rhs[1] - y) == 0
    assert cancel(rhs[2] - y**2 / 2) == 0


# ── interior interpolation ───────────────────────────────────────────────
def test_interior_interp_golden():
    """3-point quadratic interior interp matches PolyE2_1.nb p.6 `int`."""
    ic = derive_interior_interp(1, y)
    expected = [(y**2 - y) / 2, 1 - y**2, (y**2 + y) / 2]
    assert all(cancel(a - e) == 0 for a, e in zip(ic.coeffs, expected))


def test_interior_runtime_2pt():
    """Shipped runtime form is the 2-point linear bracketing interp."""
    ic = derive_interior_interp(1, y)
    assert cancel(ic.runtime_pos[0] - (1 - y)) == 0
    assert cancel(ic.runtime_pos[1] - y) == 0
    assert cancel(ic.runtime_neg[0] - (-y)) == 0
    assert cancel(ic.runtime_neg[1] - (1 + y)) == 0
    assert ic.runtime_p == 2


def test_derivative_from_interp_central():
    """d/dy|0 of the interior interp == central difference [-1/2, 0, 1/2]
    == derive_interior(0,1,1) full gamma array."""
    full = [(y**2 - y) / 2, 1 - y**2, (y**2 + y) / 2]
    dd = derivative_from_interp(full, y)
    assert dd == [Rational(-1, 2), Rational(0), Rational(1, 2)]
    prod = full_gamma_array(derive_interior(0, 1, 1))
    assert [Rational(v) for v in dd] == [Rational(v) for v in prod]


# ── cut-cell wall rows ───────────────────────────────────────────────────
def _committed_left():
    """Committed C++ interp_wall left case0/case1 (polyE2_1.cpp:116-134)."""
    t6 = 1 / (1 + psi)
    fa = [Symbol(f"fa_{k}") for k in range(4)]
    ia = [Symbol(f"ia_{k}") for k in range(4)]
    case0 = [
        (1 + psi + ia[1] - y + fa[1] * y) * Rational(1, 2) * t6,
        (1 + psi + fa[0] * y + ia[0] - y) * Rational(1, 2),
        (
            -psi
            + (fa[0] * y + ia[0]) * (-2)
            + y
            + (-2 * ia[1] - psi * ia[1] + y - 2 * fa[1] * y - psi * fa[1] * y) * t6
        )
        * Rational(1, 2),
        (ia[1] + fa[0] * y + ia[0] + fa[1] * y) * Rational(1, 2),
    ]
    case1 = [
        (1 + fa[3] * y + ia[3] - y) * Rational(1, 2) * t6,
        (1 + fa[2] * y + ia[2] - y) * Rational(1, 2),
        (
            (fa[2] * y + ia[2]) * (-2)
            + y
            + (psi - 2 * ia[3] - psi * ia[3] + y - 2 * fa[3] * y - psi * fa[3] * y) * t6
        )
        * Rational(1, 2),
        (fa[2] * y + ia[3] + ia[2] + fa[3] * y) * Rational(1, 2),
    ]
    return [case0, case1]


def _committed_right():
    """Committed C++ interp_wall right case0/case1 (polyE2_1.cpp:78-96)."""
    t17 = 1 / (1 + psi)
    fa = [Symbol(f"fa_{k}") for k in range(4)]
    ia = [Symbol(f"ia_{k}") for k in range(4)]
    case0 = [
        (fa[0] * y + fa[1] * y + ia[0] + ia[1]) * Rational(1, 2),
        (
            -psi
            - y
            + (fa[0] * y + ia[0]) * (-2)
            + (-y - 2 * ia[1] - psi * ia[1] - 2 * fa[1] * y - psi * fa[1] * y) * t17
        )
        * Rational(1, 2),
        (1 + psi + fa[0] * y + ia[0] + y) * Rational(1, 2),
        (1 + psi + fa[1] * y + ia[1] + y) * Rational(1, 2) * t17,
    ]
    case1 = [
        (ia[3] + fa[2] * y + fa[3] * y + ia[2]) * Rational(1, 2),
        (
            -y
            + (fa[2] * y + ia[2]) * (-2)
            + (psi - 2 * ia[3] - psi * ia[3] - y - 2 * fa[3] * y - psi * fa[3] * y) * t17
        )
        * Rational(1, 2),
        (1 + fa[2] * y + ia[2] + y) * Rational(1, 2),
        (1 + ia[3] + fa[3] * y + y) * Rational(1, 2) * t17,
    ]
    return [case0, case1]


def test_left_rows_golden():
    """Left wall rows match the committed C++ formulas symbolically."""
    res = derive_poly_interp(p=1, q=1)
    committed = _committed_left()
    for row, gold in zip(res.left_rows, committed):
        for c, g in zip(row.coeffs, gold):
            assert cancel(c - g) == 0


def test_right_rows_golden():
    """Right wall rows (separately derived mirror) match committed C++."""
    res = derive_poly_interp(p=1, q=1)
    committed = _committed_right()
    for row, gold in zip(res.right_rows, committed):
        for c, g in zip(row.coeffs, gold):
            assert cancel(c - g) == 0


def test_ia_vanishes_under_ddy():
    """ia symbols vanish under d/dy|0; fa symbols survive."""
    res = derive_poly_interp(p=1, q=1)
    for row in res.left_rows + res.right_rows:
        dd = derivative_from_interp(row.coeffs, y)
        ia_set = set(row.ia_syms)
        fa_set = set(row.fa_syms)
        deriv_syms = set().union(*(c.free_symbols for c in dd))
        assert ia_set.isdisjoint(deriv_syms), "ia leaked into derivative"
        # at least one fa survives
        assert fa_set & deriv_syms, "no fa survived d/dy"


def test_value_deriv_decoupling():
    """value(y=0) free syms ⊂ {ia,psi}; d/dy|0 free syms ⊂ {fa,psi};
    each coeff is exactly linear in y (O(y²) remainder == 0)."""
    res = derive_poly_interp(p=1, q=1)
    for row in res.left_rows + res.right_rows:
        ia_set = set(row.ia_syms)
        fa_set = set(row.fa_syms)
        for c in row.coeffs:
            v0 = cancel(c.subs(y, 0))
            v1 = cancel(diff(c, y).subs(y, 0))
            rem = cancel(c - v0 - y * v1)
            assert rem == 0, "coefficient is not exactly linear in y"
            assert fa_set.isdisjoint(v0.free_symbols)
            assert ia_set.isdisjoint(v1.free_symbols)


def test_interp_exactness():
    """Each wall row reproduces y^k exactly for k=0,1 (degree-1 exactness):
    Σ_j c_j δ_j^k == (target)^k, with target = closest-node delta + y."""
    res = derive_poly_interp(p=1, q=1)
    T = res.interp_T
    deltas_left = build_cut_cell_interp_deltas(T, psi, left=True)
    deltas_right = build_cut_cell_interp_deltas(T, psi, left=False)
    # left: row0 target -psi+y (closest=wall@col0), row1 target y (closest=node0@col1)
    left_targets = [deltas_left[0] + y, deltas_left[1] + y]
    right_targets = [deltas_right[T - 1] + y, deltas_right[T - 2] + y]
    for rows, deltas, targets in (
        (res.left_rows, deltas_left, left_targets),
        (res.right_rows, deltas_right, right_targets),
    ):
        for row, tgt in zip(rows, targets):
            for k in range(2):
                moment = sum(
                    row.coeffs[j] * deltas[j] ** k / factorial(k) for j in range(T)
                )
                assert cancel(moment - tgt**k / factorial(k)) == 0


def test_duality_vs_nbs_floating():
    """d/dy|0 of left interp rows == the corresponding nbs_floating rows."""
    res = derive_poly_interp(p=1, q=1)
    floating = derive_floating_coeffs(y, psi)
    T = res.interp_T
    for ri, row in enumerate(res.left_rows):
        dd = derivative_from_interp(row.coeffs, y)
        for j in range(T):
            assert cancel(dd[j] - floating[ri * T + j]) == 0


def test_dirichlet_derived_matches_oracle():
    """The DERIVED Dirichlet closure (poly recipe + conservation solve) equals
    the transcribed-from-C++ reference oracle, symbolically, for all 8 coeffs.

    Guards that the generation path's Dirichlet derivation never drifts from the
    committed nbs_dirichlet (== Mathematica e2D)."""
    derived = derive_dirichlet_coeffs(psi)
    oracle = dirichlet_coeffs_symbolic(psi)
    assert len(derived) == len(oracle) == 8
    for k, (a, b) in enumerate(zip(derived, oracle)):
        assert cancel(a - b) == 0, f"dirichlet c[{k}] derived != oracle: {cancel(a - b)}"


def test_dirichlet_discrete_conservation_symbolic():
    """The realized poly Dirichlet+interior left-matrix satisfies discrete
    conservation (telescoping/flux): with conservation (quadrature) weights, the
    WALL column weighted-sum == -1 and every fully-covered interior grid-point
    column == 0, symbolically in psi and da_0..da_2."""
    da = [Symbol(f"da_{k}") for k in range(3)]
    coeffs = derive_dirichlet_coeffs(psi)            # 8 = 2 rows x 4 cols
    row0, row1 = coeffs[:4], coeffs[4:]
    interior = [Rational(-1, 2), Rational(0), Rational(1, 2)]
    # realized matrix: 2 Dirichlet rows on T-frame deltas [-psi,0,1,2] (col0=wall),
    # plus interior band rows centered at grid cols 3,4 so cols 1,2 are covered.
    NC = 6

    def band(c0):
        r = [Rational(0)] * NC
        for k, c in enumerate(interior):
            col = c0 + (k - 1)
            if 0 <= col < NC:
                r[col] = c
        return r

    def pad(rw):
        return list(rw) + [Rational(0)] * (NC - len(rw))

    wa, wb = Symbol("w_a"), Symbol("w_b")
    M = [pad(row0), pad(row1), band(3), band(4)]
    wts = [wa, wb, Rational(1), Rational(1)]

    def colsum(col):
        return cancel(sum(wts[i] * M[i][col] for i in range(4)))

    # solve the two fully-covered interior cols for the conservation weights
    sol = solve([colsum(1), colsum(2)], [wa, wb], dict=True)[0]
    # interior columns (1,2,3) == 0
    for col in (1, 2, 3):
        assert cancel(colsum(col).subs(sol)) == 0, f"interior col {col} != 0"
    # wall column weighted-sum == -1 (the boundary flux)
    wall = cancel((wa * row0[0] + wb * row1[0]).subs(sol))
    assert cancel(wall + 1) == 0, f"wall flux != -1: {wall}"
