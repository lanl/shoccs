"""Tests for the interval-operator assembler (plan 48 item 3.1).

Covers the plan's prescribed tests:

(a) psi_L = psi_R = 1 reproduces the uniform builder: the closure rows
    coincide with the uniform boundary rows B_u (item-1.1 psi = 1
    identity, with the documented near-interior-row caveat) and the whole
    matrix matches an independent symbolic assembly exactly;
(b) row-sum consistency — each row annihilates constants — at several
    (psi_L, psi_R).  For the wall-eliminated closure rows this is the
    FULL-SUPPORT statement: the constant function has value 1 at the wall
    too, so assembled row sum + wall weight = 0 exactly (the bare grid
    sum equals minus the wall weight, which is nonzero generically —
    that is exactly the coupling ``derivative.cpp`` strips into CSR B);
    interior rows sum to 0 outright;
(c) N large: interior rows equal the circulant stencil;
(d) smallest assemblable N = m_L + m_R (all-closure, still assembles)
    and a clear IntervalTooShort below it.

E4_1 tests are @pytest.mark.slow (symbolic derivation > 170 s, plan 48
scheme-scope constraint); derivations are cached with module/class-scoped
fixtures on top of the item-1.1 lru_cache.
"""

import numpy as np
import pytest
from sympy import Symbol

from stencil_gen.cutcell_eval import make_cutcell_evaluator
from stencil_gen.interval_operator import (
    IntervalTooShort,
    build_interval_operator,
)
from stencil_gen.temo import (
    E2_1,
    E4_1,
    derive_cut_cell_scheme,
    derive_uniform_boundary_for_temo,
)

# Nonzero, non-singular alpha point for E2_1 (single structural
# denominator psi + 1 never vanishes on (0, 1]); breaks the alpha = 0
# degeneracy where Dirichlet rows 0 and 1 coincide.
E2_1_ALPHA_NONZERO = [0.3, -0.2, 0.15, 0.05]

# Several (psi_L, psi_R) pairs spanning uniform, symmetric-cut,
# asymmetric-cut, and the near-clamp extremes.
PSI_PAIRS = [
    (1.0, 1.0),
    (0.5, 0.5),
    (0.3, 0.8),
    (1e-3, 0.9999),
]


@pytest.fixture(scope="module")
def e2_evaluator():
    """Module-cached E2_1 evaluator (lru_cache makes this near-free)."""
    return make_cutcell_evaluator(E2_1)


@pytest.fixture(scope="module")
def e2_alpha_zero(e2_evaluator):
    return {s: 0.0 for s in e2_evaluator.alpha_symbols}


@pytest.fixture(scope="module")
def e2_symbolic_dirichlet():
    """Independent symbolic E2_1 Dirichlet block (fresh derivation, kept
    symbolic in psi so both ends can be substituted independently)."""
    psi_sym = Symbol("psi")
    cc = derive_cut_cell_scheme(E2_1, psi_sym)
    return cc, psi_sym


def _symbolic_interval_operator(dirichlet_block, psi_sym, scheme,
                                alpha_subs, psi_L, psi_R, n):
    """Independent test-local assembly by direct sympy substitution.

    Mirrors the repo-wide convention (tests/test_group_velocity.py's
    ``_build_cut_cell_diff_matrix``) with a CUT closure at both ends:
    wall column 0 dropped (u_wall = 0), Dirichlet row i at grid node i,
    centered interior, and the nu = 1 codegen revneg image
    ``D[n-1-i, n-1-j] = -B[i, j]`` at the right end — deliberately NOT
    via ``mirror_closure``, so the production right-block path is
    cross-checked against the raw revneg definition.
    """
    from stencil_gen.interior import derive_interior, full_gamma_array

    F = dirichlet_block
    r, T = F.rows, F.cols
    p, nu = scheme.p, scheme.nu
    interior_w = [float(c) for c in full_gamma_array(derive_interior(0, p, nu))]
    sign = (-1.0) ** nu

    D = np.zeros((n, n))
    subs_L = {psi_sym: psi_L, **alpha_subs}
    subs_R = {psi_sym: psi_R, **alpha_subs}
    for i in range(r):
        for j in range(1, T):
            D[i, j - 1] = float(F[i, j].xreplace(subs_L))
            D[n - 1 - i, n - 1 - (j - 1)] = sign * float(F[i, j].xreplace(subs_R))
    for i in range(r, n - r):
        for k, jj in enumerate(range(i - p, i + p + 1)):
            D[i, jj] = interior_w[k]
    return D


def _wall_weights(evaluator, psi, alpha):
    """Wall-column weights of the Dirichlet closure rows at (psi, alpha),
    from direct symbolic substitution into the evaluator's block."""
    subs = {evaluator._psi_symbol: psi}
    subs.update({s: v for s, v in zip(evaluator.alpha_symbols,
                                      evaluator._alpha_args(alpha))})
    return np.array(
        [float(evaluator._block[i, 0].xreplace(subs))
         for i in range(evaluator._block.rows)]
    )


class TestPsi1UniformIdentity:
    """Plan test (a): psi_L = psi_R = 1 reproduces the uniform builder."""

    @pytest.mark.parametrize(
        "alpha_vals",
        [[0.0, 0.0, 0.0, 0.0], E2_1_ALPHA_NONZERO],
        ids=["alpha_zero", "alpha_nonzero"],
    )
    def test_psi1_matches_uniform_boundary_rows(self, e2_evaluator,
                                                alpha_vals):
        """At psi = 1 both dense ends carry the uniform boundary rows.

        Item-1.1 identity: Dirichlet row i = B_u row i+1 for
        i = 0..r-2 (the last, near-interior row retains its wall
        coupling by construction and has no B_u counterpart — it is the
        production uniform-domain-boundary closure's own row, excluded
        here exactly as in tests/test_cutcell_eval.py).  The right end
        must carry the nu = 1 revneg image -B_u[i+1, j] at
        [N-1-i, N-1-j].
        """
        ev = e2_evaluator
        alpha = dict(zip(ev.alpha_symbols, alpha_vals))
        N = 12
        D = build_interval_operator(E2_1, alpha, 1.0, 1.0, N)

        ur = derive_uniform_boundary_for_temo(E2_1)
        umap = dict(zip(ur.alpha_symbols, alpha_vals))
        r, t = ur.B_u.shape
        for i in range(r - 1):
            expected = np.array(
                [float(ur.B_u[i + 1, j].xreplace(umap)) for j in range(t)]
            )
            np.testing.assert_allclose(
                D[i, :t], expected, atol=1e-10,
                err_msg=f"left closure row {i} != uniform B_u row {i + 1}",
            )
            np.testing.assert_allclose(
                D[N - 1 - i, N - 1 - np.arange(t)], -expected, atol=1e-10,
                err_msg=f"right closure row {i} != revneg uniform B_u row "
                        f"{i + 1}",
            )

    @pytest.mark.parametrize("psi_L,psi_R", PSI_PAIRS,
                             ids=lambda v: str(v))
    def test_matches_independent_symbolic_assembly(
            self, e2_evaluator, e2_symbolic_dirichlet, psi_L, psi_R):
        """The assembled operator equals an independent symbolic assembly
        exactly, at psi = 1 (plan test a) and at cut (psi_L, psi_R)."""
        cc, psi_sym = e2_symbolic_dirichlet
        ev = e2_evaluator
        alpha = dict(zip(ev.alpha_symbols, E2_1_ALPHA_NONZERO))
        N = 12
        D = build_interval_operator(E2_1, alpha, psi_L, psi_R, N)
        D_ref = _symbolic_interval_operator(
            cc.dirichlet, psi_sym, E2_1, alpha, psi_L, psi_R, N
        )
        np.testing.assert_allclose(D, D_ref, atol=1e-12)

    def test_asymmetric_psi_gives_asymmetric_ends(self, e2_evaluator):
        """psi_L != psi_R: the two dense ends differ (each end is
        evaluated at its own psi)."""
        ev = e2_evaluator
        alpha = dict(zip(ev.alpha_symbols, E2_1_ALPHA_NONZERO))
        N = 12
        D = build_interval_operator(E2_1, alpha, 0.3, 0.8, N)
        m = ev.dims.R - 1
        t = ev.dims.T - 1
        left = D[:m, :t]
        right_revneg = -D[N - m:, N - t:][::-1, ::-1]
        assert not np.allclose(left, right_revneg), (
            "left block at psi_L=0.3 must differ from the revneg-pullback "
            "of the right block at psi_R=0.8"
        )


class TestRowSumConsistency:
    """Plan test (b): each row annihilates constants at several
    (psi_L, psi_R) — full-support statement for the closure rows."""

    @pytest.mark.parametrize("psi_L,psi_R", PSI_PAIRS,
                             ids=lambda v: str(v))
    @pytest.mark.parametrize(
        "alpha_vals",
        [[0.0, 0.0, 0.0, 0.0], E2_1_ALPHA_NONZERO],
        ids=["alpha_zero", "alpha_nonzero"],
    )
    def test_row_sums_annihilate_constants(self, e2_evaluator, alpha_vals,
                                           psi_L, psi_R):
        """Interior rows sum to 0; closure rows sum to -wall_weight (the
        constant function has value 1 at the wall, so the full-support
        sum — assembled grid sum + wall weight — vanishes exactly)."""
        ev = e2_evaluator
        alpha = dict(zip(ev.alpha_symbols, alpha_vals))
        N = 14
        D = build_interval_operator(E2_1, alpha, psi_L, psi_R, N)
        m = ev.dims.R - 1
        row_sums = D.sum(axis=1)

        # Interior rows annihilate constants outright.
        np.testing.assert_allclose(
            row_sums[m:N - m], 0.0, atol=1e-12,
            err_msg="interior row sums must vanish",
        )
        # Closure rows annihilate constants over their FULL support
        # (grid nodes + wall): assembled sum + wall weight = 0.
        wall_L = _wall_weights(ev, psi_L, alpha)
        wall_R = _wall_weights(ev, psi_R, alpha)
        np.testing.assert_allclose(
            row_sums[:m] + wall_L, 0.0, atol=1e-10,
            err_msg="left closure full-support row sums must vanish",
        )
        # Right closure rows are the revneg image: row N-1-i carries
        # -row_i, and its wall weight is -wall_R[i].
        np.testing.assert_allclose(
            row_sums[N - 1 - np.arange(m)] - wall_R, 0.0, atol=1e-10,
            err_msg="right closure full-support row sums must vanish",
        )

    def test_bare_closure_row_sums_are_wall_coupling(self, e2_evaluator,
                                                     e2_alpha_zero):
        """Documents why the bare grid row sums do NOT vanish for cut
        closures: they equal minus the (generically nonzero) wall weight
        that derivative.cpp strips into CSR B."""
        ev = e2_evaluator
        N = 12
        D = build_interval_operator(E2_1, e2_alpha_zero, 0.3, 0.3, N)
        wall = _wall_weights(ev, 0.3, e2_alpha_zero)
        assert np.any(np.abs(wall) > 0.1), "wall coupling is nonzero"
        m = ev.dims.R - 1
        np.testing.assert_allclose(D[:m].sum(axis=1), -wall, atol=1e-10)


class TestInteriorRows:
    """Plan test (c): for large N the interior rows equal the circulant
    stencil."""

    def test_interior_rows_equal_circulant(self, e2_evaluator,
                                           e2_alpha_zero):
        ev = e2_evaluator
        N = 60
        D = build_interval_operator(E2_1, e2_alpha_zero, 0.4, 0.7, N)
        iw, io = ev.interior()
        m = ev.dims.R - 1
        for i in range(m, N - m):
            expected = np.zeros(N)
            expected[i + io] = iw
            np.testing.assert_allclose(
                D[i], expected, atol=1e-14,
                err_msg=f"interior row {i} != circulant stencil",
            )
        assert N - 2 * m > 0, "test must actually cover interior rows"


class TestMinimumN:
    """Plan test (d): smallest assemblable N = m_L + m_R (all-closure)
    assembles consistently; below it IntervalTooShort with a precise
    message."""

    def test_all_closure_minimum_assembles(self, e2_evaluator,
                                           e2_symbolic_dirichlet):
        """N = m_L + m_R = 6 for E2_1: no interior row, all-closure —
        exactly the dangerous case — still assembles and matches the
        independent symbolic assembly."""
        cc, psi_sym = e2_symbolic_dirichlet
        ev = e2_evaluator
        alpha = dict(zip(ev.alpha_symbols, E2_1_ALPHA_NONZERO))
        m = ev.dims.R - 1
        N = 2 * m
        assert N == 6
        D = build_interval_operator(E2_1, alpha, 0.5, 0.5, N)
        D_ref = _symbolic_interval_operator(
            cc.dirichlet, psi_sym, E2_1, alpha, 0.5, 0.5, N
        )
        np.testing.assert_allclose(D, D_ref, atol=1e-12)
        # Full-support row-sum consistency survives at the minimum N.
        wall = _wall_weights(ev, 0.5, alpha)
        row_sums = D.sum(axis=1)
        np.testing.assert_allclose(row_sums[:m] + wall, 0.0, atol=1e-10)
        np.testing.assert_allclose(
            row_sums[N - 1 - np.arange(m)] - wall, 0.0, atol=1e-10
        )

    def test_below_minimum_raises_interval_too_short(self, e2_alpha_zero):
        with pytest.raises(IntervalTooShort,
                           match=r"closure rows overlap.*m_L \+ m_R = 6"):
            build_interval_operator(E2_1, e2_alpha_zero, 0.5, 0.5, 5)

    def test_interval_too_short_is_value_error(self, e2_alpha_zero):
        """IntervalTooShort subclasses ValueError so generic callers can
        catch it without importing this module."""
        with pytest.raises(ValueError):
            build_interval_operator(E2_1, e2_alpha_zero, 0.5, 0.5, 2)


class TestValidation:
    """Argument validation."""

    @pytest.mark.parametrize("bad_psi", [0.0, -0.1, 1.5])
    def test_psi_out_of_range(self, e2_alpha_zero, bad_psi):
        with pytest.raises(ValueError, match="psi_L"):
            build_interval_operator(E2_1, e2_alpha_zero, bad_psi, 0.5, 12)
        with pytest.raises(ValueError, match="psi_R"):
            build_interval_operator(E2_1, e2_alpha_zero, 0.5, bad_psi, 12)

    def test_nonpositive_N(self, e2_alpha_zero):
        with pytest.raises(ValueError, match="positive"):
            build_interval_operator(E2_1, e2_alpha_zero, 0.5, 0.5, 0)

    def test_missing_alpha_raises(self):
        with pytest.raises(KeyError, match="alpha"):
            build_interval_operator(E2_1, {"alpha_0": 0.0}, 0.5, 0.5, 12)


@pytest.mark.slow
class TestE4_1IntervalOperator:
    """E4_1 variant — symbolic derivation > 170 s, slow-marked (plan 48
    scheme-scope constraint)."""

    @pytest.fixture(scope="class")
    def e4_evaluator(self):
        return make_cutcell_evaluator(E4_1)

    @pytest.fixture(scope="class")
    def e4_alpha_zero(self, e4_evaluator):
        """The alpha = 0 baseline (feasible across the clamped psi range
        per tests/test_cutcell_eval.py)."""
        return {s: 0.0 for s in e4_evaluator.alpha_symbols}

    def test_structure_and_symbolic_match(self, e4_evaluator,
                                          e4_alpha_zero):
        """E4_1 interval operator matches direct symbolic substitution
        into the evaluator's own (Mathematica-path) Dirichlet block."""
        ev = e4_evaluator
        assert ev.dims.R == 5 and ev.dims.T == 7
        psi_L, psi_R, N = 0.3, 0.7, 16
        D = build_interval_operator(E4_1, e4_alpha_zero, psi_L, psi_R, N)
        D_ref = _symbolic_interval_operator(
            ev._block, ev._psi_symbol, E4_1, e4_alpha_zero, psi_L, psi_R, N
        )
        np.testing.assert_allclose(D, D_ref, atol=1e-10)

    def test_row_sum_consistency(self, e4_evaluator, e4_alpha_zero):
        """Full-support row-sum consistency (plan test b) for E4_1."""
        ev = e4_evaluator
        psi_L, psi_R, N = 0.2, 0.9, 20
        D = build_interval_operator(E4_1, e4_alpha_zero, psi_L, psi_R, N)
        m = ev.dims.R - 1
        row_sums = D.sum(axis=1)
        np.testing.assert_allclose(row_sums[m:N - m], 0.0, atol=1e-12)
        wall_L = _wall_weights(ev, psi_L, e4_alpha_zero)
        wall_R = _wall_weights(ev, psi_R, e4_alpha_zero)
        np.testing.assert_allclose(row_sums[:m] + wall_L, 0.0, atol=1e-9)
        np.testing.assert_allclose(
            row_sums[N - 1 - np.arange(m)] - wall_R, 0.0, atol=1e-9
        )

    def test_interior_rows_equal_circulant(self, e4_evaluator,
                                           e4_alpha_zero):
        """Plan test (c) for E4_1: interior rows carry the E4 circulant."""
        ev = e4_evaluator
        N = 24
        D = build_interval_operator(E4_1, e4_alpha_zero, 0.5, 0.5, N)
        iw, io = ev.interior()
        m = ev.dims.R - 1
        for i in range(m, N - m):
            expected = np.zeros(N)
            expected[i + io] = iw
            np.testing.assert_allclose(D[i], expected, atol=1e-14)

    def test_minimum_N(self, e4_evaluator, e4_alpha_zero):
        """Plan test (d) for E4_1: minimum N = m_L + m_R = 8 assembles;
        7 raises IntervalTooShort."""
        m = e4_evaluator.dims.R - 1
        assert 2 * m == 8
        D = build_interval_operator(E4_1, e4_alpha_zero, 0.5, 0.5, 8)
        assert D.shape == (8, 8)
        assert np.all(np.isfinite(D))
        with pytest.raises(IntervalTooShort, match="m_L \\+ m_R = 8"):
            build_interval_operator(E4_1, e4_alpha_zero, 0.5, 0.5, 7)
