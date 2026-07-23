"""Tests for the cached lambdified cut-cell evaluator (plan 48 item 1.1)
and the psi-parameterized Kreiss check built on it (plan 48 item 1.2).

Includes the e4-align A2 stability money test: the corrected 4-parameter
E4_1 paper construction at the published id=1 alphas
(``tests/fixtures/e4_1_paper_reference.json``) is GKS-stable and
dense-eigenvalue-stable across psi, with the known interior branch-point
artifact at s ~ 1.1760 adjudicated by the deflated winding number.
"""

import json
import pathlib
import time

import numpy as np
import pytest
from sympy import Rational, Symbol

from stencil_gen.cutcell_eval import CutCellEvaluator, make_cutcell_evaluator
from stencil_gen.cutcell_normal_modes import (
    CutCellInfeasible,
    CutCellPsiSweep,
    _validate_closure,
    _vanishing_denominators,
    default_psi_grid,
    kreiss_cutcell_check,
    kreiss_margin_over_psi,
)
from stencil_gen.gks_kreiss import KreissResult
from stencil_gen.temo import (
    E2_1,
    E4_1,
    SchemeParams,
    derive_uniform_boundary_for_temo,
)

# Nonzero, non-singular alpha point for E2_1 (denominator psi + 1 never
# vanishes on (0, 1]); breaks the alpha = 0 degeneracy where all three
# uniform E2_1 boundary rows coincide.
E2_1_ALPHA_NONZERO = [0.3, -0.2, 0.15, 0.05]

# Machine-verified paper ground truth for the E4_1 cut-cell closure
# (Brady & Livescu 2021, JCP 426:109794), committed by e4-align A0.
PAPER_FIXTURE = (
    pathlib.Path(__file__).parent / "fixtures" / "e4_1_paper_reference.json"
)

# Interior kappa-collision (branch point) of the E4 centered stencil on the
# positive real s-axis: the two admissible roots coincide there, so the
# item-0.5 defective-kappa guard fires a *conservative* fail with
# witness_sigma_min = inf.  This is a property of the INTERIOR stencil, not
# of any boundary closure; verdicts with this witness are adjudicated by
# the deflated winding number below (the approach used in the E4_1 mis-port
# investigation).
E4_BRANCH_POINT_S = 1.1760072966846


def _load_paper_fixture():
    with open(PAPER_FIXTURE) as f:
        return json.load(f)


def _paper_id1_alpha(fix):
    """Map the fixture's paper alpha names onto the canonical evaluator
    order alpha_0..alpha_3 = (alphau_04, alphau_14, alphau_24, alphau_25)
    — the documented C++ ``alpha[]`` contract."""
    a = fix["alpha_id1"]
    return {
        "alpha_0": a["alphau_0_4"],
        "alpha_1": a["alphau_1_4"],
        "alpha_2": a["alphau_2_4"],
        "alpha_3": a["alphau_2_5"],
    }


def _deflated_winding(iw, io, rows, row_indices,
                      eps=0.02, rect_r=None, rect_y=15.0, n_pts=4000):
    """Winding number of the deflated Kreiss determinant on an RHP rectangle.

    G(s) = (det M(s) / (kappa_1 - kappa_2))^2 is swap-invariant in the two
    admissible interior roots and finite through the interior branch point,
    so its winding on the rectangle [eps, rect_r] x [-rect_y, rect_y]
    counts genuine RHP zeros of the Kreiss determinant (each zero of det M
    is a double zero of G, hence the /2).  This is the adjudicator used in
    the E4_1 mis-port investigation for verdicts whose witness sits on the
    interior kappa-collision at s ~ 1.1760 (E4_BRANCH_POINT_S), where the
    pointwise sigma_min check is conservative-by-design.

    Returns ``(W, max_dphase)``; ``W is None`` flags an under-resolved
    contour (a phase step > 2.5 rad), which must fail the caller.
    """
    from stencil_gen.gks_kreiss import (
        eigensolution_s_bound,
        kappa_roots,
        kreiss_matrix_general,
    )

    if rect_r is None:
        rect_r = eigensolution_s_bound(iw, rows) * 1.05

    def g_phase(s):
        M = kreiss_matrix_general(iw, io, rows, s, row_indices=row_indices)
        _, adm, defective = kappa_roots(iw, io, s)
        if defective or len(adm) != 2:
            raise RuntimeError(f"defective interior roots on contour at {s}")
        k1, k2 = adm
        with np.errstate(divide="ignore", invalid="ignore"):
            sign, _logdet = np.linalg.slogdet(M)
        if sign == 0 or not np.isfinite(sign):
            raise RuntimeError(
                f"Kreiss determinant collapsed on contour at {s}"
            )
        return 2.0 * (np.angle(sign) - np.angle(k1 - k2))

    pts = []
    pts += [complex(x, -rect_y) for x in np.linspace(eps, rect_r, n_pts // 4)]
    pts += [complex(rect_r, y) for y in np.linspace(-rect_y, rect_y, n_pts // 4)]
    pts += [complex(x, rect_y) for x in np.linspace(rect_r, eps, n_pts // 4)]
    pts += [complex(eps, y) for y in np.linspace(rect_y, -rect_y, n_pts)]
    phases = np.array([g_phase(s) for s in pts])
    d = np.diff(phases)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    max_dphase = float(np.max(np.abs(d)))
    if max_dphase > 2.5:
        return None, max_dphase
    return float(d.sum() / (2 * np.pi) / 2.0), max_dphase


def _e4_dense_dirichlet_operator(rows, iw, io, right_block, n=60):
    """Assemble the N x N dense operator of the mis-port investigation.

    Homogeneous Dirichlet at the left wall: the evaluator's 4
    wall-eliminated closure rows sit at nodes 0..3.  Centered interior on
    nodes 4..n-6.  Outflow at the right: the reflected 5 x 7 *floating*
    closure ``right_block`` (sign-flipped for nu = 1) occupies nodes
    n-5..n-1, its wall row included — no BC is imposed at the outflow end.
    Returns Q = -D so stability is max Re eig(Q) < 0 for u_t = Q u.
    """
    O = np.zeros((n, n))
    for i, (w, cols) in enumerate(rows):
        O[i, cols] = w
    for i in range(len(rows), n - 5):
        O[i, i + io] = iw
    for i in range(5):
        for j in range(7):
            O[n - 1 - i, n - 1 - j] = -right_block[i, j]
    return -O


@pytest.fixture(scope="module")
def e2_evaluator():
    """Module-cached E2_1 evaluator (lru_cache makes this near-free)."""
    return make_cutcell_evaluator(E2_1)


class TestMakeCutcellEvaluator:
    """Caching and dispatch behaviour of make_cutcell_evaluator."""

    def test_cache_hit_returns_same_object(self, e2_evaluator):
        """Second make_cutcell_evaluator(E2_1) returns the same object (plan test b)."""
        assert make_cutcell_evaluator(E2_1) is e2_evaluator

    def test_cache_keyed_on_scheme_value(self, e2_evaluator):
        """An equal (but distinct) SchemeParams instance hits the same cache slot."""
        clone = SchemeParams(p=1, q=1, s=0, nextra=1, nu=1)
        assert clone is not E2_1
        assert make_cutcell_evaluator(clone) is e2_evaluator

    def test_scheme_params_hashable(self):
        """SchemeParams is frozen/hashable — the lru_cache prerequisite (plan 48 risk table)."""
        assert hash(E2_1) == hash(SchemeParams(p=1, q=1, s=0, nextra=1, nu=1))
        assert hash(E4_1) == hash(
            SchemeParams(p=2, q=3, s=0, nextra=0, nu=1, zeros=(3, 4))
        )

    def test_non_dirichlet_bc_rejected(self):
        """The floating block is never analyzed for stability (plan 48 constraint)."""
        with pytest.raises(ValueError, match="[Dd]irichlet"):
            make_cutcell_evaluator(E2_1, bc="floating")


class TestE2_1Evaluator:
    """Structure and correctness of the E2_1 Dirichlet-row evaluator."""

    def test_dims_and_alpha_symbols(self, e2_evaluator):
        ev = e2_evaluator
        assert isinstance(ev, CutCellEvaluator)
        assert ev.dims.R == 4 and ev.dims.T == 5
        assert [s.name for s in ev.alpha_symbols] == [
            "alpha_0", "alpha_1", "alpha_2", "alpha_3",
        ]

    def test_rows_structure(self, e2_evaluator):
        """R-1 rows, wall column dropped, offsets 0..T-2, row_indices 0..R-2."""
        ev = e2_evaluator
        alpha = {s: 0.0 for s in ev.alpha_symbols}
        rows, row_indices = ev.rows(0.5, alpha)
        R, T = ev.dims.R, ev.dims.T
        assert len(rows) == R - 1
        np.testing.assert_array_equal(row_indices, np.arange(R - 1))
        for weights, col_offsets in rows:
            assert weights.shape == (T - 1,)
            assert weights.dtype == np.float64
            np.testing.assert_array_equal(col_offsets, np.arange(T - 1))

    @pytest.mark.parametrize(
        "alpha_vals",
        [[0.0, 0.0, 0.0, 0.0], E2_1_ALPHA_NONZERO],
        ids=["alpha_zero", "alpha_nonzero"],
    )
    def test_psi1_matches_uniform_rows(self, e2_evaluator, alpha_vals):
        """psi = 1 identity: rows(1.0, alpha) matches the uniform boundary rows.

        Reuses the construction of tests/test_group_velocity.py's
        ``test_psi_1_matches_uniform``: by TEMO design the floating block
        reduces to the uniform boundary B_u at psi = 1 (wall coefficient
        -> 0).  The Dirichlet block is floating rows 1..R-1, so Dirichlet
        row i corresponds to B_u row i+1 for i = 0..r-2 (verified
        symbolically during implementation).  The last Dirichlet row is
        the near-interior cut-cell row, which by construction retains a
        nonzero wall coefficient even at psi = 1 (the uniform-limit
        stencil B_l(1) couples to node -1) — it has no B_u counterpart
        and is excluded from the identity.
        """
        ev = e2_evaluator
        alpha = dict(zip(ev.alpha_symbols, alpha_vals))
        rows, _ = ev.rows(1.0, alpha)

        ur = derive_uniform_boundary_for_temo(E2_1)
        umap = dict(zip(ur.alpha_symbols, alpha_vals))
        r, t = ur.B_u.shape
        assert ev.dims.T - 1 == t  # dropped-wall rows have uniform width

        for i in range(r - 1):
            weights, _ = rows[i]
            expected = np.array(
                [float(ur.B_u[i + 1, j].xreplace(umap)) for j in range(t)]
            )
            np.testing.assert_allclose(
                weights, expected, atol=1e-10,
                err_msg=f"Dirichlet row {i} != uniform B_u row {i + 1} at psi=1",
            )

    def test_rows_match_symbolic_substitution(self, e2_evaluator):
        """Lambdified entries agree with direct sympy substitution."""
        ev = e2_evaluator
        psi_val = 0.37
        alpha = dict(zip(ev.alpha_symbols, E2_1_ALPHA_NONZERO))
        rows, _ = ev.rows(psi_val, alpha)

        subs = {ev._psi_symbol: psi_val}
        subs.update(alpha)
        T = ev.dims.T
        for i, (weights, _) in enumerate(rows):
            expected = np.array(
                [float(ev._block[i, j].xreplace(subs)) for j in range(1, T)]
            )
            np.testing.assert_allclose(weights, expected, rtol=1e-12)

    def test_denominators(self, e2_evaluator):
        """Denominators are collected as non-constant sympy factors.

        E2_1's Dirichlet block has the single structural denominator
        psi + 1, which never vanishes on (0, 1] — the closure is feasible
        across the whole psi range.
        """
        ev = e2_evaluator
        assert isinstance(ev.denominators, frozenset)
        assert Symbol("psi") + 1 in ev.denominators
        for den in ev.denominators:
            assert den.free_symbols, "constant factors must be discarded"

    def test_interior(self, e2_evaluator):
        """interior() returns the E2 central stencil from interior.py."""
        weights, offsets = e2_evaluator.interior()
        np.testing.assert_allclose(weights, [-0.5, 0.0, 0.5])
        np.testing.assert_array_equal(offsets, [-1, 0, 1])

    def test_alpha_by_name(self, e2_evaluator):
        """Alpha dict may be keyed by symbol name strings."""
        ev = e2_evaluator
        by_sym = {s: v for s, v in zip(ev.alpha_symbols, E2_1_ALPHA_NONZERO)}
        by_name = {s.name: v for s, v in zip(ev.alpha_symbols, E2_1_ALPHA_NONZERO)}
        rows_sym, _ = ev.rows(0.6, by_sym)
        rows_name, _ = ev.rows(0.6, by_name)
        for (w_a, _), (w_b, _) in zip(rows_sym, rows_name):
            np.testing.assert_array_equal(w_a, w_b)

    def test_missing_alpha_raises(self, e2_evaluator):
        ev = e2_evaluator
        incomplete = {ev.alpha_symbols[0]: 0.0}
        with pytest.raises(KeyError, match="alpha"):
            ev.rows(0.5, incomplete)

    def test_unknown_alpha_raises(self, e2_evaluator):
        ev = e2_evaluator
        alpha = {s: 0.0 for s in ev.alpha_symbols}
        alpha["alpha_99"] = 1.0
        with pytest.raises(ValueError, match="alpha_99"):
            ev.rows(0.5, alpha)

    def test_20_psi_evaluation_fast(self, e2_evaluator):
        """20-psi evaluation after warm cache completes in < 0.1 s (plan test c)."""
        ev = e2_evaluator
        alpha = {s: 0.0 for s in ev.alpha_symbols}
        ev.rows(0.5, alpha)  # warm any lazy paths
        t0 = time.perf_counter()
        for psi in np.linspace(0.05, 0.9999, 20):
            ev.rows(float(psi), alpha)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.1, f"20-psi evaluation took {elapsed:.4f} s"


@pytest.mark.slow
class TestE4_1Evaluator:
    """E4_1 variant (plan test d) — symbolic derivation > 170 s, slow-marked.

    E4_1 has ``zeros=(3, 4)`` so make_cutcell_evaluator dispatches to
    ``derive_cut_cell_paper`` (the published Brady & Livescu 2021
    construction, exactly as ``psi_sweep_group_velocity`` does), whose
    parameterization keeps the four uniform-boundary alphas free.
    """

    @pytest.fixture(scope="class")
    def e4_evaluator(self):
        return make_cutcell_evaluator(E4_1)

    # Nonzero, non-singular point (both denominator factors are O(1) here;
    # alpha = 0 is also feasible across the full psi grid).
    E4_ALPHA = [0.1, 0.05, -0.02, 0.01]

    def test_structure(self, e4_evaluator):
        ev = e4_evaluator
        assert ev.dims.R == 5 and ev.dims.T == 7
        assert [s.name for s in ev.alpha_symbols] == [
            "alpha_0", "alpha_1", "alpha_2", "alpha_3",
        ]
        alpha = dict(zip(ev.alpha_symbols, self.E4_ALPHA))
        rows, row_indices = ev.rows(0.5, alpha)
        assert len(rows) == 4
        np.testing.assert_array_equal(row_indices, np.arange(4))
        for weights, col_offsets in rows:
            assert weights.shape == (6,)
            assert np.all(np.isfinite(weights))
            np.testing.assert_array_equal(col_offsets, np.arange(6))

    def test_alpha_zero_feasible_across_psi(self, e4_evaluator):
        """The alpha = 0 baseline evaluates finite on the clamped psi range."""
        ev = e4_evaluator
        alpha = {s: 0.0 for s in ev.alpha_symbols}
        for psi in [1e-4, 0.01, 0.1, 0.5, 0.9, 0.9999, 1.0]:
            rows, _ = ev.rows(psi, alpha)
            for weights, _ in rows:
                assert np.all(np.isfinite(weights)), f"non-finite at psi={psi}"

    def test_cache_hit(self, e4_evaluator):
        assert make_cutcell_evaluator(E4_1) is e4_evaluator

    def test_rows_match_symbolic_substitution(self, e4_evaluator):
        ev = e4_evaluator
        psi_val = 0.3
        alpha = dict(zip(ev.alpha_symbols, self.E4_ALPHA))
        rows, _ = ev.rows(psi_val, alpha)
        subs = {ev._psi_symbol: psi_val}
        subs.update(alpha)
        T = ev.dims.T
        for i, (weights, _) in enumerate(rows):
            expected = np.array(
                [float(ev._block[i, j].xreplace(subs)) for j in range(1, T)]
            )
            np.testing.assert_allclose(weights, expected, rtol=1e-10)

    def test_denominators_psi_dependent(self, e4_evaluator):
        """E4_1 has psi-dependent structural denominators (item 1.4 input)."""
        ev = e4_evaluator
        psi = Symbol("psi")
        assert len(ev.denominators) > 0
        assert any(psi in den.free_symbols for den in ev.denominators)

    def test_denominator_structure(self, e4_evaluator):
        """The corrected 4-param closure's denominators (e4-align A2).

        Exactly five irreducible factors:

        - ``psi + 1``, ``psi + 2``, ``psi + 3`` — pure-psi Vandermonde
          factors of the f0 row; no zeros on (0, 1];
        - ``66*alpha_0 - 73`` — psi-INDEPENDENT uniform-conservation
          factor (f2 row); pole only at alpha_0 = 73/66;
        - one mixed cut-conservation polynomial in all of (psi, alpha_0..3)
          of degree 8 in psi (transition row).

        This replaces the dead mis-port lore (the D6 denominator and the
        C++ ``alpha[1] >= 197/288`` guard): the corrected closure has NO
        psi-pole at fixed feasible alpha — infeasibility is an alpha-driven
        surface, tested pointwise by the feasibility guard.
        """
        from sympy import Poly

        ev = e4_evaluator
        psi = Symbol("psi")
        a0 = Symbol("alpha_0")
        dens = ev.denominators
        assert len(dens) == 5
        assert {psi + 1, psi + 2, psi + 3} <= dens
        assert 66 * a0 - 73 in dens

        mixed = [d for d in dens if len(d.free_symbols) == 5]
        assert len(mixed) == 1
        assert Poly(mixed[0], psi).degree() == 8

        # No denominator vanishes on the C++ clamp range [1e-4, 1 - 1e-4]
        # at either the alpha = 0 baseline or the paper id=1 alphas.
        from sympy import lambdify

        id1 = _paper_id1_alpha(_load_paper_fixture())
        grid = np.linspace(1e-4, 1.0 - 1e-4, 4001)
        for avals in ([0.0] * 4, [id1[s.name] for s in ev.alpha_symbols]):
            for den in dens:
                f = lambdify((psi, *ev.alpha_symbols), den)
                v = np.broadcast_to(
                    np.asarray(f(grid, *avals), dtype=float), grid.shape
                )
                assert np.all(np.abs(v) > 1.0), f"near-zero of {den}"
                assert np.all(np.sign(v) == np.sign(v[0])), f"zero of {den}"

    def test_alpha_order_matches_paper_fixture(self, e4_evaluator):
        """Pins the canonical alpha order to the paper order (e4-align A2).

        Feeding the fixture id=1 values through the documented mapping
        alpha_0..alpha_3 = (alphau_04, alphau_14, alphau_24, alphau_25)
        must reproduce the fixture's wall-eliminated Dirichlet block at
        every fixture psi — any permutation of the order breaks this.
        The same order is the C++ ``alpha[]`` contract.
        """
        ev = e4_evaluator
        fix = _load_paper_fixture()
        alpha = _paper_id1_alpha(fix)
        for key in ["0.05", "0.3", "0.7", "0.97"]:
            ref = np.array(fix["blocks"][key]["dirichlet_4x6_wall_row_and_col_dropped"])
            rows, row_indices = ev.rows(float(key), alpha)
            got = np.array([w for w, _ in rows])
            np.testing.assert_allclose(
                got, ref, atol=1e-12,
                err_msg=f"Dirichlet block mismatch vs paper fixture at psi={key}",
            )
            np.testing.assert_array_equal(row_indices, np.arange(4))

    def test_interior(self, e4_evaluator):
        weights, offsets = e4_evaluator.interior()
        np.testing.assert_allclose(
            weights, [1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12]
        )
        np.testing.assert_array_equal(offsets, [-2, -1, 0, 1, 2])

    def test_evaluation_fast_after_warm(self, e4_evaluator):
        """20-psi evaluation is < 0.1 s once the derivation is cached."""
        ev = e4_evaluator
        alpha = dict(zip(ev.alpha_symbols, self.E4_ALPHA))
        ev.rows(0.5, alpha)
        t0 = time.perf_counter()
        for psi in np.linspace(0.05, 0.9999, 20):
            ev.rows(float(psi), alpha)
        assert time.perf_counter() - t0 < 0.1


# ---------------------------------------------------------------------------
# Plan 48 item 1.2: psi-parameterized Kreiss check
# ---------------------------------------------------------------------------

# Matches the item-0.3/0.5 calibration grid (odd n_imag puts Im = 0 on the
# grid).
_CUT_GRID = {"s_max": 10.0, "n_radial": 30, "n_imag": 81, "imag_max": 15.0}

# The plan-specified psi test set for the E2_1 alpha=0 baseline.
_CUT_PSI = [1e-3, 0.1, 0.5, 0.9, 0.9999]


def _halfline_operator(ev, alpha, psi, n=40):
    """Assemble the N x N cut-cell half-line operator at *psi*.

    Test-local assembly per plan 48 item 1.2 (the production interval
    builder arrives in item 3.1): mirrors
    tests/test_group_velocity.py's ``_build_cut_cell_diff_matrix`` — cut
    Dirichlet closure rows at the left (wall column already eliminated by
    the evaluator), centered interior, reflected uniform tension-sigma=0
    closure at the right (known stable, so any instability is the left
    closure's).
    """
    from stencil_gen.phs import uniform_boundary_weights_rbf

    iw, io = ev.interior()
    rows, _ = ev.rows(psi, alpha)
    r = len(rows)
    t = ev.dims.t
    D = np.zeros((n, n))
    for i, (w, cols) in enumerate(rows):
        D[i, cols] = w
    for i in range(r, n - r):
        D[i, i + io] = iw
    sign = (-1.0) ** ev.scheme.nu
    for i in range(r):
        w = uniform_boundary_weights_rbf(
            i, t, ev.scheme.nu, ev.scheme.q, 0.0, kernel="tension"
        )
        for j in range(t):
            D[n - 1 - i, n - 1 - j] = sign * float(w[j])
    return D


class TestKreissCutcellCheck:
    """Plan 48 item 1.2: kreiss_cutcell_check on the E2_1 alpha=0 baseline."""

    @pytest.fixture()
    def alpha_zero(self, e2_evaluator):
        return {s: 0.0 for s in e2_evaluator.alpha_symbols}

    @pytest.mark.parametrize("psi", _CUT_PSI)
    def test_e2_1_alpha0_stable(self, alpha_zero, psi):
        """E2_1 at alpha = 0 (the plan-35 baseline configuration) is
        GKS-stable across the psi range, including the near-wall
        (psi -> 0) and row-coincidence (psi -> 1) regimes."""
        result = kreiss_cutcell_check(
            E2_1, alpha_zero, psi, s_grid_params=_CUT_GRID
        )
        assert isinstance(result, KreissResult)
        assert not getattr(result, "infeasible", False)
        assert result.is_stable is True, (
            f"false-unstable at psi={psi}: verdict "
            f"{result.imaginary_axis_perturbation_verdict!r} / "
            f"{result.near_origin_verdict!r}, witness {result.witness_s}, "
            f"sigma {result.witness_sigma_min:.3e}"
        )
        assert result.defective_kappa_detected is False

    def test_dense_eigenvalue_cross_check(self, e2_evaluator, alpha_zero):
        """Kreiss verdict agrees with dense eigenvalues of the assembled
        half-line operator at N = 40 (plan 48 item 1.2 cross-check;
        semi-discrete u_t = -D u, so stability is max Re eig(-D) <~ 0)."""
        psi = 0.5
        result = kreiss_cutcell_check(
            E2_1, alpha_zero, psi, s_grid_params=_CUT_GRID
        )
        assert result.is_stable is True

        D = _halfline_operator(e2_evaluator, alpha_zero, psi, n=40)
        max_re = float(np.max(np.real(np.linalg.eigvals(-D))))
        assert max_re < 1e-5, (
            f"dense eigenvalues contradict GKS verdict at psi={psi}: "
            f"max Re = {max_re:.3e}"
        )

    def test_validation_dims_e2_1(self, e2_evaluator, alpha_zero):
        """E2_1 dimension contract: m = R - 1 = 3 rows = 2 nodal unknowns
        + p = 1 admissible root (plan 48 item 1.2 validation)."""
        rows, _ = e2_evaluator.rows(0.5, alpha_zero)
        _, io = e2_evaluator.interior()
        assert _validate_closure(e2_evaluator, rows, io) == (3, 1, 2)

        result = kreiss_cutcell_check(
            E2_1, alpha_zero, 0.5, s_grid_params=_CUT_GRID
        )
        assert result.n_admissible_roots == 1

    def test_validation_rejects_row_count_mismatch(self, e2_evaluator,
                                                   alpha_zero):
        """A closure with the wrong row count fails the m = R - 1 check."""
        rows, _ = e2_evaluator.rows(0.5, alpha_zero)
        _, io = e2_evaluator.interior()
        with pytest.raises(ValueError, match="R - 1"):
            _validate_closure(e2_evaluator, rows[:-1], io)

    def test_infeasible_marker(self, e2_evaluator, alpha_zero, monkeypatch):
        """A vanishing denominator returns the distinct CutCellInfeasible
        marker, never a stability verdict (plan 48 item 1.2 feasibility
        guard).

        E2_1's only structural denominator is psi + 1, which never vanishes
        on (0, 1], so an engineered denominator psi - 1/2 is substituted on
        the cached evaluator (monkeypatch restores it; the lambdify cache is
        keyed on the denominator set's value, so the substitute is compiled
        fresh).
        """
        monkeypatch.setattr(
            e2_evaluator, "denominators",
            frozenset({Symbol("psi") - Rational(1, 2)}),
        )
        result = kreiss_cutcell_check(E2_1, alpha_zero, 0.5)
        assert isinstance(result, CutCellInfeasible)
        assert result.infeasible is True
        assert result.psi == 0.5
        assert result.alpha == (
            ("alpha_0", 0.0), ("alpha_1", 0.0),
            ("alpha_2", 0.0), ("alpha_3", 0.0),
        )
        assert len(result.vanishing) == 1
        den, value = result.vanishing[0]
        assert den == Symbol("psi") - Rational(1, 2)
        assert value == pytest.approx(0.0, abs=1e-12)
        # A parameterization failure is not a stability verdict.
        assert not hasattr(result, "is_stable")

        # Away from the engineered zero the guard passes.
        assert _vanishing_denominators(
            e2_evaluator, 0.9, alpha_zero, 1e-12
        ) == ()

    def test_e2_1_feasible_across_psi(self, e2_evaluator, alpha_zero):
        """E2_1's real denominator (psi + 1) never vanishes on (0, 1]."""
        for psi in _CUT_PSI + [1.0]:
            assert _vanishing_denominators(
                e2_evaluator, psi, alpha_zero, 1e-12
            ) == ()

    def test_row_indices_override_rejected(self, alpha_zero):
        """row_indices comes from the evaluator's Dirichlet row placement
        and may not be overridden through kreiss_kwargs."""
        with pytest.raises(TypeError, match="row_indices"):
            kreiss_cutcell_check(
                E2_1, alpha_zero, 0.5, row_indices=[0, 1, 2]
            )

    @pytest.mark.parametrize("psi", [0.0, -1.0, 1.5])
    def test_psi_out_of_range(self, alpha_zero, psi):
        with pytest.raises(ValueError, match="psi"):
            kreiss_cutcell_check(E2_1, alpha_zero, psi)


# ---------------------------------------------------------------------------
# Plan 48 item 1.3: worst-case-over-psi margin
# ---------------------------------------------------------------------------


class TestKreissMarginOverPsi:
    """Plan 48 item 1.3: kreiss_margin_over_psi / default_psi_grid.

    Test names contain 'margin' or 'psi_sweep' so the plan's focused run
    (``-k "psi_sweep or margin"``) selects them.
    """

    @pytest.fixture()
    def alpha_zero(self, e2_evaluator):
        return {s: 0.0 for s in e2_evaluator.alpha_symbols}

    def test_default_psi_grid_margin_coverage(self):
        """The default grid is the plan-specified 20-point set: near-zero
        cluster ∪ Chebyshev-11 on [0.05, 0.95] ∪ near-one cluster, sorted,
        matched to the C++ clamp [1e-4, 1 - 1e-4]."""
        grid = default_psi_grid()
        assert grid.shape == (20,)
        assert np.all(np.diff(grid) > 0), "grid must be sorted, duplicate-free"
        assert grid[0] == pytest.approx(1e-4)
        assert grid[-1] == pytest.approx(1.0 - 1e-4)

        near_zero = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        cheb = 0.5 + 0.45 * np.cos(np.pi * np.arange(11) / 10)
        near_one = [0.99, 0.999, 0.9999]
        for p in [*near_zero, *cheb, *near_one]:
            assert np.any(np.isclose(grid, p, rtol=0, atol=1e-15)), (
                f"default grid missing plan-specified point {p}"
            )
        # Chebyshev-Lobatto block includes the [0.05, 0.95] endpoints.
        assert np.any(np.isclose(grid, 0.05))
        assert np.any(np.isclose(grid, 0.95))

    def test_margin_e2_1_alpha0_default_grid(self, alpha_zero):
        """E2_1 at alpha = 0 (plan-35 baseline): stable and feasible across
        the full default psi grid; worst-case bookkeeping consistent with
        the per-psi results."""
        sweep = kreiss_margin_over_psi(
            E2_1, alpha_zero, s_grid_params=_CUT_GRID
        )
        assert isinstance(sweep, CutCellPsiSweep)
        assert sweep.any_unstable is False
        assert sweep.infeasible_psi == []

        grid = default_psi_grid()
        assert sorted(sweep.per_psi) == [float(p) for p in grid]
        for psi, res in sweep.per_psi.items():
            assert isinstance(res, KreissResult), f"psi={psi}"
            assert res.is_stable is True, f"false-unstable at psi={psi}"

        # worst_sigma_min / worst_psi are the min/argmin of the per-psi
        # (finite) witness margins.
        finite = {
            p: float(r.witness_sigma_min)
            for p, r in sweep.per_psi.items()
            if np.isfinite(r.witness_sigma_min)
        }
        expected_psi = min(finite, key=finite.get)
        assert sweep.worst_psi == expected_psi
        assert sweep.worst_sigma_min == pytest.approx(finite[expected_psi])
        assert 0.0 < sweep.worst_sigma_min < np.inf
        # The psi -> 1 row-coincidence degeneracy makes the near-one edge
        # the worst case for this closure (measured during item 1.2).
        assert sweep.worst_psi == pytest.approx(1.0 - 1e-4)

    def test_margin_custom_psi_grid(self, alpha_zero):
        """A caller-supplied grid is used verbatim (float-keyed per_psi)."""
        grid = [0.1, 0.5, 0.9]
        sweep = kreiss_margin_over_psi(
            E2_1, alpha_zero, psi_grid=grid, s_grid_params=_CUT_GRID
        )
        assert sorted(sweep.per_psi) == grid
        assert sweep.worst_psi in grid
        assert sweep.any_unstable is False
        assert sweep.infeasible_psi == []

    def test_psi_sweep_partitions_infeasible(self, e2_evaluator, alpha_zero,
                                             monkeypatch):
        """Infeasible psi points are listed in infeasible_psi, carried as
        CutCellInfeasible in per_psi, and excluded from the margin — a
        parameterization failure never folds into worst_sigma_min
        (plan 48 item 1.3, reusing the 1.2 feasibility guard)."""
        monkeypatch.setattr(
            e2_evaluator, "denominators",
            frozenset({Symbol("psi") - Rational(1, 2)}),
        )
        sweep = kreiss_margin_over_psi(
            E2_1, alpha_zero, psi_grid=[0.25, 0.5, 0.75],
            s_grid_params=_CUT_GRID,
        )
        assert sweep.infeasible_psi == [0.5]
        assert isinstance(sweep.per_psi[0.5], CutCellInfeasible)
        assert isinstance(sweep.per_psi[0.25], KreissResult)
        assert isinstance(sweep.per_psi[0.75], KreissResult)
        assert sweep.worst_psi in (0.25, 0.75)
        assert np.isfinite(sweep.worst_sigma_min)
        assert sweep.any_unstable is False

    def test_psi_sweep_all_infeasible(self, e2_evaluator, alpha_zero,
                                      monkeypatch):
        """All-infeasible sweep: no margin (inf / None), not 'unstable'."""
        monkeypatch.setattr(
            e2_evaluator, "denominators",
            frozenset({Symbol("psi") - Rational(1, 2)}),
        )
        sweep = kreiss_margin_over_psi(
            E2_1, alpha_zero, psi_grid=[0.5], s_grid_params=_CUT_GRID
        )
        assert sweep.infeasible_psi == [0.5]
        assert sweep.worst_psi is None
        assert np.isinf(sweep.worst_sigma_min)
        assert sweep.any_unstable is False

    def test_psi_sweep_any_unstable_aggregation(self, alpha_zero, monkeypatch):
        """any_unstable aggregates per-psi verdicts; an anomaly-driven fail
        (witness_sigma_min = inf) flips the flag without polluting the
        finite-margin bookkeeping."""
        import stencil_gen.cutcell_normal_modes as cnm

        def fake_check(scheme, alpha, psi, **kwargs):
            if psi == 0.5:
                # Anomaly-driven conservative fail (item 0.1 shape).
                return KreissResult(
                    is_stable=False,
                    witness_s=0.4 + 0.0j,
                    witness_sigma_min=float("inf"),
                    imaginary_axis_perturbation_verdict="defective",
                    defective_kappa_detected=True,
                )
            return KreissResult(
                is_stable=True,
                witness_s=1e-3j,
                witness_sigma_min=0.1 * psi,
            )

        monkeypatch.setattr(cnm, "kreiss_cutcell_check", fake_check)
        sweep = cnm.kreiss_margin_over_psi(
            E2_1, alpha_zero, psi_grid=[0.25, 0.5, 0.75]
        )
        assert sweep.any_unstable is True
        assert sweep.infeasible_psi == []
        # inf margin at the unstable point does not become the argmin.
        assert sweep.worst_psi == 0.25
        assert sweep.worst_sigma_min == pytest.approx(0.1 * 0.25)

    def test_psi_sweep_empty_grid_raises(self, alpha_zero):
        with pytest.raises(ValueError, match="psi_grid"):
            kreiss_margin_over_psi(E2_1, alpha_zero, psi_grid=[])

    def test_psi_sweep_forwards_kreiss_kwargs(self, alpha_zero):
        """s_grid_params reach kreiss_stability_check through the sweep."""
        sweep = kreiss_margin_over_psi(
            E2_1, alpha_zero, psi_grid=[0.5], s_grid_params=_CUT_GRID
        )
        result = sweep.per_psi[0.5]
        # make_s_grid's shape depends on its parameters; assert the sweep
        # used the non-default grid by comparing against a default-grid run.
        default = kreiss_cutcell_check(E2_1, alpha_zero, 0.5)
        assert result.s_grid_shape != default.s_grid_shape


@pytest.mark.slow
class TestKreissCutcellCheckE4_1:
    """E4_1 variant of the item-1.2 wrapper (symbolic derivation > 170 s).

    Only the plumbing contract is asserted here — dimension validation and
    a feasible (non-infeasible) verdict object.  The E4_1 stability
    calibration itself is plan 48 item 1.4 (findings item), so no
    is_stable value is asserted.
    """

    @pytest.fixture(scope="class")
    def e4_evaluator(self):
        return make_cutcell_evaluator(E4_1)

    def test_dims_and_verdict_object(self, e4_evaluator):
        alpha = {s: 0.0 for s in e4_evaluator.alpha_symbols}
        rows, _ = e4_evaluator.rows(0.5, alpha)
        _, io = e4_evaluator.interior()
        # m = R - 1 = 4 rows = 2 nodal unknowns + p = 2 admissible roots.
        assert _validate_closure(e4_evaluator, rows, io) == (4, 2, 2)

        result = kreiss_cutcell_check(
            E4_1, alpha, 0.5, s_grid_params=_CUT_GRID
        )
        assert isinstance(result, KreissResult)
        assert not getattr(result, "infeasible", False)
        assert result.n_admissible_roots == 2


# ---------------------------------------------------------------------------
# e4-align A2 money test: paper id=1 alphas are stable across psi
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestE4_1PaperId1MoneyTest:
    """Stability of the corrected E4_1 closure at the published id=1 alphas.

    The paper's true id=1 scheme (Appendix B text values; the printed
    Table A.4 is corrupted and NOT used) was verified during the mis-port
    investigation to pass dense eigenvalues through N = 240 and GKS
    winding W = 0 at 14 psi.  These tests assert the repo pipeline now
    reproduces that: (a) the augmented Kreiss check reports stable at the
    fixture psi, with the interior branch-point artifact at
    s ~ E4_BRANCH_POINT_S adjudicated by the deflated winding + a dense
    fallback; (b) dense eigenvalues of the investigation's assembled
    operator at N = 60 have max Re < 0; (c) the default 20-psi sweep runs,
    and every unstable verdict it reports is the adjudicated branch-point
    artifact — no genuine bad-psi pocket.
    """

    FIXTURE_PSI = [0.05, 0.3, 0.7, 0.97]

    @pytest.fixture(scope="class")
    def e4_evaluator(self):
        return make_cutcell_evaluator(E4_1)

    @pytest.fixture(scope="class")
    def paper_fix(self):
        return _load_paper_fixture()

    @pytest.fixture(scope="class")
    def id1_alpha(self, paper_fix):
        return _paper_id1_alpha(paper_fix)

    def _adjudicate_branch_point_verdict(self, ev, alpha, psi, result,
                                         right_block):
        """A not-stable verdict may ONLY be the known interior
        branch-point artifact; adjudicate it via the deflated winding and
        assert stability via the dense fallback."""
        # 1. Artifact signature: conservative defective-kappa fail exactly
        #    at the interior kappa collision, margin inf (not a genuine
        #    determinant zero).
        assert result.defective_kappa_detected is True, (
            f"psi={psi}: unstable verdict without the branch-point "
            f"signature (witness {result.witness_s})"
        )
        assert abs(result.witness_s - E4_BRANCH_POINT_S) < 1e-6, (
            f"psi={psi}: defective-kappa witness {result.witness_s} is not "
            f"the interior branch point"
        )
        assert np.isinf(result.witness_sigma_min)

        # 2. Deflated winding: zero RHP zeros of the Kreiss determinant.
        iw, io = ev.interior()
        rows, row_indices = ev.rows(psi, alpha)
        W, max_dphase = _deflated_winding(iw, io, rows, list(row_indices))
        assert W is not None, (
            f"psi={psi}: winding contour under-resolved "
            f"(max dphase {max_dphase:.2f} rad)"
        )
        assert abs(W) < 1e-6, (
            f"psi={psi}: deflated winding W = {W} != 0 — genuine RHP zero"
        )

        # 3. Dense fallback: the assembled operator is strictly stable.
        Q = _e4_dense_dirichlet_operator(rows, iw, io, right_block, n=60)
        max_re = float(np.max(np.real(np.linalg.eigvals(Q))))
        assert max_re < 0.0, (
            f"psi={psi}: dense fallback contradicts adjudication, "
            f"max Re = {max_re:.3e}"
        )

    @pytest.mark.parametrize("psi", FIXTURE_PSI)
    def test_kreiss_stable_at_fixture_psi(self, e4_evaluator, paper_fix,
                                          id1_alpha, psi):
        """(a) kreiss_cutcell_check (augmented Kreiss matrix, evaluator
        row_indices) at the id=1 alphas: stable at every fixture psi.

        Measured 2026-07-23: psi in {0.05, 0.3, 0.7} report is_stable=True
        directly (finite margins 2.2e-2 / 3.9e-2 / 1.1e-1); at psi = 0.97
        the interior branch-point artifact fires (witness s = 1.17600...,
        defective kappa, sigma = inf) and is adjudicated: winding W = 0,
        dense max Re < 0.
        """
        result = kreiss_cutcell_check(
            E4_1, id1_alpha, psi, s_grid_params=_CUT_GRID
        )
        assert isinstance(result, KreissResult)
        if result.is_stable:
            assert np.isfinite(result.witness_sigma_min)
            assert result.witness_sigma_min > 0.0
            return
        right = np.array(paper_fix["blocks"][repr(psi)]["floating_5x7"])
        self._adjudicate_branch_point_verdict(
            e4_evaluator, id1_alpha, psi, result, right
        )

    @pytest.mark.parametrize("psi", FIXTURE_PSI)
    def test_dense_eigenvalues_at_fixture_psi(self, e4_evaluator, paper_fix,
                                              id1_alpha, psi):
        """(b) Dense eigenvalues of the assembled operator at N = 60:
        max Re lambda < 0 at every fixture psi (the investigation's
        acceptance assembly — left cut Dirichlet closure under test, right
        reflected floating closure from the committed fixture).

        Measured 2026-07-23: max Re in [-1.5e-4, -1.6e-5].
        """
        rows, _ = e4_evaluator.rows(psi, id1_alpha)
        iw, io = e4_evaluator.interior()
        right = np.array(paper_fix["blocks"][repr(psi)]["floating_5x7"])
        Q = _e4_dense_dirichlet_operator(rows, iw, io, right, n=60)
        max_re = float(np.max(np.real(np.linalg.eigvals(Q))))
        assert max_re < -1e-6, (
            f"psi={psi}: N=60 dense spectrum not strictly stable, "
            f"max Re = {max_re:.3e}"
        )

    def test_psi_sweep_default_grid_verdict(self, e4_evaluator, paper_fix,
                                            id1_alpha):
        """(c) kreiss_margin_over_psi on the default 20-psi grid.

        RECORDED VERDICT (2026-07-23, faithfully): the sweep returns
        ``any_unstable=True`` — but every one of the 6 unstable per-psi
        verdicts (the near-one grid points psi >= 0.864) carries the
        identical interior branch-point witness s = 1.1760... with
        sigma = inf and defective kappa, and every one adjudicates clean
        (deflated winding W = 0, dense N = 60 max Re < 0).  The remaining
        14 psi are stable with finite margins; the worst finite margin is
        2.10e-2 at psi = 0.0720 (near-wall regime, as the ILW literature
        predicts).  No genuine bad-psi pocket was found — the raw
        any_unstable flag is entirely the adjudicated artifact.
        """
        sweep = kreiss_margin_over_psi(
            E4_1, id1_alpha, s_grid_params=_CUT_GRID
        )
        assert isinstance(sweep, CutCellPsiSweep)
        assert sweep.infeasible_psi == []
        grid = default_psi_grid()
        assert sorted(sweep.per_psi) == [float(p) for p in grid]
        for psi, res in sweep.per_psi.items():
            assert isinstance(res, KreissResult), f"psi={psi}"

        # Adjudicate every unstable verdict; none may be genuine.
        uniform_right = np.array(paper_fix["blocks"]["1.0"]["floating_5x7"])
        unstable = {p: r for p, r in sweep.per_psi.items() if not r.is_stable}
        for psi, res in unstable.items():
            self._adjudicate_branch_point_verdict(
                e4_evaluator, id1_alpha, psi, res, uniform_right
            )

        # Recorded-verdict regression anchors (loose bands around the
        # measured values; a drift outside them is a real finding).
        assert sweep.any_unstable is (len(unstable) > 0)
        assert all(p >= 0.8 for p in unstable), (
            f"branch-point artifact fired outside the near-one block: "
            f"{sorted(unstable)}"
        )
        assert np.isfinite(sweep.worst_sigma_min)
        assert 0.01 < sweep.worst_sigma_min < 0.05
        assert sweep.worst_psi is not None and sweep.worst_psi < 0.15
