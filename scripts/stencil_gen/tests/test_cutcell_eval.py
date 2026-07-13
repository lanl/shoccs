"""Tests for the cached lambdified cut-cell evaluator (plan 48 item 1.1)."""

import time

import numpy as np
import pytest
from sympy import Symbol

from stencil_gen.cutcell_eval import CutCellEvaluator, make_cutcell_evaluator
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
    ``derive_cut_cell_mathematica`` (the singularity-free workflow, exactly
    as ``psi_sweep_group_velocity`` does), whose parameterization keeps the
    four uniform-boundary alphas free — NOT the 2-parameter
    (alpha_2, w_4) coordinates of the legacy zeros path that generated
    ``src/stencils/E4_1.cpp``.
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
