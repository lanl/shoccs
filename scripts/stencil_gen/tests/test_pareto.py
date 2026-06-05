"""Tests for :mod:`stencil_gen.pareto` (plan 45)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from stencil_gen.brady2d_stability import StabilityReport
from stencil_gen.pareto import (
    _PARETO_SENTINEL,
    ParetoPoint,
    ParetoResult,
    make_multi_objective,
    run_nsga2,
)


def _empty_report_with(**layers) -> StabilityReport:
    """Build a ``StabilityReport`` with the given layer payloads populated."""
    r = StabilityReport.empty()
    for name, value in layers.items():
        setattr(r, name, value)
    return r


class TestParetoDataclasses:
    """Plan 45.1a: ``ParetoPoint`` and ``ParetoResult`` are frozen dataclasses."""

    @staticmethod
    def _make_point(offset: float = 0.0) -> ParetoPoint:
        return ParetoPoint(
            x=np.array([offset, offset + 1.0]),
            params={"alpha": [offset, offset + 1.0]},
            objectives=np.array([offset * 2, offset * 2 + 1]),
            report={"failed_layer": None},
        )

    @staticmethod
    def _make_result(front: tuple[ParetoPoint, ...]) -> ParetoResult:
        return ParetoResult(
            front=front,
            objective_fields=(
                "layer1.boundary_gv_err",
                "layer_bl42.max_spectral_abscissa",
            ),
            scheme="E4",
            kernel="classical",
            bounds=((-2.0, 2.0), (0.05, 2.0)),
            method="NSGA-II",
            pop_size=20,
            n_gen=10,
            n_evals=200,
            seed=1,
            compute_time=1.23,
            hv_trace=(0.1, 0.2, 0.3),
            ref_point=(1.0, 1.0),
            extras={"n_sentinel_filtered": 0},
        )

    def test_pareto_point_frozen(self):
        pt = self._make_point()
        with pytest.raises(FrozenInstanceError):
            pt.x = np.array([99.0, 99.0])
        with pytest.raises(FrozenInstanceError):
            pt.objectives = np.array([0.0, 0.0])

    def test_pareto_result_frozen(self):
        front = tuple(self._make_point(i) for i in range(3))
        result = self._make_result(front)
        assert len(result.front) == 3
        with pytest.raises(FrozenInstanceError):
            result.front = ()
        with pytest.raises(FrozenInstanceError):
            result.seed = 2

    def test_pareto_result_front_is_tuple(self):
        # Plan 45.1c picks the "require tuple for immutability" option: the
        # field is annotated ``tuple[ParetoPoint, ...]`` and constructed with
        # tuples.  Python dataclasses do not enforce annotations at runtime,
        # so this test guards the producer side (run_nsga2, load_pareto_front,
        # test fixtures) against accidentally passing a list.
        front = tuple(self._make_point(i) for i in range(3))
        result = self._make_result(front)
        assert isinstance(result.front, tuple)
        # Each member is a ParetoPoint; indexing and iteration both work.
        assert all(isinstance(p, ParetoPoint) for p in result.front)
        assert result.front[0] is front[0]


class TestMakeMultiObjective:
    """Plan 45.1b: vector-valued feasibility-gated objective."""

    def test_shape_matches_field_count_two(self, monkeypatch):
        # Two fields → shape (2,).
        import stencil_gen.pareto as pareto_mod

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            return _empty_report_with(
                layer1={"boundary_gv_err": 0.03},
                layer3={"max_stab_eig": 1e-12},
                layer_bl42={"max_spectral_abscissa": 0.9},
            )

        monkeypatch.setattr(pareto_mod, "brady2d_stability_score", fake_score)
        f = make_multi_objective(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"],
        )
        out = f(np.array([-0.77, 0.16]))
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
        assert out.dtype == float
        np.testing.assert_allclose(out, [0.03, 0.9])

    def test_shape_matches_field_count_three(self, monkeypatch):
        # Three fields → shape (3,).
        import stencil_gen.pareto as pareto_mod

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            return _empty_report_with(
                layer1={"boundary_gv_err": 0.04},
                layer3={"max_stab_eig": 2e-12},
                layer_bl42={"max_spectral_abscissa": 0.8},
                layer6={"transient_growth_bound": 3.3},
            )

        monkeypatch.setattr(pareto_mod, "brady2d_stability_score", fake_score)
        f = make_multi_objective(
            "E4",
            "classical",
            [
                "layer1.boundary_gv_err",
                "layer_bl42.max_spectral_abscissa",
                "layer6.transient_growth_bound",
            ],
        )
        out = f(np.array([-0.77, 0.16]))
        assert out.shape == (3,)
        np.testing.assert_allclose(out, [0.04, 0.8, 3.3])

    def test_sentinel_on_gate_trip(self, monkeypatch):
        # Simulate a L1 failure (the gate, since max_layer=3 → gate_layer=2
        # and failed_layer=1 <= 2).  Should return a vector of sentinel
        # values, not +inf, so pymoo's hypervolume indicator stays well-
        # defined.
        import stencil_gen.pareto as pareto_mod

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            r = StabilityReport.empty()
            r.failed_layer = 1
            r.failed_reason = "synthetic L1 failure (alpha=[5,5] non-feasible)"
            return r

        monkeypatch.setattr(pareto_mod, "brady2d_stability_score", fake_score)
        f = make_multi_objective(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"],
        )
        out = f(np.array([5.0, 5.0]))
        assert out.shape == (2,)
        assert np.all(np.isfinite(out))
        np.testing.assert_array_equal(out, [_PARETO_SENTINEL, _PARETO_SENTINEL])

    def test_sentinel_on_shape_mismatch(self):
        # E4 classical expects x of length 2 (alpha_0, alpha_1).  A length-3
        # input causes params_from_vector to raise; the closure must swallow
        # the exception and return the sentinel vector.
        f = make_multi_objective(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"],
        )
        out = f(np.array([-0.77, 0.16, 99.0]))
        assert out.shape == (2,)
        np.testing.assert_array_equal(out, [_PARETO_SENTINEL, _PARETO_SENTINEL])

    def test_sentinel_on_score_exception(self, monkeypatch):
        # Any exception from brady2d_stability_score (singular RBF system,
        # numerical blow-up at extreme parameters, …) returns the sentinel
        # vector without propagating.
        import stencil_gen.pareto as pareto_mod

        def raising_score(scheme, kernel, params, *, max_layer, short_circuit):
            raise RuntimeError("simulated numerical blow-up")

        monkeypatch.setattr(pareto_mod, "brady2d_stability_score", raising_score)
        f = make_multi_objective(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"],
        )
        out = f(np.array([-0.77, 0.16]))
        np.testing.assert_array_equal(out, [_PARETO_SENTINEL, _PARETO_SENTINEL])

    def test_finite_on_known_feasible_point(self):
        # BL published optimum for E4 classical.  With both L1 and L3r
        # populated the closure must return an all-finite vector.  This is a
        # real brady2d_stability_score call and the most expensive test in
        # this file; kept because it exercises the full
        # make_multi_objective → short-circuit cascade → extract_field chain
        # end-to-end.
        f = make_multi_objective(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"],
        )
        out = f(np.array([-0.7733323791884821, 0.1623961700641681]))
        assert out.shape == (2,)
        assert np.all(np.isfinite(out))
        # L1 group-velocity error is well below the L1 tolerance for the BL
        # optimum; L3r spectral abscissa passes but may be small-positive or
        # negative depending on integrator cutoff — only sign constraint is
        # "finite, populated".
        assert out[0] >= 0.0

    def test_gate_layer_auto_inferred_from_max_field(self, monkeypatch):
        # Fields span multiple layers (layer1 → 1, layer_bl42 → 3).  The
        # auto-inferred max_layer is 3 and gate_layer is 2: an L2 failure
        # gates; an L3r failure does not (that's the bl42-self-gate trap the
        # scalar analogue in 45.0b unblocked).
        import stencil_gen.pareto as pareto_mod

        captured: dict = {}

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            captured["max_layer"] = max_layer
            r = StabilityReport.empty()
            r.failed_layer = captured["failed_layer_sentinel"]
            r.layer1 = {"boundary_gv_err": 0.04}
            r.layer_bl42 = {"max_spectral_abscissa": 5.0}
            return r

        monkeypatch.setattr(pareto_mod, "brady2d_stability_score", fake_score)
        f = make_multi_objective(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"],
        )

        # L2 failure (below gate_layer=2 → <=2 → gates to sentinel).
        captured["failed_layer_sentinel"] = 2
        out = f(np.array([-0.77, 0.16]))
        assert captured["max_layer"] == 3
        np.testing.assert_array_equal(out, [_PARETO_SENTINEL, _PARETO_SENTINEL])

        # L3 failure (at max_layer, above gate_layer=2 → does not gate; the
        # populated L1 / BL42 payload passes through extract_field).
        captured["failed_layer_sentinel"] = 3
        out = f(np.array([-0.77, 0.16]))
        np.testing.assert_allclose(out, [0.04, 5.0])

    def test_rejects_fewer_than_two_fields(self):
        with pytest.raises(ValueError, match="requires >= 2 report_fields"):
            make_multi_objective("E4", "classical", ["layer1.boundary_gv_err"])

    def test_rejects_unknown_field(self):
        # Mirrors make_objective's failure mode: unrecognised prefix raises
        # before the closure is constructed.
        with pytest.raises(ValueError, match="cannot infer max_layer"):
            make_multi_objective(
                "E4",
                "classical",
                ["layer1.boundary_gv_err", "bogus_field"],
            )


# --- NSGA-II driver synthetic problems ---------------------------------------


def _zdt1_like(x: np.ndarray) -> np.ndarray:
    """2-variable ZDT1-style vector objective on ``[0,1]^2``.

    ``f_1(x) = x_0``; ``f_2(x) = g(x) * (1 - sqrt(x_0 / g(x)))``, ``g = 1 + x_1``.
    Known Pareto-optimal front on ``x_1 = 0``: ``f_1 = x_0``, ``f_2 = 1 - sqrt(x_0)``.
    Deterministic, cheap, and has a well-spread convex front — ideal for
    pinning down NSGA-II determinism/non-dominance without touching the
    expensive ``brady2d_stability_score`` cascade.
    """
    x = np.asarray(x, dtype=float).ravel()
    f1 = float(x[0])
    g = 1.0 + float(x[1])
    f2 = float(g * (1.0 - np.sqrt(max(1e-18, x[0] / g))))
    return np.array([f1, f2], dtype=float)


def _half_sentinel(x: np.ndarray) -> np.ndarray:
    """Vector objective that returns the sentinel for ``x[0] > 0.5``.

    Used to exercise the sentinel-filtering path in :func:`run_nsga2`: roughly
    half the uniformly-sampled population is forced into the infeasible
    region.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x[0] > 0.5:
        return np.array([_PARETO_SENTINEL, _PARETO_SENTINEL], dtype=float)
    f1 = float(x[0])
    f2 = float(1.0 - x[0] + x[1])
    return np.array([f1, f2], dtype=float)


def _pareto_dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True iff ``a`` Pareto-dominates ``b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a <= b) and np.any(a < b))


class TestRunNSGA2:
    """Plan 45.2a + 45.2c: NSGA-II driver on synthetic and real objectives."""

    def test_determinism_same_seed(self):
        # Two runs with identical inputs must produce identical objective
        # matrices.  pymoo's NSGA2 seeds numpy's legacy RNG; the evaluator is
        # deterministic; therefore the returned front (as a multiset of
        # objective vectors) is exactly reproducible modulo ordering.
        r1 = run_nsga2(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer3.max_stab_eig"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            pop_size=12,
            n_gen=4,
            seed=1,
            objective=_zdt1_like,
        )
        r2 = run_nsga2(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer3.max_stab_eig"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            pop_size=12,
            n_gen=4,
            seed=1,
            objective=_zdt1_like,
        )
        F1 = np.array([p.objectives for p in r1.front])
        F2 = np.array([p.objectives for p in r2.front])
        assert F1.shape == F2.shape
        # Sort lexicographically so set-equality on ordered rows is robust to
        # any internal ordering permutation.
        order1 = np.lexsort(F1.T[::-1])
        order2 = np.lexsort(F2.T[::-1])
        np.testing.assert_allclose(F1[order1], F2[order2], rtol=0, atol=1e-12)

    def test_non_dominated_front(self):
        res = run_nsga2(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer3.max_stab_eig"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            pop_size=16,
            n_gen=6,
            seed=2,
            objective=_zdt1_like,
        )
        F = np.array([p.objectives for p in res.front])
        n = F.shape[0]
        assert n >= 2
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                assert not _pareto_dominates(F[j], F[i]), (
                    f"member {i} ({F[i]}) dominated by {j} ({F[j]})"
                )

    def test_hv_trace_monotone_nondecreasing(self):
        # Elitist selection in NSGA-II ⇒ hypervolume is monotone
        # non-decreasing across generations.  Small numerical tolerance for
        # indicator evaluation noise.
        res = run_nsga2(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer3.max_stab_eig"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            pop_size=16,
            n_gen=6,
            seed=3,
            objective=_zdt1_like,
        )
        hv = np.asarray(res.hv_trace, dtype=float)
        assert hv.shape == (6,)
        deltas = np.diff(hv)
        assert np.all(deltas >= -1e-10), (
            f"hv_trace not monotone non-decreasing: deltas={deltas}"
        )

    def test_sentinel_rows_excluded(self):
        res = run_nsga2(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer3.max_stab_eig"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            pop_size=20,
            n_gen=4,
            seed=4,
            objective=_half_sentinel,
        )
        # Front must be non-empty — without this, the per-point asserts below
        # are vacuous when run_nsga2 returns an empty front.
        assert len(res.front) >= 1, "front must be non-empty for half-sentinel objective"
        for p in res.front:
            assert np.all(np.isfinite(p.objectives))
            assert np.all(p.objectives < _PARETO_SENTINEL)
        # With ~half the initial population in the infeasible region, the final
        # non-dominated front must be a strict subset of the pop_size=20 — a
        # 20-member final front would indicate no filtering / no selection ran.
        # Empirically pymoo's NSGA-II crowds out sentinel rows during selection
        # so n_sentinel_filtered is typically 0 at this budget; the population-
        # size guard is the meaningful signal that the pipeline did real work.
        assert len(res.front) < 20, (
            f"front size {len(res.front)} == pop_size; sentinel filtering / "
            "selection did not run"
        )
        # 46.5a's empirical analysis (probed seeds 1–11, n_gen ∈ {1,2,3,4,6,8},
        # pop_size ∈ {10,20,40}) found pymoo's NSGA-II crowds out sentinel rows
        # during selection before keep_mask sees them, so n_sentinel_filtered is
        # always 0 at this budget. Hardening this into a regression signal: if a
        # future pymoo upgrade changes selection so sentinels survive to res.F,
        # this assert catches it instead of the test silently passing.
        assert res.extras["n_sentinel_filtered"] == 0, (
            "regression: pymoo no longer crowds out sentinel rows in selection "
            "before keep_mask sees them; revisit the test budget or the "
            "keep_mask filter"
        )

    def test_ref_point_override(self):
        ref = (2.0, 2.0)
        res = run_nsga2(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer3.max_stab_eig"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            pop_size=10,
            n_gen=3,
            seed=1,
            ref_point=ref,
            objective=_zdt1_like,
        )
        assert res.ref_point == ref

    def test_rejects_fewer_than_two_fields(self):
        with pytest.raises(ValueError, match="requires >= 2 report_fields"):
            run_nsga2(
                "E4",
                "classical",
                ["layer1.boundary_gv_err"],
                bounds=[(0.0, 1.0)],
                pop_size=4,
                n_gen=2,
                seed=1,
                objective=_zdt1_like,
            )

    def test_rejects_bad_ref_point_shape(self):
        with pytest.raises(ValueError, match="ref_point shape"):
            run_nsga2(
                "E4",
                "classical",
                ["layer1.boundary_gv_err", "layer3.max_stab_eig"],
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                pop_size=6,
                n_gen=2,
                seed=1,
                ref_point=(1.0, 2.0, 3.0),
                objective=_zdt1_like,
            )

    def test_result_metadata_populated(self):
        res = run_nsga2(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer3.max_stab_eig"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            pop_size=8,
            n_gen=3,
            seed=5,
            objective=_zdt1_like,
        )
        assert res.method == "NSGA-II"
        assert res.scheme == "E4"
        assert res.kernel == "classical"
        assert res.pop_size == 8
        assert res.n_gen == 3
        assert res.seed == 5
        assert res.n_evals > 0
        assert res.compute_time >= 0.0
        assert res.bounds == ((0.0, 1.0), (0.0, 1.0))
        assert res.objective_fields == (
            "layer1.boundary_gv_err",
            "layer3.max_stab_eig",
        )
        # auto-picked ref_point must dominate the front
        ref = np.asarray(res.ref_point, dtype=float)
        assert len(res.front) >= 1, "front must be non-empty for ZDT1-like; check seeding"
        for p in res.front:
            assert np.all(p.objectives <= ref + 1e-12)

    @pytest.mark.slow
    def test_integration_classical_alpha_2d(self):
        # Real ``brady2d_stability_score`` on the L1 vs L3r trade-off
        # (the plan-45 primary calibration objective pair).  Plan 45.2c
        # originally specified L3/L6 objectives; empirically the L6
        # feasibility region is too narrow (~0% random hit-rate) for a
        # 12x4 NSGA-II budget, so the front collapses to empty.  The L1 vs
        # layer_bl42 pair has the same scientific shape (it's the pair used
        # by 45.6a for the persisted calibration fronts) and admits a few
        # feasible samples inside a BL-centred box at the same budget.
        res = run_nsga2(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa"],
            bounds=[(-1.0, -0.5), (0.05, 0.3)],
            pop_size=12,
            n_gen=4,
            seed=1,
        )
        assert len(res.front) >= 2
        assert res.hv_trace[-1] > 0.0
        F = np.array([p.objectives for p in res.front])
        n = F.shape[0]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                assert not _pareto_dominates(F[j], F[i])
        for p in res.front:
            alpha = p.params.get("alpha")
            assert alpha is not None and len(alpha) == 2


class TestHVCallback:
    """Plan 45.2b: per-generation hypervolume recorder."""

    def test_per_gen_count_matches_n_gen(self):
        res = run_nsga2(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer3.max_stab_eig"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            pop_size=8,
            n_gen=5,
            seed=1,
            objective=_zdt1_like,
        )
        assert len(res.hv_trace) == 5
        assert len(res.extras["hv_n_nds"]) == 5

    def test_empty_front_records_zero_hv(self):
        # With every evaluation returning the sentinel the filtered set is
        # empty; the callback must record 0.0 instead of raising or recording
        # NaN.
        def all_sentinel(x):
            return np.array([_PARETO_SENTINEL, _PARETO_SENTINEL], dtype=float)

        res = run_nsga2(
            "E4",
            "classical",
            ["layer1.boundary_gv_err", "layer3.max_stab_eig"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            pop_size=6,
            n_gen=3,
            seed=1,
            ref_point=(1.0, 1.0),
            objective=all_sentinel,
        )
        assert len(res.hv_trace) == 3
        assert all(v == 0.0 for v in res.hv_trace)
        assert all(n == 0 for n in res.extras["hv_n_nds"])
        assert len(res.front) == 0
