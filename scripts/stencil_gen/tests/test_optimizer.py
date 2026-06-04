"""Tests for :mod:`stencil_gen.optimizer` (plan 43)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import qmc

from stencil_gen.brady2d_stability import StabilityReport
from stencil_gen.gks_kreiss import KreissResult
from stencil_gen.optimizer import (
    DEFAULT_BOUNDS,
    _COBYQA_AVAILABLE,
    OptimizeResult,
    _report_to_dict,
    extract_field,
    make_objective,
    multi_start_optimize,
    params_from_vector,
    run_scipy_de,
    run_scipy_local,
    run_scipy_shgo,
    run_staged_optimize,
    vector_from_params,
)


class TestParamsVector:
    @pytest.mark.parametrize(
        ("kernel", "x", "expected"),
        [
            ("tension", [3.0], {"sigma": 3.0}),
            ("gaussian", [1.5], {"epsilon": 1.5}),
            ("multiquadric", [0.7], {"epsilon": 0.7}),
            ("classical", [-0.5, 197.0 / 288.0], {"alpha": [-0.5, 197.0 / 288.0]}),
        ],
    )
    def test_params_from_vector(self, kernel, x, expected):
        assert params_from_vector(kernel, np.asarray(x)) == expected

    @pytest.mark.parametrize(
        ("kernel", "params", "expected_x"),
        [
            ("tension", {"sigma": 3.0}, [3.0]),
            ("gaussian", {"epsilon": 1.5}, [1.5]),
            ("multiquadric", {"epsilon": 0.7}, [0.7]),
            ("classical", {"alpha": [-0.5, 1.0]}, [-0.5, 1.0]),
        ],
    )
    def test_vector_from_params(self, kernel, params, expected_x):
        v = vector_from_params(kernel, params)
        assert v.dtype == np.float64
        np.testing.assert_array_equal(v, np.asarray(expected_x, dtype=float))

    @pytest.mark.parametrize(
        ("kernel", "params"),
        [
            ("tension", {"sigma": 3.0}),
            ("gaussian", {"epsilon": 1.5}),
            ("multiquadric", {"epsilon": 0.7}),
            ("classical", {"alpha": [-0.5, 1.0]}),
        ],
    )
    def test_roundtrip_params_vector_params(self, kernel, params):
        v = vector_from_params(kernel, params)
        assert params_from_vector(kernel, v) == params

    @pytest.mark.parametrize(
        ("kernel", "x"),
        [
            ("tension", [1.0, 2.0]),
            ("gaussian", [1.0, 2.0]),
            ("multiquadric", []),
            ("classical", [1.0]),
        ],
    )
    def test_params_from_vector_wrong_dim(self, kernel, x):
        with pytest.raises(ValueError):
            params_from_vector(kernel, np.asarray(x, dtype=float))

    @pytest.mark.parametrize("kernel", ["tension-penalty", "mixed-epsilon"])
    def test_pruned_kernels_rejected(self, kernel):
        # Plan 43.1d (option b): these families are out of scope for the
        # layered optimizer; brady2d_stability_score does not route them.
        with pytest.raises(ValueError, match="unknown kernel"):
            params_from_vector(kernel, np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="unknown kernel"):
            vector_from_params(kernel, {"sigma": 1.0, "gamma": 2.0})

    def test_params_from_vector_unknown_kernel(self):
        with pytest.raises(ValueError, match="unknown kernel"):
            params_from_vector("nope", np.array([1.0]))

    def test_vector_from_params_unknown_kernel(self):
        with pytest.raises(ValueError, match="unknown kernel"):
            vector_from_params("nope", {})

    def test_classical_requires_length_two_alpha(self):
        with pytest.raises(ValueError):
            vector_from_params("classical", {"alpha": [0.1]})

    def test_default_bounds_shapes_match_roundtrip(self):
        # For each entry in DEFAULT_BOUNDS, build a midpoint vector and round-trip
        # through params_from_vector / vector_from_params.
        for (scheme, kernel), bounds in DEFAULT_BOUNDS.items():
            mid = np.array([0.5 * (lo + hi) for (lo, hi) in bounds], dtype=float)
            params = params_from_vector(kernel, mid)
            v = vector_from_params(kernel, params)
            assert v.shape == mid.shape, f"{scheme}/{kernel}: shape mismatch"
            np.testing.assert_array_equal(v, mid)


class TestClassicalAlphaBounds:
    """Sanity tests for the E4 classical-α bound decision in plan 43.9a.

    The C++ ``E4_1`` stencil requires ``alpha[1] >= 197/288 ≈ 0.684`` to
    avoid a cut-cell psi-denominator singularity.  The Python
    ``_build_classical_diff_matrix`` (used by L1–L7) operates on a uniform
    grid and has no such constraint; its analytical feasible region sits
    entirely *below* 197/288 around ``alpha[1] ≈ 0.16``.  ``DEFAULT_BOUNDS``
    reflects the analytical envelope, not the cut-cell envelope — L8
    validation is where the 197/288 check fires.
    """

    _BRADY_LIVESCU_E4 = (-0.7733323791884821, 0.1623961700641681)

    def test_default_bounds_admit_brady_livescu_point(self):
        bounds = DEFAULT_BOUNDS[("E4", "classical")]
        a0, a1 = self._BRADY_LIVESCU_E4
        (lo0, hi0), (lo1, hi1) = bounds
        assert lo0 <= a0 <= hi0, (
            f"Brady-Livescu alpha[0]={a0} outside bound [{lo0}, {hi0}]"
        )
        assert lo1 <= a1 <= hi1, (
            f"Brady-Livescu alpha[1]={a1} outside bound [{lo1}, {hi1}]"
        )

    def test_default_bounds_drop_cpp_cutcell_floor(self):
        # The 197/288 floor is a cut-cell C++ constraint; L1-L7 use
        # uniform grids with no psi denominator.  The analytical-only
        # optimizer bound must therefore admit α₁ below 197/288.
        (_, (lo1, _)) = DEFAULT_BOUNDS[("E4", "classical")]
        assert lo1 < 197.0 / 288.0, (
            f"DEFAULT_BOUNDS alpha[1] lower bound {lo1} >= 197/288; the "
            "analytical pipeline can't reach the Brady-Livescu feasible "
            "region with a C++-cut-cell floor"
        )

    def test_params_roundtrip_at_brady_livescu(self):
        # Round-trip the published point to ensure params_from_vector /
        # vector_from_params handle it cleanly.
        x = np.array(self._BRADY_LIVESCU_E4, dtype=float)
        params = params_from_vector("classical", x)
        assert params == {"alpha": [float(x[0]), float(x[1])]}
        np.testing.assert_array_equal(vector_from_params("classical", params), x)

    def test_python_accepts_alpha_below_cpp_floor(self):
        # Belt-and-braces: ``_build_classical_diff_matrix`` must not raise
        # for the published α (where α₁ < 197/288); the analytical pipeline
        # relies on this.
        from stencil_gen.brady2d_stability import _build_classical_diff_matrix

        D = _build_classical_diff_matrix(
            n=20, p=2, nu=1, alpha_list=list(self._BRADY_LIVESCU_E4),
        )
        assert D.shape == (20, 20)
        assert np.all(np.isfinite(D))

    def test_objective_feasible_at_brady_livescu_within_default_bounds(self):
        # The published point lies in the feasible L3 envelope.  Confirms
        # the optimizer's feasibility cliff has a non-empty interior inside
        # DEFAULT_BOUNDS, so multi-start / SHGO / DE have something to
        # descend on.
        f = make_objective(
            "E4", "classical", "layer3.max_stab_eig",
            gate_layer=3, max_layer=3,
        )
        x = np.array(self._BRADY_LIVESCU_E4, dtype=float)
        val = f(x)
        assert np.isfinite(val), (
            "Brady-Livescu E4 alpha should be L3-feasible under the "
            "widened DEFAULT_BOUNDS"
        )
        assert val < 0.0, (
            f"layer3.max_stab_eig at Brady-Livescu α is {val}; expected "
            "negative (stable)"
        )

    def test_objective_infeasible_above_cpp_floor(self):
        # Picks a point inside DEFAULT_BOUNDS that lies *above* 197/288 —
        # where the C++ cut-cell code is happy but the Python analytical
        # pipeline is unstable.  The objective must return +inf gracefully
        # rather than raising.
        f = make_objective(
            "E4", "classical", "layer3.max_stab_eig",
            gate_layer=3, max_layer=3,
        )
        # alpha[1] well above 197/288 is in the infeasible analytical zone.
        val = f(np.array([-0.77, 1.0]))
        assert not np.isfinite(val)


class TestExtractField:
    @staticmethod
    def _populated_report() -> StabilityReport:
        return StabilityReport(
            layer1={"boundary_gv_err": 1.25e-3},
            layer2=KreissResult(is_stable=True, witness_sigma_min=0.42),
            layer3={"max_stab_eig": -1.5e-4},
            layer6={
                "spectral_abscissa": -2.0e-3,
                "kreiss_constant": 3.7,
                "transient_growth_bound": 12.5,
                "henrici_departure": 0.01,
            },
            layer7={"max_spectral_abscissa": 5.0e-4},
            kreiss=KreissResult(is_stable=True, witness_sigma_min=0.77),
            overall_verdict="pass",
        )

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("layer1.boundary_gv_err", 1.25e-3),
            ("layer3.max_stab_eig", -1.5e-4),
            ("layer6.spectral_abscissa", -2.0e-3),
            ("layer6.kreiss_constant", 3.7),
            ("layer6.transient_growth_bound", 12.5),
            ("layer7.max_spectral_abscissa", 5.0e-4),
            ("kreiss.witness_sigma_min", 0.77),
        ],
    )
    def test_extract_populated_fields(self, path, expected):
        r = self._populated_report()
        assert extract_field(r, path) == pytest.approx(expected)

    def test_extract_unknown_first_segment_returns_inf(self):
        r = self._populated_report()
        assert extract_field(r, "layer99.foo") == float("inf")

    def test_extract_missing_key_returns_inf(self):
        r = self._populated_report()
        assert extract_field(r, "layer1.not_a_metric") == float("inf")

    def test_extract_layer_not_run_returns_inf(self):
        # layer4/layer5/layer8 were not populated above
        r = self._populated_report()
        assert extract_field(r, "layer4.max_local_gv_error") == float("inf")
        assert extract_field(r, "layer8.final_linf") == float("inf")

    def test_extract_empty_report_returns_inf(self):
        r = StabilityReport.empty()
        assert extract_field(r, "layer1.boundary_gv_err") == float("inf")
        assert extract_field(r, "kreiss.witness_sigma_min") == float("inf")

    def test_extract_empty_path_returns_inf(self):
        r = self._populated_report()
        assert extract_field(r, "") == float("inf")

    def test_extract_missing_dataclass_attr_returns_inf(self):
        r = self._populated_report()
        assert extract_field(r, "kreiss.no_such_attr") == float("inf")


class TestExtractFieldBL42:
    def test_extract_max_spectral_abscissa(self):
        r = StabilityReport(
            layer_bl42={
                "spectral_abscissa_by_n": {21: 1e-12, 41: 2e-12},
                "max_spectral_abscissa": 2e-12,
                "purely_imaginary": True,
            },
            overall_verdict="pass",
        )
        assert extract_field(r, "layer_bl42.max_spectral_abscissa") == pytest.approx(
            2e-12
        )

    def test_extract_purely_imaginary(self):
        r = StabilityReport(
            layer_bl42={
                "spectral_abscissa_by_n": {21: 1e-12},
                "max_spectral_abscissa": 1e-12,
                "purely_imaginary": True,
            },
            overall_verdict="pass",
        )
        assert extract_field(r, "layer_bl42.purely_imaginary") == 1.0

    def test_extract_purely_imaginary_false(self):
        r = StabilityReport(
            layer_bl42={
                "spectral_abscissa_by_n": {21: 0.5},
                "max_spectral_abscissa": 0.5,
                "purely_imaginary": False,
            },
            overall_verdict="fail",
        )
        assert extract_field(r, "layer_bl42.purely_imaginary") == 0.0

    def test_extract_layer_bl42_not_populated_returns_inf(self):
        r = StabilityReport.empty()
        assert extract_field(r, "layer_bl42.max_spectral_abscissa") == float("inf")


class TestReportToDictLayerBL42:
    def test_report_to_dict_includes_layer_bl42(self):
        r = StabilityReport(
            layer_bl42={"max_spectral_abscissa": 0.5},
            overall_verdict="pass",
        )
        out = _report_to_dict(r)
        assert "layer_bl42" in out
        assert out["layer_bl42"]["max_spectral_abscissa"] == 0.5

    def test_report_to_dict_layer_bl42_filters_non_numeric(self):
        r = StabilityReport(
            layer_bl42={
                "spectral_abscissa_by_n": {21: 1e-12, 41: 2e-12},
                "max_spectral_abscissa": 2e-12,
                "purely_imaginary": True,
            },
            overall_verdict="pass",
        )
        out = _report_to_dict(r)
        assert "layer_bl42" in out
        # dict-valued entry dropped; numeric entries kept.
        assert "spectral_abscissa_by_n" not in out["layer_bl42"]
        assert out["layer_bl42"]["max_spectral_abscissa"] == pytest.approx(2e-12)
        # bool is an int subclass; comprehension keeps it as float.
        assert out["layer_bl42"]["purely_imaginary"] == 1.0

    def test_report_to_dict_omits_layer_bl42_when_none(self):
        r = StabilityReport.empty()
        out = _report_to_dict(r)
        assert "layer_bl42" not in out


class TestReportToDictSchemaParity:
    """Plan 46.2d: enforce schema parity across the three ``_report_to_dict``
    copies (``optimizer``, ``brady2d_sweep``, ``brady2d_calibration``).

    Future-proofs the schema: any new field added to ``StabilityReport`` will
    fail this test until added to all three serializers.  Also guards the
    bloat strip from 46.2b.1: the heavy diagnostic ndarrays on ``KreissResult``
    must not survive into the serialized form.
    """

    @staticmethod
    def _fully_populated_report():
        """Build a ``StabilityReport`` with every public field set."""
        from stencil_gen.gks_kreiss import KreissResult
        from stencil_gen.non_normality import NonNormalityReport

        kr = KreissResult(
            is_stable=True,
            witness_s=1.0 + 2.0j,
            witness_sigma_min=0.5,
            imaginary_axis_perturbation_verdict="all_incoming",
            defective_kappa_detected=False,
            s_grid_shape=(30, 80),
            compute_time=0.1,
            sigma_min_field=np.zeros((30, 80)),
            s_grid=np.zeros((30, 80), dtype=complex),
            n_admissible_roots=4,
        )
        nn = NonNormalityReport(
            spectral_abscissa=-1.0,
            numerical_abscissa=-0.5,
            henrici_departure=0.01,
            eigenvector_condition=10.0,
            pseudospectral_abscissae={1e-3: 0.1, 1e-2: 0.2},
            kreiss_constant=2.0,
            transient_growth_bound=5.0,
            n=100,
            compute_time=0.5,
        )
        return StabilityReport(
            layer1={
                "boundary_gv_err": 1e-4,
                "interior_gv_err_x": 1e-5,
                "interior_gv_err_y": 1e-5,
                "cutoff_fraction": 0.1,
            },
            layer2=KreissResult(is_stable=True, witness_sigma_min=0.42),
            layer3={"max_stab_eig": -1.5e-4},
            layer4={"max_local_gv_error": 0.05},
            layer5={"max_aligned_error": 0.04},
            layer6={
                "spectral_abscissa": -2e-3,
                "kreiss_constant": 3.7,
                "transient_growth_bound": 12.5,
                "henrici_departure": 0.01,
            },
            layer7={"max_spectral_abscissa": 5e-4},
            layer8={"final_linf": 1e-3, "stable": True, "wall_time_s": 1.0},
            layer_bl42={
                "max_spectral_abscissa": 1e-12,
                "purely_imaginary": True,
                "spectral_abscissa_by_n": {21: 1e-12},
            },
            non_normality=nn,
            kreiss=kr,
            overall_verdict="pass",
            failed_layer=None,
            failed_reason="",
            compute_time=1.5,
        )

    @staticmethod
    def _serializers():
        from stencil_gen.benchmarks.brady2d_calibration import (
            _report_to_dict as _calib_serialize,
        )
        from stencil_gen.optimizer import _report_to_dict as _opt_serialize
        from sweeps.brady2d_sweep import _report_to_dict as _sweep_serialize

        return [
            pytest.param(_opt_serialize, id="optimizer"),
            pytest.param(_sweep_serialize, id="brady2d_sweep"),
            pytest.param(_calib_serialize, id="brady2d_calibration"),
        ]

    @pytest.fixture
    def report(self):
        return self._fully_populated_report()

    @pytest.mark.parametrize("serializer", _serializers())
    def test_all_layers_present(self, serializer, report):
        out = serializer(report)
        for k in (
            "layer1", "layer2", "layer3", "layer4", "layer5",
            "layer6", "layer7", "layer8", "layer_bl42",
        ):
            assert k in out, f"missing {k!r} in {serializer.__module__} output"

    @pytest.mark.parametrize("serializer", _serializers())
    def test_top_level_metadata_present(self, serializer, report):
        out = serializer(report)
        for k in (
            "compute_time",
            "failed_layer",
            "failed_reason",
            "overall_verdict",
            "non_normality",
            "kreiss",
        ):
            assert k in out, f"missing {k!r} in {serializer.__module__} output"

    @pytest.mark.parametrize("serializer", _serializers())
    def test_kreiss_diagnostic_arrays_stripped(self, serializer, report):
        # 46.2b.1 bloat guard: heavy diagnostic ndarrays on KreissResult
        # (sigma_min_field shape (30, 80), s_grid same shape complex) must
        # not survive into the serialized form.  Re-introducing them
        # silently inflates Pareto JSON files by ~40 KB per record.
        out = serializer(report)
        assert "sigma_min_field" not in out["kreiss"]
        assert "s_grid" not in out["kreiss"]

    @pytest.mark.parametrize("serializer", _serializers())
    def test_serialized_under_bloat_threshold(self, serializer, report):
        from sweeps._pareto_io import _ParetoEncoder

        out = serializer(report)
        blob = json.dumps(out, cls=_ParetoEncoder)
        # 4 KB ceiling: legitimate scalar payload is ~1.2 KB; un-stripped
        # diagnostic arrays push past 40 KB.  Comfortable margin either way.
        assert len(blob) < 4096, (
            f"{serializer.__module__} serialized to {len(blob)} bytes — "
            "did the kreiss-array strip regress?"
        )

    @pytest.mark.parametrize("serializer", _serializers())
    def test_pareto_encoder_roundtrip(self, serializer, report):
        # ``KreissResult.witness_s`` is ``complex``; the encoder must handle
        # it (46.2b added the ``complex`` -> ``[real, imag]`` rule).
        from sweeps._pareto_io import _ParetoEncoder

        out = serializer(report)
        blob = json.dumps(out, cls=_ParetoEncoder)
        # Round-trip parses without error and preserves witness_s as a
        # 2-element list.
        loaded = json.loads(blob)
        assert loaded["kreiss"]["witness_s"] == [1.0, 2.0]

    @pytest.mark.parametrize("serializer", _serializers())
    def test_save_known_values_roundtrip(
        self, serializer, report, tmp_path, monkeypatch
    ):
        # 46.4a.0 regression: ``save_known_values`` must accept the complex /
        # numpy / dataclass types reachable via ``_report_to_dict``.  Before
        # 46.4a.0 the default ``json.dump`` raised
        # ``TypeError: Object of type complex is not JSON serializable`` on
        # ``kreiss.witness_s`` (added by 46.2b).
        from sweeps import _common

        path = tmp_path / "known_values.json"
        monkeypatch.setattr(_common, "KNOWN_VALUES_PATH", path)
        payload = {"brady2d_sweep": {"E4": {"classical": {
            "report": serializer(report),
        }}}}
        _common.save_known_values(payload)
        loaded = _common.load_known_values()
        # ``complex`` round-trips as ``[real, imag]`` per the encoder.
        kr = loaded["brady2d_sweep"]["E4"]["classical"]["report"]["kreiss"]
        assert kr["witness_s"] == [1.0, 2.0]


class TestMakeObjective:
    def test_objective_returns_finite_on_feasible(self):
        # E4 tension σ=3.0 passes L1-L2 (GKS-stable); gate at L2 to avoid
        # L3r (BL42 reflecting-hyperbolic) which rejects this sigma.
        f = make_objective(
            "E4", "tension", "layer1.boundary_gv_err",
            gate_layer=2, max_layer=2,
        )
        val = f(np.array([3.0]))
        assert np.isfinite(val)
        assert val >= 0.0

    def test_objective_returns_inf_on_gate_failure(self):
        # A tiny gaussian ε makes the RBF matrix nearly singular: L1 or L3
        # fails and the feasibility gate forces +inf.
        f = make_objective(
            "E4", "gaussian", "layer1.boundary_gv_err",
            gate_layer=3, max_layer=3,
        )
        assert f(np.array([0.01])) == float("inf")

    def test_objective_catches_exception(self, monkeypatch):
        import stencil_gen.optimizer as opt

        def _boom(*_args, **_kwargs):
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr(opt, "brady2d_stability_score", _boom)
        f = opt.make_objective(
            "E4", "tension", "layer1.boundary_gv_err",
            gate_layer=3, max_layer=3,
        )
        assert f(np.array([3.0])) == float("inf")

    def test_objective_raises_on_bad_field(self):
        # A nonsense dotted path at a feasible point returns +inf (extract_field
        # treats missing segments as inf) rather than raising.
        # Use classical E4 with known-good alpha that passes all layers
        # including L3r, so the gate passes and extract_field is exercised.
        f = make_objective(
            "E4", "classical", "layer99.foo",
            gate_layer=3, max_layer=3,
        )
        assert f(np.array([-0.7733323791884821, 0.1623961700641681])) == float("inf")

    def test_objective_infers_max_layer_from_field(self):
        # layer6.* implies max_layer=6; use classical E4 with known-good alpha
        # that passes all layers including L3r.
        f = make_objective("E4", "classical", "layer6.spectral_abscissa")
        val = f(np.array([-0.7733323791884821, 0.1623961700641681]))
        assert np.isfinite(val)

    def test_objective_rejects_max_layer_below_gate(self):
        with pytest.raises(ValueError, match="less than gate_layer"):
            make_objective(
                "E4", "tension", "layer1.boundary_gv_err",
                gate_layer=3, max_layer=1,
            )

    def test_objective_rejects_uninferable_field_without_max_layer(self):
        with pytest.raises(ValueError, match="cannot infer max_layer"):
            make_objective("E4", "tension", "no_prefix_here")


class TestMakeObjectiveBL42:
    def test_infer_max_layer_from_layer_bl42(self):
        from stencil_gen.optimizer import _infer_max_layer

        assert _infer_max_layer("layer_bl42.max_spectral_abscissa") == 3
        assert _infer_max_layer("layer_bl42.purely_imaginary") == 3

    def test_objective_bl42_classical_finite(self):
        f = make_objective(
            "E4", "classical", "layer_bl42.max_spectral_abscissa",
            gate_layer=3,
        )
        val = f(np.array([-0.7733323791884821, 0.1623961700641681]))
        assert np.isfinite(val)

    def test_objective_bl42_infers_max_layer_3(self):
        f = make_objective(
            "E4", "classical", "layer_bl42.max_spectral_abscissa",
        )
        val = f(np.array([-0.7733323791884821, 0.1623961700641681]))
        assert np.isfinite(val)


class TestGateLayerInfer:
    """Plan 45.0b: ``gate_layer`` defaults to ``max(max_layer - 1, 0)``.

    Rationale: for an objective living in layer N, the natural feasibility
    gate is "every strictly-earlier layer passes".  The old hard-coded
    ``gate_layer=3`` caused layer-6/7 objectives to never gate (wasting
    evals on infeasible points) and ``layer_bl42`` (layer 3) objectives to
    gate themselves into a permanent ``+inf`` trap.
    """

    def test_default_gate_for_layer6_objective(self, monkeypatch):
        import stencil_gen.optimizer as opt

        captured: dict = {}

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            captured["max_layer"] = max_layer
            r = StabilityReport.empty()
            # Simulate a cascade failure at layer 5.  With auto-infer,
            # gate_layer=5, so ``failed_layer <= gate_layer`` is True and
            # the closure must return +inf.
            r.failed_layer = 5
            r.failed_reason = "synthetic L5 failure"
            return r

        monkeypatch.setattr(opt, "brady2d_stability_score", fake_score)
        f = make_objective("E4", "classical", "layer6.transient_growth_bound")
        val = f(np.array([-0.77, 0.16]))
        assert captured["max_layer"] == 6
        assert val == float("inf")

    @pytest.mark.parametrize(
        ("failed_layer", "layer_bl42_payload", "expected"),
        [
            # L2 failure: at the auto-inferred gate (gate_layer=2 for a
            # layer-3 objective).  Gates under both old hardcoded
            # gate_layer=3 and new auto-infer; this sub-case verifies the
            # feasibility cliff fires for at-or-below-gate failures.
            (2, None, float("inf")),
            # L3 failure: above the gate.  Under the OLD hardcoded
            # gate_layer=3 this would gate (failed_layer=3 <= 3) — a
            # self-gate trap because the objective lives in the "failing"
            # layer.  Under the NEW auto-inferred gate_layer=2,
            # failed_layer=3 > 2 does NOT gate and the closure returns the
            # populated ``layer_bl42`` payload.  This is the true
            # regression case for plan 45.0b auto-infer; without it the
            # test was vacuous (both old and new defaults gate at L2).
            # See also ``test_bl42_l3r_failure_returns_finite`` (45.0e)
            # which exercises the same L3-failure scenario standalone.
            (3, {"max_spectral_abscissa": 5.0}, 5.0),
        ],
    )
    def test_default_gate_for_bl42_objective(
        self, monkeypatch, failed_layer, layer_bl42_payload, expected
    ):
        import stencil_gen.optimizer as opt

        captured: dict = {}

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            captured["max_layer"] = max_layer
            r = StabilityReport.empty()
            r.failed_layer = failed_layer
            r.failed_reason = f"synthetic L{failed_layer} failure"
            if layer_bl42_payload is not None:
                r.layer_bl42 = layer_bl42_payload
            return r

        monkeypatch.setattr(opt, "brady2d_stability_score", fake_score)
        f = make_objective(
            "E4", "classical", "layer_bl42.max_spectral_abscissa",
        )
        val = f(np.array([-0.77, 0.16]))
        assert captured["max_layer"] == 3
        assert val == expected

    def test_default_gate_for_layer1_objective_no_gate(self):
        # layer1.* → max_layer=1, gate_layer=0 (degenerate no-gate case:
        # failed_layer starts at 1, so ``failed_layer <= 0`` is never
        # true).  Verify the closure works end-to-end at a known-feasible
        # σ and returns a finite GV error.
        f = make_objective("E2", "tension", "layer1.boundary_gv_err")
        val = f(np.array([3.0]))
        assert np.isfinite(val)
        assert val >= 0.0

    def test_explicit_gate_layer_preserved(self, monkeypatch):
        # Passing gate_layer=3 explicitly with a layer6.* objective must
        # override the auto-infer.  Simulate an L5 failure together with a
        # populated layer6 payload: with gate_layer=3, the closure does
        # NOT gate (3 < 5), and extract_field returns the stored value.
        import stencil_gen.optimizer as opt

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            r = StabilityReport.empty()
            r.failed_layer = 5
            r.failed_reason = "synthetic"
            r.layer6 = {"transient_growth_bound": 42.0}
            return r

        monkeypatch.setattr(opt, "brady2d_stability_score", fake_score)
        f = make_objective(
            "E4", "classical", "layer6.transient_growth_bound",
            gate_layer=3,
        )
        val = f(np.array([-0.77, 0.16]))
        assert val == 42.0

    def test_no_auto_infer_raises_on_unknown_field(self):
        # A field with no recognised layer prefix still raises from
        # _infer_max_layer before gate_layer is considered.
        with pytest.raises(ValueError, match="cannot infer max_layer"):
            make_objective("E4", "classical", "bogus_field")

    def test_bl42_l3r_failure_returns_finite(self, monkeypatch):
        # Plan 45.0e: the self-gate +inf trap.
        # layer_bl42 is layer 3.  Under the OLD hard-coded gate_layer=3,
        # an L3r failure (failed_layer=3) would satisfy
        # ``failed_layer <= gate_layer`` and gate to +inf, even though the
        # payload the user is optimising is *exactly* the one populated by
        # that "failing" layer.  Under the NEW auto-infer gate_layer=2,
        # an L3 failure does NOT gate and ``extract_field`` returns the
        # stored value.  This is the exact scenario 45.0b unblocks
        # (the 2634-eval +inf DE run documented in
        # ``docs/handoff/next_steps.md``).
        #
        # Cross-reference: ``test_default_gate_for_bl42_objective`` (above)
        # covers the same L3-failure scenario as one parametrized sub-case
        # alongside the L2-failure (at-gate) case.  This standalone form is
        # retained for the named-scenario history of plan 45.0e.
        import stencil_gen.optimizer as opt

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            r = StabilityReport.empty()
            r.failed_layer = 3
            r.failed_reason = "synthetic L3r failure"
            r.layer_bl42 = {"max_spectral_abscissa": 5.0}
            return r

        monkeypatch.setattr(opt, "brady2d_stability_score", fake_score)
        f = make_objective(
            "E4", "classical", "layer_bl42.max_spectral_abscissa",
        )
        val = f(np.array([-0.77, 0.16]))
        assert val == 5.0

    def test_layer6_lower_failure_with_populated_payload(self, monkeypatch):
        # Plan 45.0e: distinguish below-gate vs above-gate failures when the
        # user passes ``--max-layer`` deeper than the objective's native
        # layer.  With explicit max_layer=7 and a layer6.* objective,
        # auto-infer gives gate_layer=6: every failure strictly before the
        # final layer gates, but a failure at layer 7 (above the gate) does
        # NOT gate, so the optimiser still sees the populated layer6 value.
        #
        # Under the OLD hardcoded gate_layer=3, both failed_layer=5 and
        # failed_layer=7 would fall outside the gate (5 > 3 and 7 > 3), so
        # neither case would gate — and the new-vs-old behavioural split
        # would be invisible.  The split below is visible only under the
        # new auto-infer.
        import stencil_gen.optimizer as opt

        state: dict = {}

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit):
            r = StabilityReport.empty()
            r.failed_layer = state["failed_layer"]
            r.failed_reason = "synthetic"
            r.layer5 = {"max_aligned_error": 1.0}
            r.layer6 = {"transient_growth_bound": 42.0}
            return r

        monkeypatch.setattr(opt, "brady2d_stability_score", fake_score)
        f = make_objective(
            "E4", "classical", "layer6.transient_growth_bound",
            max_layer=7,
        )

        # Below-gate failure: gates.  (Old hardcoded gate_layer=3 would
        # NOT gate here — this is the behavioural split.)
        state["failed_layer"] = 5
        assert f(np.array([-0.77, 0.16])) == float("inf")

        # Above-gate failure: does not gate; closure returns the populated
        # layer6 value so the optimiser sees a finite landscape.
        state["failed_layer"] = 7
        assert f(np.array([-0.77, 0.16])) == 42.0


class TestRunScipyLocal:
    """Driver-level tests for :func:`run_scipy_local`.

    Uses simple analytic objectives (quadratics) so the tests run fast and do
    not depend on the Brady-Livescu pipeline.  End-to-end tests against
    ``make_objective`` live further down so that a failure there is clearly
    tagged as an integration issue, not a driver bug.
    """

    @staticmethod
    def _quadratic(x: np.ndarray) -> float:
        # Minimum at [3.0] on the [0, 10] interval.
        return float((x[0] - 3.0) ** 2)

    def test_nelder_mead_converges_on_quadratic(self):
        r = run_scipy_local(
            self._quadratic,
            x0=np.array([5.0]),
            bounds=[(0.0, 10.0)],
            method="Nelder-Mead",
            max_evals=100,
        )
        assert r.method == "Nelder-Mead"
        assert r.converged
        assert np.isfinite(r.best_objective)
        assert r.best_x[0] == pytest.approx(3.0, abs=1e-3)
        assert r.best_objective == pytest.approx(0.0, abs=1e-6)
        assert r.n_evals > 0
        assert r.compute_time >= 0.0
        assert len(r.history) > 0
        # history entries should be (ndarray, float)
        x0, f0 = r.history[0]
        assert isinstance(x0, np.ndarray)
        assert isinstance(f0, float)

    def test_nelder_mead_history_records_every_eval(self):
        r = run_scipy_local(
            self._quadratic,
            x0=np.array([5.0]),
            bounds=[(0.0, 10.0)],
            method="Nelder-Mead",
            max_evals=50,
        )
        # The recorder should capture every objective call — not just
        # iteration endpoints.
        assert len(r.history) == r.n_evals

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            run_scipy_local(
                self._quadratic,
                x0=np.array([5.0]),
                bounds=[(0.0, 10.0)],
                method="BFGS",
            )

    def test_rejects_bounds_length_mismatch(self):
        with pytest.raises(ValueError, match="bounds length"):
            run_scipy_local(
                self._quadratic,
                x0=np.array([5.0]),
                bounds=[(0.0, 10.0), (0.0, 10.0)],
                method="Nelder-Mead",
            )

    def test_nelder_mead_returns_inf_when_only_infeasible(self):
        # Objective that is +inf everywhere — optimizer cannot converge.
        r = run_scipy_local(
            lambda x: float("inf"),
            x0=np.array([5.0]),
            bounds=[(0.0, 10.0)],
            method="Nelder-Mead",
            max_evals=30,
        )
        assert not np.isfinite(r.best_objective)
        assert not r.converged

    def test_nelder_mead_on_make_objective(self):
        # Integration: local optimize of the real tension-E4 objective starting
        # from a nearby feasible point.  Gate at L2 (not L3) because tension
        # σ~3.0 fails L3r (BL42 reflecting-hyperbolic).
        f = make_objective(
            "E4", "tension", "layer1.boundary_gv_err",
            gate_layer=2, max_layer=2,
        )
        x0 = np.array([2.0])
        f0 = f(x0)
        r = run_scipy_local(
            f, x0=x0, bounds=[(0.5, 20.0)],
            method="Nelder-Mead", max_evals=80,
        )
        assert np.isfinite(r.best_objective)
        assert r.best_objective <= f0 + 1e-12


@pytest.mark.skipif(not _COBYQA_AVAILABLE, reason="COBYQA requires scipy >= 1.14")
class TestRunScipyLocalCOBYQA:
    @staticmethod
    def _quadratic(x: np.ndarray) -> float:
        return float((x[0] - 3.0) ** 2)

    def test_cobyqa_converges_on_quadratic(self):
        r = run_scipy_local(
            self._quadratic,
            x0=np.array([5.0]),
            bounds=[(0.0, 10.0)],
            method="COBYQA",
            max_evals=100,
        )
        assert r.method == "COBYQA"
        assert r.converged
        assert r.best_x[0] == pytest.approx(3.0, abs=1e-3)

    def test_cobyqa_converges_on_tension_e4(self):
        # Plan 43.3b: COBYQA drives the real tension-E4 objective to a finite,
        # feasible minimum.  Gate at L2 (not L3) because tension σ~2–3
        # fails L3r (BL42 reflecting-hyperbolic).
        f = make_objective(
            "E4", "tension", "layer1.boundary_gv_err",
            gate_layer=2, max_layer=2,
        )
        x0 = np.array([2.0])
        f0 = f(x0)
        r = run_scipy_local(
            f,
            x0=x0,
            bounds=[(0.5, 20.0)],
            method="COBYQA",
            max_evals=120,
        )
        assert np.isfinite(r.best_objective)
        assert r.best_objective <= f0 + 1e-9
        assert 0.5 <= r.best_x[0] <= 20.0


class TestMultiStart:
    """Tests for :func:`multi_start_optimize` (plan 43.4)."""

    @staticmethod
    def _quadratic(x: np.ndarray) -> float:
        # Minimum at [3.0] on [0, 10].
        return float((x[0] - 3.0) ** 2)

    def test_multi_start_converges_on_quadratic(self):
        r = multi_start_optimize(
            self._quadratic,
            bounds=[(0.0, 10.0)],
            n_restarts=4,
            method="Nelder-Mead",
            seed=0,
            max_evals=50,
        )
        assert r.method == "multi-start"
        assert r.converged
        assert np.isfinite(r.best_objective)
        assert r.best_x[0] == pytest.approx(3.0, abs=1e-3)
        # n_evals sums across restarts; history concatenates.
        assert r.n_evals > 0
        assert len(r.history) == r.n_evals
        assert r.extras["inner_method"] == "Nelder-Mead"
        assert r.extras["n_restarts"] == 4
        assert r.extras["n_feasible_restarts"] >= 1

    def test_multi_start_deterministic(self):
        r1 = multi_start_optimize(
            self._quadratic,
            bounds=[(0.0, 10.0)],
            n_restarts=3,
            method="Nelder-Mead",
            seed=42,
            max_evals=40,
        )
        r2 = multi_start_optimize(
            self._quadratic,
            bounds=[(0.0, 10.0)],
            n_restarts=3,
            method="Nelder-Mead",
            seed=42,
            max_evals=40,
        )
        np.testing.assert_array_equal(r1.best_x, r2.best_x)
        assert r1.best_objective == r2.best_objective
        assert r1.n_evals == r2.n_evals

    def test_multi_start_handles_fully_infeasible(self):
        r = multi_start_optimize(
            lambda x: float("inf"),
            bounds=[(0.0, 10.0)],
            n_restarts=3,
            method="Nelder-Mead",
            seed=0,
            max_evals=20,
        )
        assert not np.isfinite(r.best_objective)
        assert not r.converged
        assert r.extras["n_feasible_restarts"] == 0

    def test_multi_start_rejects_zero_restarts(self):
        with pytest.raises(ValueError, match="n_restarts must be >= 1"):
            multi_start_optimize(
                self._quadratic,
                bounds=[(0.0, 10.0)],
                n_restarts=0,
            )

    def test_multi_start_rejects_empty_bounds(self):
        with pytest.raises(ValueError, match="bounds must be non-empty"):
            multi_start_optimize(
                self._quadratic,
                bounds=[],
                n_restarts=2,
            )

    def test_multi_start_finds_feasible_optimum(self):
        # Plan 43.4b: tension E4 against layer3.max_stab_eig.  Gate at L2
        # (not L3) because tension σ~3 fails L3r (BL42 reflecting-hyperbolic).
        # layer3 is still populated since L3 runs before L3r in the cascade.
        f = make_objective(
            "E4", "tension", "layer3.max_stab_eig",
            gate_layer=2, max_layer=3,
        )
        bounds = [(0.5, 20.0)]
        r = multi_start_optimize(
            f,
            bounds=bounds,
            n_restarts=4,
            method="Nelder-Mead",
            seed=0,
            max_evals=60,
        )
        assert np.isfinite(r.best_objective)
        assert bounds[0][0] <= r.best_x[0] <= bounds[0][1]
        # Compare against the best initial-point value across restarts.
        sampler = qmc.Sobol(d=1, seed=0)
        x0s = qmc.scale(
            sampler.random(4),
            np.array([bounds[0][0]]),
            np.array([bounds[0][1]]),
        )
        best_f0 = min(float(f(x0)) for x0 in x0s)
        assert r.best_objective <= best_f0 + 1e-9


class TestSHGO:
    """Tests for :func:`run_scipy_shgo` (plan 43.5a)."""

    @staticmethod
    def _quadratic(x: np.ndarray) -> float:
        # Single global minimum at [3.0] on [0, 10].
        return float((x[0] - 3.0) ** 2)

    @staticmethod
    def _two_basin(x: np.ndarray) -> float:
        # Two distinct local minima: one at x=2 (shallow) and one at x=7
        # (deep global).  Constructed so SHGO will find both basins.
        return float(
            0.3 * (x[0] - 2.0) ** 2 * (x[0] < 4.5)
            + ((x[0] - 7.0) ** 2 + 0.1) * (x[0] >= 4.5)
        )

    def test_shgo_converges_on_quadratic(self):
        r = run_scipy_shgo(
            self._quadratic,
            bounds=[(0.0, 10.0)],
            n=20,
            iters=2,
        )
        assert r.method == "SHGO"
        assert r.converged
        assert np.isfinite(r.best_objective)
        assert r.best_x[0] == pytest.approx(3.0, abs=1e-3)
        assert r.best_objective == pytest.approx(0.0, abs=1e-6)
        assert r.n_evals > 0
        assert r.compute_time >= 0.0
        # Extras carry the local-minima table.
        assert "n_local_minima" in r.extras
        assert r.extras["n_local_minima"] >= 1
        assert len(r.extras["local_minima"]) == r.extras["n_local_minima"]

    def test_shgo_records_local_minima(self):
        # Multi-basin landscape — SHGO should discover at least one minimum,
        # and the extras["local_minima"] table should be parseable.
        r = run_scipy_shgo(
            self._two_basin,
            bounds=[(0.0, 10.0)],
            n=30,
            iters=2,
        )
        assert r.extras["n_local_minima"] >= 1
        for x, fv in r.extras["local_minima"]:
            assert isinstance(x, np.ndarray)
            assert isinstance(fv, float)
            assert 0.0 <= x[0] <= 10.0

    def test_shgo_rejects_empty_bounds(self):
        with pytest.raises(ValueError, match="bounds must be non-empty"):
            run_scipy_shgo(self._quadratic, bounds=[])

    def test_shgo_handles_fully_infeasible(self):
        r = run_scipy_shgo(
            lambda x: float("inf"),
            bounds=[(0.0, 10.0)],
            n=10,
            iters=1,
        )
        assert not np.isfinite(r.best_objective)
        assert not r.converged
        assert r.method == "SHGO"
        # Fallback best_x is the bound midpoint — a sensible placeholder.
        assert r.best_x.shape == (1,)


class TestDE:
    """Tests for :func:`run_scipy_de` (plan 43.5b)."""

    @staticmethod
    def _quadratic(x: np.ndarray) -> float:
        return float((x[0] - 3.0) ** 2)

    @staticmethod
    def _rosenbrock(x: np.ndarray) -> float:
        # Classic 2D Rosenbrock, minimum at (1, 1).
        return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)

    def test_de_converges_on_quadratic(self):
        r = run_scipy_de(
            self._quadratic,
            bounds=[(0.0, 10.0)],
            popsize=8,
            maxiter=100,
            seed=0,
        )
        assert r.method == "DE"
        assert np.isfinite(r.best_objective)
        # Population-convergence tolerance means DE may stop with success=False
        # if maxiter runs out before the population collapses, even when the
        # polish pass has already pinned the minimum — so require finite
        # convergence-to-a-known-optimum rather than ``result.success``.
        assert r.best_x[0] == pytest.approx(3.0, abs=1e-3)
        assert r.best_objective == pytest.approx(0.0, abs=1e-6)
        assert r.n_evals > 0
        assert r.compute_time >= 0.0
        assert r.extras["popsize"] == 8
        assert r.extras["maxiter"] == 100
        assert r.extras["seed"] == 0
        assert r.extras["strategy"] == "best1bin"

    def test_de_deterministic(self):
        kwargs = dict(bounds=[(-5.0, 5.0), (-5.0, 5.0)], popsize=8, maxiter=10, seed=42)
        r1 = run_scipy_de(self._rosenbrock, **kwargs)
        r2 = run_scipy_de(self._rosenbrock, **kwargs)
        assert r1.best_objective == pytest.approx(r2.best_objective, rel=0, abs=1e-12)
        assert np.allclose(r1.best_x, r2.best_x)
        assert r1.n_evals == r2.n_evals

    def test_de_records_history(self):
        r = run_scipy_de(
            self._quadratic,
            bounds=[(0.0, 10.0)],
            popsize=5,
            maxiter=5,
            seed=0,
        )
        assert len(r.history) > 0
        for x, fv in r.history:
            assert isinstance(x, np.ndarray)
            assert isinstance(fv, float)
            assert 0.0 <= x[0] <= 10.0

    def test_de_rejects_empty_bounds(self):
        with pytest.raises(ValueError, match="bounds must be non-empty"):
            run_scipy_de(self._quadratic, bounds=[])

    def test_de_rejects_bad_popsize(self):
        with pytest.raises(ValueError, match="popsize must be >= 1"):
            run_scipy_de(self._quadratic, bounds=[(0.0, 1.0)], popsize=0)

    def test_de_rejects_bad_maxiter(self):
        with pytest.raises(ValueError, match="maxiter must be >= 1"):
            run_scipy_de(self._quadratic, bounds=[(0.0, 1.0)], maxiter=0)

    def test_de_handles_fully_infeasible(self):
        r = run_scipy_de(
            lambda x: float("inf"),
            bounds=[(0.0, 1.0)],
            popsize=4,
            maxiter=3,
            seed=0,
        )
        assert not np.isfinite(r.best_objective)
        assert r.converged is False
        assert r.best_x.shape == (1,)
        assert len(r.history) > 0


class TestRunScipyLocalCOBYQAUnavailable:
    def test_cobyqa_unavailable_raises_runtime_error(self, monkeypatch):
        import stencil_gen.optimizer as opt

        monkeypatch.setattr(opt, "_COBYQA_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="COBYQA requires scipy"):
            opt.run_scipy_local(
                lambda x: float(x[0] ** 2),
                x0=np.array([1.0]),
                bounds=[(-5.0, 5.0)],
                method="COBYQA",
            )


class TestGlobalOptimizers:
    """Integration tests for :func:`run_scipy_shgo` / :func:`run_scipy_de`
    against real :mod:`brady2d_stability` objectives (plan 43.5c)."""

    def test_shgo_finds_tension_optimum(self):
        # 1D tension E4 against layer3.max_stab_eig.  Gate at L2 (not L3)
        # because tension fails L3r (BL42 reflecting-hyperbolic).
        f = make_objective(
            "E4", "tension", "layer3.max_stab_eig",
            gate_layer=2, max_layer=3,
        )
        bounds = [(0.5, 20.0)]
        r = run_scipy_shgo(f, bounds=bounds, n=8, iters=1)
        assert np.isfinite(r.best_objective)
        assert bounds[0][0] <= r.best_x[0] <= bounds[0][1]
        assert r.extras["n_local_minima"] >= 1

    def test_de_finds_tension_optimum(self):
        # Same objective via differential_evolution.  Gate at L2 (not L3)
        # because tension fails L3r (BL42 reflecting-hyperbolic).
        f = make_objective(
            "E4", "tension", "layer3.max_stab_eig",
            gate_layer=2, max_layer=3,
        )
        bounds = [(0.5, 20.0)]
        r = run_scipy_de(f, bounds=bounds, popsize=6, maxiter=8, seed=0)
        assert np.isfinite(r.best_objective)
        assert bounds[0][0] <= r.best_x[0] <= bounds[0][1]

    @pytest.mark.slow
    def test_shgo_2d_classical_alpha(self):
        # 2D E4 classical-α.  Plan 43.9a widened ``DEFAULT_BOUNDS[("E4",
        # "classical")]`` to ``[(-2, 2), (0.05, 2)]`` — dropping the C++
        # cut-cell 197/288 floor since L1–L7 run on uniform grids where that
        # constraint doesn't apply.  This test still uses the narrower
        # relaxed bounds ``[(-1.2, -0.3), (0.05, 0.4)]`` to keep SHGO's
        # simplicial budget (n=6, iters=1) within the ~30 s time budget
        # while landing in the Brady-Livescu basin; a test on full
        # DEFAULT_BOUNDS lives in :class:`TestClassicalAlphaBounds`.
        f = make_objective(
            "E4", "classical", "layer3.max_stab_eig",
            gate_layer=3, max_layer=3,
        )
        bounds = [(-1.2, -0.3), (0.05, 0.4)]
        r = run_scipy_shgo(f, bounds=bounds, n=6, iters=1)
        assert np.isfinite(r.best_objective), (
            "SHGO should land on at least one feasible minimum in the "
            "Brady-Livescu-adjacent region"
        )
        # Bound-respecting.
        assert bounds[0][0] <= r.best_x[0] <= bounds[0][1]
        assert bounds[1][0] <= r.best_x[1] <= bounds[1][1]
        assert r.extras["n_local_minima"] >= 1
        # Loose (within-basin) comparison against the Brady-Livescu stored
        # α ≈ [-0.7733, 0.1624].  A 0.5 L∞ tolerance keeps this a
        # containment check, not an identity check (matching the 43.9d
        # convention).
        published = np.array([-0.7733323791884821, 0.1623961700641681])
        assert np.max(np.abs(r.best_x - published)) < 0.5


class TestStaged:
    """Tests for :func:`run_staged_optimize` (plan 43.6)."""

    def test_staged_rejects_shallow_validator(self):
        with pytest.raises(ValueError, match="validator_max_layer"):
            run_staged_optimize(
                scheme="E4",
                kernel="tension",
                report_field="layer3.max_stab_eig",
                bounds=[(0.5, 20.0)],
                inner_gate=3,
                inner_max_layer=3,
                validator_max_layer=2,
            )

    def test_staged_rejects_inner_shallower_than_gate(self):
        with pytest.raises(ValueError, match="inner_max_layer"):
            run_staged_optimize(
                scheme="E4",
                kernel="tension",
                report_field="layer3.max_stab_eig",
                bounds=[(0.5, 20.0)],
                inner_gate=3,
                inner_max_layer=2,
                validator_max_layer=3,
            )

    def test_staged_rejects_zero_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            run_staged_optimize(
                scheme="E4",
                kernel="tension",
                report_field="layer3.max_stab_eig",
                bounds=[(0.5, 20.0)],
                top_k=0,
            )

    @pytest.mark.slow
    def test_staged_tension_e4_convergence(self):
        # Inner/validator at the same layer so the "improves on or ties"
        # invariant is a pure re-ranking check: validator best <= inner best
        # at the same x is tautological here, but the staged pipeline must
        # still deliver a finite feasible winner, bound-respecting, and
        # populate both best_params and best_report.  (The earlier
        # specific-σ acceptance was dropped per 43.3c.)
        bounds = [(0.5, 20.0)]
        r = run_staged_optimize(
            scheme="E4",
            kernel="tension",
            report_field="layer3.max_stab_eig",
            bounds=bounds,
            inner_gate=2,
            inner_max_layer=3,
            validator_max_layer=3,
            top_k=3,
            method="Nelder-Mead",
            n_restarts=3,
            seed=0,
            max_evals=40,
        )
        assert r.method == "staged"
        assert r.converged
        assert np.isfinite(r.best_objective)
        assert bounds[0][0] <= r.best_x[0] <= bounds[0][1]
        assert r.best_params == {"sigma": float(r.best_x[0])}
        # Validator-picked winner is at least as good as the inner-stage
        # best (both measured at the same L3 field, since they share the
        # same max_layer here).
        assert r.best_objective <= r.extras["inner_best_objective"] + 1e-9
        # Validator ranking is populated and sorted ascending.
        ranking = r.extras["validator_ranking"]
        assert len(ranking) >= 1
        ranking_f = [fv for (_x, fv) in ranking]
        assert ranking_f == sorted(ranking_f)
        # Serialized report has at least the gate layers.
        assert "layer3" in r.best_report

    def test_staged_validator_reorders(self, monkeypatch):
        # Deterministic synthetic re-order test (plan 43.6d): stub both
        # ``multi_start_optimize`` (to return a canned inner history) and
        # ``brady2d_stability_score`` (to give validator-depth rankings that
        # disagree with the inner ranking).  This replaces the previous
        # tension-E4 L6 integration test whose outcome was data-dependent and
        # whose assertion only checked that ``stage`` was populated with one
        # of two possible values — a regression that silently made the
        # validator mirror the inner winner would have passed.
        #
        # Design:
        #   - inner ranks A=2.0 best on ``layer3.max_stab_eig`` (lowest value),
        #     then B=8.0, then C=5.0.
        #   - validator ranks B=8.0 best on ``layer6.transient_growth_bound``
        #     (quadratic well centered at 8.0).
        #   - With ``top_k=3`` the validator sees all three; its winner is B,
        #     distinct from the inner's winner A, so ``stage`` must be
        #     ``"validated"`` and ``best_x`` must be B.
        import stencil_gen.optimizer as opt_mod

        A = np.array([2.0])
        B = np.array([8.0])
        C = np.array([5.0])

        canned_history = [
            (A.copy(), -0.5),
            (B.copy(), -0.3),
            (C.copy(), -0.1),
            (np.array([9.0]), 0.2),
        ]
        canned_inner = OptimizeResult(
            best_params={"sigma": 2.0},
            best_x=A.copy(),
            best_objective=-0.5,
            best_report={},
            method="Nelder-Mead",
            converged=True,
            n_evals=4,
            compute_time=0.0,
            history=canned_history,
            extras={
                "inner_method": "Nelder-Mead",
                "n_restarts": 4,
                "seed": 0,
                "n_feasible_restarts": 4,
            },
        )

        def fake_multi_start(f, bounds, **kwargs):
            return canned_inner

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit=True):
            # Inner is bypassed via fake_multi_start; the validator stage is
            # the only caller that reaches here in this test.
            sigma = float(params["sigma"])
            tgb = (sigma - 8.0) ** 2
            return StabilityReport(
                layer1={"boundary_gv_err": 1e-4},
                layer3={"max_stab_eig": -0.3},
                layer6={
                    "transient_growth_bound": tgb,
                    "spectral_abscissa": -0.1,
                    "kreiss_constant": 1.0,
                },
                failed_layer=None,
                overall_verdict="pass",
            )

        monkeypatch.setattr(opt_mod, "multi_start_optimize", fake_multi_start)
        monkeypatch.setattr(opt_mod, "brady2d_stability_score", fake_score)

        r = run_staged_optimize(
            scheme="E4",
            kernel="tension",
            report_field="layer6.transient_growth_bound",
            bounds=[(0.5, 20.0)],
            inner_gate=3,
            inner_max_layer=3,
            validator_max_layer=6,
            top_k=3,
        )

        assert r.method == "staged"
        # Validator reordered: winner is B, not the inner's A.
        assert r.extras["stage"] == "validated"
        np.testing.assert_allclose(r.best_x, B)
        assert not np.allclose(r.best_x, canned_inner.best_x)
        # Validator's transient_growth_bound at σ=8 is exactly 0.
        assert r.best_objective == pytest.approx(0.0)
        # Inner fallback field was used (report_field is L6-only).
        assert r.extras["inner_field"] == "layer3.max_stab_eig"
        assert r.extras["validator_max_layer"] == 6
        # Inner diagnostics preserved in extras.
        np.testing.assert_allclose(r.extras["inner_best_x"], A)
        assert r.extras["inner_best_objective"] == pytest.approx(-0.5)
        # Validator ranking sorted ascending; B is first, C second, A last.
        ranking = r.extras["validator_ranking"]
        assert len(ranking) == 3
        ranking_f = [fv for (_x, fv) in ranking]
        assert ranking_f == sorted(ranking_f)
        np.testing.assert_allclose(ranking[0][0], B)
        # best_report carries the L6 payload from the validator run.
        assert "layer6" in r.best_report

    def test_staged_validator_all_blowups(self, monkeypatch):
        # Every validator re-run raises: the staged pipeline must return the
        # inner-stage result wrapped with ``method="staged"``,
        # ``stage="inner"``, ``converged=False`` (the validator did not
        # confirm), and the fallback ``extras`` must carry the
        # ``inner_best_objective`` / ``inner_best_x`` keys that the
        # success-path extras populates — otherwise downstream callers and
        # tests that read those keys ``KeyError`` on the fallback branch.
        import stencil_gen.optimizer as opt_mod

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit=True):
            if max_layer <= 3:
                # Inner stage: feasible, L3 max_stab_eig varies with sigma so
                # multi_start_optimize has a real objective to descend on.
                sigma = float(params["sigma"])
                return StabilityReport(
                    layer1={"boundary_gv_err": 1e-4},
                    layer3={"max_stab_eig": -0.1 + 1e-3 * (sigma - 3.0) ** 2},
                    failed_layer=None,
                    overall_verdict="pass",
                )
            # Validator stage (max_layer=6): blow up every time.
            raise RuntimeError("validator blew up")

        monkeypatch.setattr(opt_mod, "brady2d_stability_score", fake_score)

        r = run_staged_optimize(
            scheme="E4",
            kernel="tension",
            report_field="layer6.transient_growth_bound",
            bounds=[(0.5, 20.0)],
            inner_gate=3,
            inner_max_layer=3,
            validator_max_layer=6,
            top_k=3,
            method="Nelder-Mead",
            n_restarts=2,
            seed=0,
            max_evals=20,
        )

        assert r.method == "staged"
        assert r.extras["stage"] == "inner"
        assert r.converged is False
        # Fallback extras parity with the success path: both keys must be
        # present and mirror the inner result.
        assert "inner_best_objective" in r.extras
        assert "inner_best_x" in r.extras
        # Fallback builds the result via ``replace(inner_result, method="staged",
        # converged=False, ...)`` without touching ``best_objective`` / ``best_x``,
        # so the exposed fields must match the ``inner_*`` extras.
        assert r.extras["inner_best_objective"] == pytest.approx(r.best_objective)
        assert np.allclose(r.extras["inner_best_x"], r.best_x)
        assert r.extras["inner_best_x"].shape == r.best_x.shape
        assert np.isfinite(r.extras["inner_best_objective"])
        assert r.extras["inner_best_x"].shape == (1,)
        # Validator ranking entries are all infeasible.
        ranking = r.extras["validator_ranking"]
        assert len(ranking) >= 1
        assert all(not np.isfinite(fv) for (_x, fv) in ranking)

    def test_staged_records_cpp_cutcell_flag_for_e4_classical(self, monkeypatch):
        # Fast synthetic coverage for the ``cpp_cutcell_violates_197_288``
        # diagnostic flag (plan 43.9b-r1 option a).  The flag must be
        # populated on E4 classical results in both the success and
        # fallback paths, and must encode ``best_x[1] < 197/288``.
        import stencil_gen.optimizer as opt_mod

        # Canned 2D inner history with the winner below the C++ floor
        # (α₁ = 0.1 < 197/288 ≈ 0.684 → flag should be True).
        winner_below = np.array([-0.8, 0.1])
        canned_history = [(winner_below.copy(), -0.5)]
        canned_inner = OptimizeResult(
            best_params={"alpha": [-0.8, 0.1]},
            best_x=winner_below.copy(),
            best_objective=-0.5,
            best_report={},
            method="Nelder-Mead",
            converged=True,
            n_evals=1,
            compute_time=0.0,
            history=canned_history,
            extras={
                "inner_method": "Nelder-Mead",
                "n_restarts": 1,
                "seed": 0,
                "n_feasible_restarts": 1,
            },
        )

        def fake_multi_start(f, bounds, **kwargs):
            return canned_inner

        def fake_score_ok(scheme, kernel, params, *, max_layer, short_circuit=True):
            return StabilityReport(
                layer3={"max_stab_eig": -0.5},
                layer6={
                    "transient_growth_bound": 1.0,
                    "spectral_abscissa": -0.1,
                    "kreiss_constant": 1.0,
                },
                failed_layer=None,
                overall_verdict="pass",
            )

        monkeypatch.setattr(opt_mod, "multi_start_optimize", fake_multi_start)
        monkeypatch.setattr(opt_mod, "brady2d_stability_score", fake_score_ok)

        r_ok = run_staged_optimize(
            scheme="E4",
            kernel="classical",
            report_field="layer6.transient_growth_bound",
            bounds=[(-2.0, 2.0), (0.05, 2.0)],
            inner_gate=3,
            inner_max_layer=3,
            validator_max_layer=6,
            top_k=1,
        )
        assert "cpp_cutcell_violates_197_288" in r_ok.extras
        assert r_ok.extras["cpp_cutcell_violates_197_288"] is True

        # Fallback path: validator raises every time.  Flag must still be
        # recorded from the inner best_x.
        def fake_score_blowup(scheme, kernel, params, *, max_layer, short_circuit=True):
            if max_layer >= 6:
                raise RuntimeError("validator blew up")
            return fake_score_ok(scheme, kernel, params, max_layer=max_layer)

        monkeypatch.setattr(opt_mod, "brady2d_stability_score", fake_score_blowup)

        r_fb = run_staged_optimize(
            scheme="E4",
            kernel="classical",
            report_field="layer6.transient_growth_bound",
            bounds=[(-2.0, 2.0), (0.05, 2.0)],
            inner_gate=3,
            inner_max_layer=3,
            validator_max_layer=6,
            top_k=1,
        )
        assert r_fb.extras["stage"] == "inner"
        assert "cpp_cutcell_violates_197_288" in r_fb.extras
        assert r_fb.extras["cpp_cutcell_violates_197_288"] is True

        # Winner above the floor → flag is False.
        winner_above = np.array([-0.8, 0.9])
        canned_inner_above = replace(
            canned_inner,
            best_x=winner_above.copy(),
            history=[(winner_above.copy(), -0.5)],
            best_params={"alpha": [-0.8, 0.9]},
        )
        monkeypatch.setattr(
            opt_mod, "multi_start_optimize", lambda f, bounds, **kw: canned_inner_above
        )
        monkeypatch.setattr(opt_mod, "brady2d_stability_score", fake_score_ok)

        r_above = run_staged_optimize(
            scheme="E4",
            kernel="classical",
            report_field="layer6.transient_growth_bound",
            bounds=[(-2.0, 2.0), (0.05, 2.0)],
            inner_gate=3,
            inner_max_layer=3,
            validator_max_layer=6,
            top_k=1,
        )
        assert r_above.extras["cpp_cutcell_violates_197_288"] is False

    def test_staged_omits_cpp_cutcell_flag_for_other_kernels(self, monkeypatch):
        # Non-E4-classical results must not carry the diagnostic flag (keep
        # extras uncluttered for kernels where 197/288 is meaningless).
        import stencil_gen.optimizer as opt_mod

        canned_inner = OptimizeResult(
            best_params={"sigma": 2.0},
            best_x=np.array([2.0]),
            best_objective=-0.5,
            best_report={},
            method="Nelder-Mead",
            converged=True,
            n_evals=1,
            compute_time=0.0,
            history=[(np.array([2.0]), -0.5)],
            extras={
                "inner_method": "Nelder-Mead",
                "n_restarts": 1,
                "seed": 0,
                "n_feasible_restarts": 1,
            },
        )

        def fake_score(scheme, kernel, params, *, max_layer, short_circuit=True):
            return StabilityReport(
                layer3={"max_stab_eig": -0.5},
                layer6={
                    "transient_growth_bound": 1.0,
                    "spectral_abscissa": -0.1,
                    "kreiss_constant": 1.0,
                },
                failed_layer=None,
                overall_verdict="pass",
            )

        monkeypatch.setattr(
            opt_mod, "multi_start_optimize", lambda f, bounds, **kw: canned_inner
        )
        monkeypatch.setattr(opt_mod, "brady2d_stability_score", fake_score)

        r = run_staged_optimize(
            scheme="E4",
            kernel="tension",
            report_field="layer6.transient_growth_bound",
            bounds=[(0.5, 20.0)],
            inner_gate=3,
            inner_max_layer=3,
            validator_max_layer=6,
            top_k=1,
        )
        assert "cpp_cutcell_violates_197_288" not in r.extras


class TestStagedClassicalAlpha:
    """Single-seed staged run on E4_1 classical-α (plan 43.9b).

    Drives :func:`run_staged_optimize` against the full widened
    ``DEFAULT_BOUNDS[("E4", "classical")]`` envelope (plan 43.9a) with the
    inner gate at L3 and a ``layer6.transient_growth_bound`` validator.
    Records the L8 cut-cell diagnostic flag ``cpp_cutcell_violates_197_288``
    in ``r.extras`` for downstream plan-43.10 wiring; the flag is purely
    informational here and does not fail the test (see 43.9a-r1).
    """

    @pytest.mark.slow
    def test_staged_classical_e4_single_seed(self):
        bounds = DEFAULT_BOUNDS[("E4", "classical")]
        r = run_staged_optimize(
            scheme="E4",
            kernel="classical",
            report_field="layer6.transient_growth_bound",
            bounds=bounds,
            inner_gate=3,
            inner_max_layer=3,
            validator_max_layer=6,
            top_k=5,
            method="Nelder-Mead",
            n_restarts=20,
            seed=0,
            max_evals=60,
        )

        assert r.method == "staged"
        assert np.isfinite(r.best_objective), (
            "staged E4 classical-α on DEFAULT_BOUNDS should land on at least "
            "one feasible validator candidate (the Brady-Livescu basin "
            "α ≈ [-0.77, 0.16] sits inside the widened bounds)"
        )
        for i, (lo, hi) in enumerate(bounds):
            assert lo <= r.best_x[i] <= hi, (
                f"best_x[{i}]={r.best_x[i]} outside bound [{lo}, {hi}]"
            )
        # best_params is a 2-element ``alpha`` list built by
        # ``params_from_vector`` on the classical kernel.
        assert list(r.best_params.keys()) == ["alpha"]
        assert len(r.best_params["alpha"]) == 2

        # L8 cut-cell diagnostic flag — informational.  The 197/288 floor is a
        # C++-side constraint (non-zero psi denominator in ``E4_1.cpp``) that
        # the Python L1–L7 pipeline does not enforce, so an analytical winner
        # with ``α₁ < 197/288`` is expected and *not* a failure (see 43.9a).
        # ``run_staged_optimize`` records the flag itself for the E4 classical
        # kernel (plan 43.9b-r1 option a); the test only observes it.
        assert "cpp_cutcell_violates_197_288" in r.extras
        assert r.extras["cpp_cutcell_violates_197_288"] == (
            r.best_x[1] < 197.0 / 288.0
        )


class TestOptimizeCLI:
    """Smoke tests for ``sweeps.optimize`` (plan 43.7c).

    One subprocess test verifies the real ``python -m sweeps.optimize`` entry
    point end-to-end; the error-path tests call ``main`` in-process to keep
    the suite fast (parser.error raises SystemExit, which is what "exits
    non-zero" means in a subprocess).
    """

    @pytest.mark.slow
    def test_cli_tension_nelder_mead(self):
        """A tiny tension-E4 Nelder-Mead run completes and prints a summary.

        Marked slow: subprocess pays the SymPy cold-start tax (~5-7 min on
        first invocation in a fresh environment).
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        stencil_gen_dir = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env["SYMPY_CACHE_SIZE"] = env.get("SYMPY_CACHE_SIZE", "50000")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "sweeps.optimize",
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer3.max_stab_eig",
                "--gate-layer", "2",
                "--max-layer", "3",
                "--bounds", "0.5", "20",
                "--method", "Nelder-Mead",
                "--n-restarts", "1",
                "--max-evals", "10",
                "--seed", "0",
            ],
            cwd=str(stencil_gen_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, (
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        assert "best_objective" in proc.stdout
        assert "best_params" in proc.stdout
        assert "inf" not in proc.stdout, "all evaluations returned +inf (vacuous run)"

    def test_cli_rejects_bad_objective(self):
        """Unknown objective prefix cannot infer max_layer → SystemExit.

        ``bogus.field`` has neither a ``layerN.`` prefix nor an entry in
        ``_FIELD_LAYER_ALIAS``, so ``make_objective`` raises ``ValueError`` at
        construction (before any evaluation), which the CLI surfaces as
        ``parser.error`` → ``SystemExit(2)``.
        """
        from sweeps.optimize import main

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--scheme", "E4",
                    "--kernel", "tension",
                    "--objective", "bogus.field",
                    "--bounds", "0.5", "20",
                    "--method", "Nelder-Mead",
                    "--n-restarts", "1",
                    "--max-evals", "4",
                ]
            )
        # argparse.error exits with code 2.
        assert exc_info.value.code != 0

    def test_cli_rejects_kernel_bounds_dim_mismatch(self):
        """``--kernel classical --bounds 0.5 20`` (1D bounds for 2D kernel) errors."""
        from sweeps.optimize import main

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--scheme", "E4",
                    "--kernel", "classical",
                    "--objective", "layer3.max_stab_eig",
                    "--bounds", "0.5", "20",
                    "--method", "Nelder-Mead",
                    "--n-restarts", "1",
                    "--max-evals", "4",
                ]
            )
        assert exc_info.value.code != 0

    def test_cli_update_known_values_additive_and_drops_history(
        self, monkeypatch
    ):
        """``--update-known-values`` writes under ``brady2d_optima`` without
        touching unrelated keys, and omits ``history`` from the persisted form.

        Uses a monkey-patched ``_run_method`` so the test never enters the real
        SymPy pipeline — the goal is to pin the persistence contract in
        ``sweeps/optimize.py`` (plan item 43.8a), not to re-exercise the
        optimizers.
        """
        from sweeps import optimize as optimize_mod

        store: dict = {
            "brady2d_calibration": {"E4": {"tension": [1.0, 2.0]}},
            "brady2d_sweep": {"E4": {"tension": {"sigma": 3.0}}},
        }

        def fake_load() -> dict:
            # Shallow copy so the CLI's setdefault mutations go through
            # ``fake_save`` rather than silently aliasing ``store``.
            import copy
            return copy.deepcopy(store)

        def fake_save(data: dict) -> None:
            store.clear()
            store.update(data)

        monkeypatch.setattr(optimize_mod, "load_known_values", fake_load)
        monkeypatch.setattr(optimize_mod, "save_known_values", fake_save)

        canned = OptimizeResult(
            best_params={"sigma": 3.1},
            best_x=np.array([3.1]),
            best_objective=-1.5,
            best_report={"layer3": {"max_stab_eig": -1.5}},
            method="Nelder-Mead",
            converged=True,
            n_evals=17,
            compute_time=0.1,
            history=[
                (np.array([3.0]), -1.4),
                (np.array([3.1]), -1.5),
            ],
            extras={"n_restarts": 1},
        )
        monkeypatch.setattr(
            optimize_mod, "_run_method", lambda args, bounds: canned
        )

        rc = optimize_mod.main(
            [
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer3.max_stab_eig",
                "--bounds", "0.5", "20",
                "--method", "Nelder-Mead",
                "--n-restarts", "1",
                "--max-evals", "4",
                "--update-known-values",
            ]
        )
        assert rc == 0

        # Persisted under brady2d_optima[scheme][kernel][objective].
        opt = store["brady2d_optima"]["E4"]["tension"]["layer3.max_stab_eig"]
        assert opt["best_objective"] == pytest.approx(-1.5)
        assert opt["best_params"] == {"sigma": 3.1}
        assert opt["converged"] is True
        assert opt["n_evals"] == 17
        assert opt["method"] == "Nelder-Mead"
        assert opt["bounds"] == [[0.5, 20.0]]
        assert opt["best_x"] == [pytest.approx(3.1)]
        # history is intentionally omitted from the persisted form.
        assert "history" not in opt
        # Plan 43.8c: gate_layer and inferred max_layer round-trip (no
        # --max-layer was passed for this call, so max_layer is the inferred
        # layer3 from the "layer3.max_stab_eig" prefix).  Non-staged method,
        # so validator_max_layer must be absent.  Plan 45.0d: --gate-layer is
        # now auto-inferred as ``max(max_layer - 1, 0)`` when omitted, so the
        # persisted gate_layer reflects that (2 for an L3 objective).
        assert opt["gate_layer"] == 2
        assert opt["max_layer"] == 3
        assert "validator_max_layer" not in opt
        # Existing top-level keys are untouched.
        assert store["brady2d_calibration"] == {"E4": {"tension": [1.0, 2.0]}}
        assert store["brady2d_sweep"] == {"E4": {"tension": {"sigma": 3.0}}}

        # A second CLI call at a different objective must coexist with the
        # first under the same scheme/kernel bucket (additive behaviour).
        second = replace(canned, best_objective=-0.75, best_x=np.array([4.0]),
                         best_params={"sigma": 4.0})
        monkeypatch.setattr(
            optimize_mod, "_run_method", lambda args, bounds: second
        )
        rc2 = optimize_mod.main(
            [
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer6.transient_growth_bound",
                "--max-layer", "6",
                "--bounds", "0.5", "20",
                "--method", "Nelder-Mead",
                "--n-restarts", "1",
                "--max-evals", "4",
                "--update-known-values",
            ]
        )
        assert rc2 == 0
        kernel_bucket = store["brady2d_optima"]["E4"]["tension"]
        assert set(kernel_bucket.keys()) == {
            "layer3.max_stab_eig",
            "layer6.transient_growth_bound",
        }
        assert kernel_bucket["layer3.max_stab_eig"]["best_objective"] == \
            pytest.approx(-1.5)
        assert kernel_bucket["layer6.transient_growth_bound"][
            "best_objective"
        ] == pytest.approx(-0.75)
        # Plan 43.8c: explicit --max-layer 6 on the second call must round-trip
        # (no inference fallback).  Still non-staged, so no validator field.
        # Plan 45.0d: with --gate-layer omitted, it auto-infers to max_layer-1.
        second_opt = kernel_bucket["layer6.transient_growth_bound"]
        assert second_opt["gate_layer"] == 5
        assert second_opt["max_layer"] == 6
        assert "validator_max_layer" not in second_opt

        # Plan 43.8c: a staged call must round-trip gate_layer + max_layer
        # (inner depth) + validator_max_layer.  Use a fresh objective bucket so
        # this case is distinct from the two above.
        staged_canned = replace(
            canned,
            method="staged",
            best_objective=-2.25,
            best_x=np.array([5.0]),
            best_params={"sigma": 5.0},
        )
        monkeypatch.setattr(
            optimize_mod, "_run_method", lambda args, bounds: staged_canned
        )
        rc3 = optimize_mod.main(
            [
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer6.kreiss_constant",
                "--bounds", "0.5", "20",
                "--method", "staged",
                "--n-restarts", "1",
                "--max-evals", "4",
                "--validator-max-layer", "7",
                "--update-known-values",
            ]
        )
        assert rc3 == 0
        staged_opt = store["brady2d_optima"]["E4"]["tension"][
            "layer6.kreiss_constant"
        ]
        # No --max-layer passed and method is staged, so the inner-depth
        # default of 3 is persisted; validator_max_layer is the explicit 7.
        # Plan 45.0d: --gate-layer defaults to max(inner_max_layer - 1, 0) = 2.
        assert staged_opt["method"] == "staged"
        assert staged_opt["gate_layer"] == 2
        assert staged_opt["max_layer"] == 3
        assert staged_opt["validator_max_layer"] == 7

        # Plan 43.9b-r2: the cpp_cutcell_violates_197_288 diagnostic flag
        # (recorded in extras for E4 classical-alpha by
        # _record_cpp_cutcell_diagnostic) is serialised into the persisted
        # entry so 43.10 consumers reading known_values.json can see it
        # without recomputing.  An E4-classical OptimizeResult with the flag
        # set should round-trip; tension results (no such flag) must not
        # carry the key.
        cutcell_canned = OptimizeResult(
            best_params={"alpha": [-0.8, 0.1]},
            best_x=np.array([-0.8, 0.1]),
            best_objective=-1.0,
            best_report={"layer3": {"max_stab_eig": -1.0}},
            method="staged",
            converged=True,
            n_evals=5,
            compute_time=0.05,
            history=[(np.array([-0.8, 0.1]), -1.0)],
            extras={"cpp_cutcell_violates_197_288": True, "n_restarts": 1},
        )
        monkeypatch.setattr(
            optimize_mod, "_run_method", lambda args, bounds: cutcell_canned
        )
        rc4 = optimize_mod.main(
            [
                "--scheme", "E4",
                "--kernel", "classical",
                "--objective", "layer3.max_stab_eig",
                "--bounds", "-2", "2", "0.05", "2",
                "--method", "staged",
                "--n-restarts", "1",
                "--max-evals", "4",
                "--update-known-values",
            ]
        )
        assert rc4 == 0
        classical_opt = store["brady2d_optima"]["E4"]["classical"][
            "layer3.max_stab_eig"
        ]
        assert classical_opt["cpp_cutcell_violates_197_288"] is True

        # Tension entries (already persisted above without the flag in extras)
        # must not carry the key.
        assert "cpp_cutcell_violates_197_288" not in opt
        assert "cpp_cutcell_violates_197_288" not in second_opt
        assert "cpp_cutcell_violates_197_288" not in staged_opt


class TestAlphaBasinSurvey:
    """Synthetic coverage for ``run_survey`` / ``format_survey_table``
    (plan 43.9c-r1).

    The live pipeline is exercised by the slow ``43.9d`` test.  Here we
    monkey-patch :func:`run_staged_optimize` so each branch — basin
    clustering, winner-replaces-basin, fallback-infeasible, flag propagation,
    input validation, and markdown rendering — is deterministic and fast.
    """

    @staticmethod
    def _canned(
        *,
        best_x,
        best_objective,
        cpp_flag=None,
        stage="validated",
    ):
        x = np.asarray(best_x, dtype=float)
        extras = {"stage": stage, "inner_method": "Nelder-Mead"}
        if cpp_flag is not None:
            extras["cpp_cutcell_violates_197_288"] = cpp_flag
        return OptimizeResult(
            best_params={"alpha": x.tolist()} if x.shape[0] == 2 else {"sigma": float(x[0])},
            best_x=x,
            best_objective=float(best_objective),
            best_report={"layer6": {"transient_growth_bound": float(best_objective)}},
            method="staged",
            converged=np.isfinite(best_objective),
            n_evals=7,
            compute_time=0.01,
            history=[(x.copy(), float(best_objective))],
            extras=extras,
        )

    def test_run_survey_clusters_multiple_basins(self, monkeypatch):
        from stencil_gen.benchmarks import alpha_basin_survey as survey_mod

        # 4 seeds: seeds 0 and 2 cluster near [-0.80, 0.10] (two-decimal key),
        # seed 1 is a separate basin at [-0.30, 0.90], seed 3 infeasible.
        # Seed 2 has the *lower* objective, so the shared basin should
        # retain seed-2's alpha/objective (winner-replaces-basin branch).
        seed_to_result = {
            0: self._canned(best_x=[-0.803, 0.102], best_objective=5.0, cpp_flag=True),
            1: self._canned(best_x=[-0.300, 0.900], best_objective=2.0, cpp_flag=False),
            2: self._canned(best_x=[-0.797, 0.099], best_objective=1.0, cpp_flag=True),
            3: self._canned(
                best_x=[0.0, 0.0], best_objective=float("inf"), stage="inner"
            ),
        }

        def fake_staged(**kwargs):
            return seed_to_result[kwargs["seed"]]

        monkeypatch.setattr(survey_mod, "run_staged_optimize", fake_staged)

        r = survey_mod.run_survey(
            n_seeds=4,
            bounds=[(-2.0, 2.0), (0.05, 2.0)],
            cluster_decimals=2,
            base_seed=0,
        )

        assert r["n_feasible_seeds"] == 3
        assert r["n_distinct_basins"] == 2
        basins = r["basins"]
        # Sorted ascending by best_objective → seed-2 basin first (obj=1.0),
        # then seed-1 basin (obj=2.0).
        assert basins[0]["best_objective"] == pytest.approx(1.0)
        assert basins[1]["best_objective"] == pytest.approx(2.0)

        # First basin holds seeds {0, 2} with seed 2's alpha (the winner).
        assert basins[0]["n_seeds_in_basin"] == 2
        assert sorted(basins[0]["seeds"]) == [0, 2]
        np.testing.assert_allclose(basins[0]["alpha"], [-0.797, 0.099])

        # Second basin holds only seed 1.
        assert basins[1]["n_seeds_in_basin"] == 1
        assert basins[1]["seeds"] == [1]
        np.testing.assert_allclose(basins[1]["alpha"], [-0.300, 0.900])

        # Config round-trips all keys (including cluster_decimals).
        cfg = r["config"]
        for key in (
            "scheme",
            "kernel",
            "report_field",
            "bounds",
            "n_seeds",
            "base_seed",
            "cluster_decimals",
            "inner_gate",
            "inner_max_layer",
            "validator_max_layer",
            "top_k",
            "method",
            "n_restarts",
            "max_evals",
        ):
            assert key in cfg, f"config missing {key!r}"
        assert cfg["cluster_decimals"] == 2
        assert cfg["n_seeds"] == 4

        assert len(r["seed_results"]) == 4
        assert r["compute_time"] >= 0.0

    def test_run_survey_propagates_cpp_cutcell_flag(self, monkeypatch):
        from stencil_gen.benchmarks import alpha_basin_survey as survey_mod

        # Three distinct feasible basins with flag values {True, False, None}.
        seed_to_result = {
            0: self._canned(best_x=[-0.80, 0.10], best_objective=3.0, cpp_flag=True),
            1: self._canned(best_x=[-0.30, 0.90], best_objective=2.0, cpp_flag=False),
            2: self._canned(best_x=[0.50, 0.50], best_objective=1.0, cpp_flag=None),
        }

        def fake_staged(**kwargs):
            return seed_to_result[kwargs["seed"]]

        monkeypatch.setattr(survey_mod, "run_staged_optimize", fake_staged)

        r = survey_mod.run_survey(
            n_seeds=3,
            bounds=[(-2.0, 2.0), (0.05, 2.0)],
            cluster_decimals=2,
        )

        # Per-seed entries carry the flag untouched.
        seed_map = {e["seed"]: e for e in r["seed_results"]}
        assert seed_map[0]["cpp_cutcell_violates_197_288"] is True
        assert seed_map[1]["cpp_cutcell_violates_197_288"] is False
        assert seed_map[2]["cpp_cutcell_violates_197_288"] is None

        # Per-basin summaries carry the flag too (each basin is a single
        # seed here so the propagation is unambiguous).
        basin_by_alpha = {tuple(round(a, 2) for a in b["alpha"]): b for b in r["basins"]}
        assert basin_by_alpha[(-0.80, 0.10)]["cpp_cutcell_violates_197_288"] is True
        assert basin_by_alpha[(-0.30, 0.90)]["cpp_cutcell_violates_197_288"] is False
        assert basin_by_alpha[(0.50, 0.50)]["cpp_cutcell_violates_197_288"] is None

    def test_run_survey_all_infeasible_returns_empty_basins(self, monkeypatch):
        from stencil_gen.benchmarks import alpha_basin_survey as survey_mod

        def fake_staged(**kwargs):
            return self._canned(
                best_x=[0.0, 0.0], best_objective=float("inf"), stage="inner"
            )

        monkeypatch.setattr(survey_mod, "run_staged_optimize", fake_staged)

        r = survey_mod.run_survey(
            n_seeds=3,
            bounds=[(-2.0, 2.0), (0.05, 2.0)],
            cluster_decimals=2,
        )

        assert r["basins"] == []
        assert r["n_distinct_basins"] == 0
        assert r["n_feasible_seeds"] == 0
        assert len(r["seed_results"]) == 3
        assert all(not e["feasible"] for e in r["seed_results"])

        # format_survey_table must still render without raising; it should
        # include the header lines and an empty table body.
        table = survey_mod.format_survey_table(r)
        assert "distinct basins=0" in table
        assert "| α₀ | α₁" in table

    def test_run_survey_rejects_bad_inputs(self):
        from stencil_gen.benchmarks import alpha_basin_survey as survey_mod

        with pytest.raises(ValueError, match="n_seeds"):
            survey_mod.run_survey(n_seeds=0, bounds=[(-2.0, 2.0), (0.05, 2.0)])
        with pytest.raises(ValueError, match="cluster_decimals"):
            survey_mod.run_survey(
                n_seeds=1,
                bounds=[(-2.0, 2.0), (0.05, 2.0)],
                cluster_decimals=-1,
            )

    def test_format_survey_table_renders_header_and_rows(self):
        from stencil_gen.benchmarks import alpha_basin_survey as survey_mod

        fake = {
            "basins": [
                {
                    "alpha": [-0.7733, 0.1624],
                    "best_objective": 4.83,
                    "n_seeds_in_basin": 3,
                    "seeds": [0, 2, 5],
                    "cpp_cutcell_violates_197_288": True,
                },
                {
                    "alpha": [0.5, 0.9],
                    "best_objective": 7.21,
                    "n_seeds_in_basin": 2,
                    "seeds": [1, 3],
                    "cpp_cutcell_violates_197_288": False,
                },
                {
                    "alpha": [1.0, 0.2],
                    "best_objective": 12.0,
                    "n_seeds_in_basin": 1,
                    "seeds": [4],
                    "cpp_cutcell_violates_197_288": None,
                },
            ],
            "n_distinct_basins": 3,
            "n_feasible_seeds": 6,
            "compute_time": 12.3,
            "config": {
                "scheme": "E4",
                "kernel": "classical",
                "report_field": "layer6.transient_growth_bound",
                "n_seeds": 10,
            },
        }

        table = survey_mod.format_survey_table(fake)
        lines = table.splitlines()

        # Header: scheme, kernel, report_field, and the summary stats line
        # must all appear.
        header = lines[0]
        assert "E4" in header and "classical" in header
        assert "layer6.transient_growth_bound" in header
        stats = lines[1]
        assert "n_seeds=10" in stats
        assert "feasible=6" in stats
        assert "distinct basins=3" in stats
        assert "wall=12.3s" in stats

        # The 5-column schema.
        assert "| α₀ | α₁ | best_objective | n_seeds | 197/288 viol |" in lines[2]

        # viol cell renders yes / no / - for True / False / None.
        body = "\n".join(lines[4:])
        assert "yes" in body
        assert "no" in body
        assert " - " in body or "| -" in body


class TestAlphaSurveyVsPublished:
    """Live basin-survey assertion against Brady-Livescu's published E4 α
    (plan 43.9d).

    The plan cites ``stencil_gen/alpha_extraction.py`` as the published-value
    source, but the Brady-Livescu E4 α is stored in
    :data:`sweeps.brady2d_sweep.CLASSICAL_E4_ALPHA` (the codebase's canonical
    constant; ``sweeps/alpha_extraction.py`` only defines E2 production α's).
    We import from the canonical location and cross-check that at least one
    basin from the live survey sits within a 0.5 L∞ ball of it.  The paper
    does not claim uniqueness of this α (it reports 101 distinct E4 schemes
    at the full budget); this is a containment check, not identity.
    """

    @pytest.mark.slow
    def test_top_basin_within_published_l_infinity_ball(self):
        from stencil_gen.benchmarks import alpha_basin_survey as survey_mod
        from sweeps.brady2d_sweep import CLASSICAL_E4_ALPHA

        bounds = DEFAULT_BOUNDS[("E4", "classical")]
        # Single seed matches ``TestStagedClassicalAlpha`` — seed=0 is
        # empirically known to land in the Brady-Livescu basin at this
        # budget (~2 min wall; 43.9b observed α ≈ [-0.81, 0.09]).  One
        # feasible seed suffices to populate the basin list; a wider
        # multi-seed diversity study is a production-run concern and out
        # of scope for the regression gate here.
        survey = survey_mod.run_survey(
            n_seeds=1,
            bounds=bounds,
            scheme="E4",
            kernel="classical",
            report_field="layer6.transient_growth_bound",
            inner_gate=3,
            inner_max_layer=3,
            validator_max_layer=6,
            top_k=5,
            method="Nelder-Mead",
            n_restarts=20,
            base_seed=0,
            max_evals=60,
            cluster_decimals=2,
        )

        assert survey["n_feasible_seeds"] >= 1, (
            "expected at least one feasible seed; "
            f"got seed_results={survey['seed_results']}"
        )
        assert survey["basins"], "survey returned zero basins"

        published = np.asarray(CLASSICAL_E4_ALPHA, dtype=float)
        tol = 0.5
        distances = [
            float(np.max(np.abs(np.asarray(b["alpha"], dtype=float) - published)))
            for b in survey["basins"]
        ]
        assert any(d <= tol for d in distances), (
            f"no basin within L∞ distance {tol} of Brady-Livescu α "
            f"{published.tolist()}; basin distances={distances}, "
            f"basins={survey['basins']}"
        )


class TestOptimizeCppValidation:
    """Covers ``sweeps.optimize._run_cpp_validation`` banner criterion.

    Created under plan 43.10a-r1 so the PASS/soft-failure/FAIL split is
    tested alongside the code change.  43.10b will extend this class with
    a live-pipeline case against the shoccs binary when the bridge
    exercises the full E4-classical winner.
    """

    @staticmethod
    def _fake_binary(tmp_path, monkeypatch):
        """Make :data:`sweeps.optimize.SHOCCS_BINARY` appear to exist."""
        from sweeps import optimize as opt_mod

        fake = tmp_path / "shoccs"
        fake.write_text("")
        monkeypatch.setattr(opt_mod, "SHOCCS_BINARY", fake)

    @staticmethod
    def _fake_report(stable: bool, final_linf: float, wall_time_s: float = 0.1):
        return StabilityReport(
            layer8={
                "stable": bool(stable),
                "final_linf": float(final_linf),
                "wall_time_s": float(wall_time_s),
            },
        )

    def test_soft_failure_when_stable_but_linf_over_tol(
        self, tmp_path, monkeypatch, capsys
    ):
        from stencil_gen import brady2d_stability as bs
        from sweeps import optimize as opt_mod

        self._fake_binary(tmp_path, monkeypatch)
        tol = bs.L8_FINAL_LINF_TOL
        monkeypatch.setattr(
            opt_mod,
            "brady2d_stability_score",
            lambda *a, **k: self._fake_report(True, tol + 1.0),
        )

        result = opt_mod._run_cpp_validation(
            scheme="E4",
            kernel="tension",
            best_params={"sigma": 1.0},
            best_objective=-1e-4,
        )

        assert result is not None
        assert result["stable"] is True
        assert result["final_linf"] == pytest.approx(tol + 1.0)
        out = capsys.readouterr().out
        assert "soft-failure" in out
        assert f"L8_FINAL_LINF_TOL={tol}" in out
        assert "L8 PASS" not in out

    def test_pass_when_stable_and_linf_within_tol(
        self, tmp_path, monkeypatch, capsys
    ):
        from stencil_gen import brady2d_stability as bs
        from sweeps import optimize as opt_mod

        self._fake_binary(tmp_path, monkeypatch)
        tol = bs.L8_FINAL_LINF_TOL
        monkeypatch.setattr(
            opt_mod,
            "brady2d_stability_score",
            lambda *a, **k: self._fake_report(True, tol * 0.5),
        )

        result = opt_mod._run_cpp_validation(
            scheme="E4",
            kernel="tension",
            best_params={"sigma": 1.0},
            best_objective=-1e-4,
        )

        assert result["stable"] is True
        out = capsys.readouterr().out
        assert "L8 PASS" in out
        assert "soft-failure" not in out

    def test_hard_fail_when_unstable(self, tmp_path, monkeypatch, capsys):
        from sweeps import optimize as opt_mod

        self._fake_binary(tmp_path, monkeypatch)
        monkeypatch.setattr(
            opt_mod,
            "brady2d_stability_score",
            lambda *a, **k: self._fake_report(False, float("nan")),
        )

        result = opt_mod._run_cpp_validation(
            scheme="E4",
            kernel="tension",
            best_params={"sigma": 1.0},
            best_objective=-1e-4,
        )

        assert result["stable"] is False
        out = capsys.readouterr().out
        assert "L8 FAIL" in out
        assert "soft-failure" not in out

    @pytest.mark.slow
    def test_validate_classical_e4_published_alpha(self, capsys):
        """Live shoccs L8 run at the Brady-Livescu published E4 classical α.

        Plan 43.10b: end-to-end exercise of ``_run_cpp_validation`` with no
        ``brady2d_stability_score`` monkeypatch — the full L8 bridge actually
        runs the shoccs binary and reports a real ``final_linf``.  Skips when
        the binary has not been built.  Published α is sourced from the
        canonical ``sweeps.brady2d_sweep.CLASSICAL_E4_ALPHA`` (see 43.9d for
        why ``stencil_gen.alpha_extraction`` is not the source — that module
        only holds E2 production α's).
        """
        from stencil_gen import brady2d_stability as bs
        from sweeps import optimize as opt_mod
        from sweeps.brady2d_sweep import CLASSICAL_E4_ALPHA

        if not opt_mod.SHOCCS_BINARY.exists():
            pytest.skip("shoccs binary not built")

        result = opt_mod._run_cpp_validation(
            scheme="E4",
            kernel="classical",
            best_params={"alpha": list(CLASSICAL_E4_ALPHA)},
            best_objective=-1.0,
            N=21,
            t_final=5.0,
        )

        assert result is not None
        assert result["stable"] is True
        assert result["final_linf"] <= bs.L8_FINAL_LINF_TOL
        out = capsys.readouterr().out
        assert "L8 PASS" in out
        assert "soft-failure" not in out
        assert "L8 FAIL" not in out


class TestOptimizerBL42:
    @pytest.mark.slow
    def test_objective_bl42_classical_finite(self):
        f = make_objective(
            "E4", "classical", "layer_bl42.max_spectral_abscissa",
            gate_layer=3,
        )
        val = f(np.array([-0.7733323791884821, 0.1623961700641681]))
        assert np.isfinite(val)
        assert val >= 0.0

    @pytest.mark.slow
    def test_objective_bl42_gaussian_unstable_infinite(self):
        f = make_objective(
            "E4", "gaussian", "layer_bl42.max_spectral_abscissa",
            gate_layer=3, max_layer=3,
        )
        assert f(np.array([0.1])) == float("inf")

    @pytest.mark.slow
    def test_cli_optimize_bl42_tension(self):
        import os
        import subprocess
        import sys
        from pathlib import Path

        stencil_gen_dir = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env["SYMPY_CACHE_SIZE"] = env.get("SYMPY_CACHE_SIZE", "50000")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "sweeps.optimize",
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer_bl42.max_spectral_abscissa",
                "--gate-layer", "2",
                "--bounds", "0.5", "20",
                "--method", "Nelder-Mead",
                "--n-restarts", "4",
                "--max-evals", "40",
                "--seed", "0",
            ],
            cwd=str(stencil_gen_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, (
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        assert "best_objective" in proc.stdout
        assert "inf" not in proc.stdout, "all evaluations returned +inf (vacuous run)"

    @pytest.mark.slow
    def test_cli_optimize_bl42_tension_auto_gate_layer(self):
        """Same as ``test_cli_optimize_bl42_tension`` but with ``--gate-layer``
        omitted — asserts the CLI's auto-infer path (plan 45.0d) produces a
        finite result for the L3r objective without the user knowing to pass
        ``--gate-layer 2`` manually. Under the pre-45.0d hardcoded
        ``default=3`` this would have self-gated and every evaluation would
        have returned ``+inf``.
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        stencil_gen_dir = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env["SYMPY_CACHE_SIZE"] = env.get("SYMPY_CACHE_SIZE", "50000")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "sweeps.optimize",
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer_bl42.max_spectral_abscissa",
                "--bounds", "0.5", "20",
                "--method", "Nelder-Mead",
                "--n-restarts", "4",
                "--max-evals", "40",
                "--seed", "0",
            ],
            cwd=str(stencil_gen_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, (
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
        assert "best_objective" in proc.stdout
        assert "inf" not in proc.stdout, "all evaluations returned +inf (vacuous run)"


class TestSweepsMainDispatchGateLayerInfer:
    """Plan 46.0b: assert ``python -m sweeps optimize`` (umbrella dispatch)
    forwards ``--gate-layer`` only when the user passes it explicitly.

    The bug fixed in 46.0a is that ``sweeps/__main__.py`` registered
    ``--gate-layer`` with ``default=3`` and forwarded it unconditionally,
    nullifying the standalone-CLI auto-infer added in 45.0d for users hitting
    the documented entry point. These tests pin the dispatch-level contract:
    omitting ``--gate-layer`` must result in the standalone CLI receiving no
    ``--gate-layer`` flag at all (so its own ``default=None`` and auto-infer
    kick in); passing it explicitly must round-trip the value untouched.
    """

    def _capture_forwarded(self, monkeypatch, argv: list[str]) -> list[str]:
        """Run ``sweeps.__main__.main()`` with ``argv`` and return the args
        the dispatcher forwarded into ``sweeps.optimize.main``."""
        import sys

        from sweeps import __main__ as main_mod
        from sweeps import optimize as opt_mod

        captured: list[list[str]] = []

        def fake_optimize_main(forwarded: list[str]) -> int:
            captured.append(list(forwarded))
            return 0

        monkeypatch.setattr(opt_mod, "main", fake_optimize_main)
        monkeypatch.setattr(sys, "argv", argv)
        rc = main_mod.main()
        assert rc == 0, f"dispatcher returned {rc}"
        assert len(captured) == 1, f"expected one forward, got {len(captured)}"
        return captured[0]

    def test_dispatch_omitting_gate_layer_uses_auto_infer(self, monkeypatch):
        """When ``--gate-layer`` is omitted, the dispatcher must NOT inject it
        with a hardcoded value — the standalone CLI's ``default=None`` then
        auto-infers ``max(max_layer - 1, 0)`` (= 2 for an L3 objective).
        """
        forwarded = self._capture_forwarded(
            monkeypatch,
            [
                "sweeps", "optimize",
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer_bl42.max_spectral_abscissa",
                "--bounds", "0.5", "20",
                "--method", "Nelder-Mead",
                "--max-evals", "5",
            ],
        )
        assert "--gate-layer" not in forwarded, (
            "dispatcher must not forward --gate-layer when the user omitted "
            "it; otherwise the standalone CLI's None default and auto-infer "
            "(plan 45.0d) are bypassed."
        )

    def test_dispatch_explicit_gate_layer_preserved(self, monkeypatch):
        """When ``--gate-layer N`` is passed, the dispatcher forwards it
        verbatim.
        """
        forwarded = self._capture_forwarded(
            monkeypatch,
            [
                "sweeps", "optimize",
                "--scheme", "E4",
                "--kernel", "tension",
                "--objective", "layer_bl42.max_spectral_abscissa",
                "--bounds", "0.5", "20",
                "--method", "Nelder-Mead",
                "--max-evals", "5",
                "--gate-layer", "4",
            ],
        )
        assert "--gate-layer" in forwarded
        idx = forwarded.index("--gate-layer")
        assert forwarded[idx + 1] == "4", (
            f"expected --gate-layer 4 to round-trip; got {forwarded[idx + 1]!r}"
        )
