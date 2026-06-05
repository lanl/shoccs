"""Tests for stencil_gen.benchmarks modules."""

import math

import numpy as np
import pytest

from stencil_gen.benchmarks.brady_livescu_2d import (
    L_DOMAIN,
    PSI_OFFSET,
    c_x,
    c_y,
    exact_solution,
    inflow_bc_x,
    inflow_bc_y,
    initial_condition,
    make_coefficient_field,
    psi,
)
from stencil_gen.benchmarks.brady_livescu_4_2 import (
    L_DOMAIN as L_DOMAIN_42,
    continuous_eigenvalues,
    exact_solution as exact_solution_42,
    initial_u,
    initial_v,
)
from stencil_gen.benchmarks.brady2d_calibration import (
    FAMILIES,
    _E4_CLASSICAL_ALPHA,
    _report_to_dict,
    format_calibration_table,
    run_calibration,
)


class TestBradyLivescu2D:
    """Tests for the Brady & Livescu 2019 §4.3 reference problem."""

    def test_constants(self):
        assert L_DOMAIN == pytest.approx(math.sqrt(2.0))
        assert PSI_OFFSET == pytest.approx(0.25)

    def test_exact_solution_initial_time(self):
        """exact_solution(x, y, 0) == initial_condition(x, y) at sample points."""
        rng = np.random.default_rng(42)
        xs = rng.uniform(0.0, L_DOMAIN, 5)
        ys = rng.uniform(0.0, L_DOMAIN, 5)
        for x_val, y_val in zip(xs, ys):
            assert exact_solution(x_val, y_val, 0.0) == pytest.approx(
                initial_condition(x_val, y_val)
            )

    def test_exact_solution_satisfies_pde(self):
        """Central-difference approximation of u_t + c_x*u_x + c_y*u_y ~ 0."""
        rng = np.random.default_rng(123)
        h = 1e-6
        for _ in range(5):
            x_val = rng.uniform(0.1, L_DOMAIN - 0.1)
            y_val = rng.uniform(0.1, L_DOMAIN - 0.1)
            t_val = rng.uniform(0.1, 10.0)

            # u_t via central difference
            u_t = (
                exact_solution(x_val, y_val, t_val + h)
                - exact_solution(x_val, y_val, t_val - h)
            ) / (2.0 * h)

            # u_x via central difference
            u_x = (
                exact_solution(x_val + h, y_val, t_val)
                - exact_solution(x_val - h, y_val, t_val)
            ) / (2.0 * h)

            # u_y via central difference
            u_y = (
                exact_solution(x_val, y_val + h, t_val)
                - exact_solution(x_val, y_val - h, t_val)
            ) / (2.0 * h)

            residual = u_t + c_x(x_val, y_val) * u_x + c_y(x_val, y_val) * u_y
            assert abs(residual) < 1e-6, (
                f"PDE residual {residual} at ({x_val}, {y_val}, {t_val})"
            )

    def test_coefficient_field_shape(self):
        """make_coefficient_field(31) returns (31, 31) arrays with unit speed."""
        x, y, cx, cy = make_coefficient_field(31)
        assert x.shape == (31, 31)
        assert y.shape == (31, 31)
        assert cx.shape == (31, 31)
        assert cy.shape == (31, 31)
        speed = np.sqrt(cx**2 + cy**2)
        np.testing.assert_allclose(speed, 1.0, atol=1e-14)

    def test_inflow_bc_matches_exact_at_edges(self):
        """Inflow BCs match exact solution at the domain edges."""
        rng = np.random.default_rng(77)
        ys = rng.uniform(0.0, L_DOMAIN, 5)
        ts = rng.uniform(0.0, 100.0, 5)
        for y_val, t_val in zip(ys, ts):
            assert inflow_bc_x(y_val, t_val) == pytest.approx(
                exact_solution(0.0, y_val, t_val)
            )

        xs = rng.uniform(0.0, L_DOMAIN, 5)
        ts = rng.uniform(0.0, 100.0, 5)
        for x_val, t_val in zip(xs, ts):
            assert inflow_bc_y(x_val, t_val) == pytest.approx(
                exact_solution(x_val, 0.0, t_val)
            )


class TestCalibrationDataclass:
    """Tests for brady2d_calibration module data structures and enumeration."""

    def test_families_length(self):
        """FAMILIES has the expected 9 entries (no E2 classical per 41.5c)."""
        assert len(FAMILIES) == 9

    def test_families_entries_are_4_tuples(self):
        """Each family entry has (scheme, kernel, params, display_label)."""
        for entry in FAMILIES:
            assert len(entry) == 4
            scheme, kernel, params, label = entry
            assert scheme in ("E2", "E4")
            assert kernel in ("classical", "tension", "gaussian", "multiquadric")
            assert isinstance(params, dict)
            assert isinstance(label, str)

    def test_e4_classical_alpha_values(self):
        """E4 classical alpha matches known production values."""
        assert _E4_CLASSICAL_ALPHA == pytest.approx(
            [-0.7733323791884821, 0.1623961700641681]
        )

    def test_phs_k2_entries_use_sigma_zero(self):
        """PHS k=2 entries use kernel='tension' with sigma=0.0."""
        phs_entries = [(s, k, p, l) for s, k, p, l in FAMILIES if "phs_k2" in l]
        assert len(phs_entries) == 2  # E2 and E4
        for _, kernel, params, _ in phs_entries:
            assert kernel == "tension"
            assert params["sigma"] == 0.0

    def test_display_labels_unique(self):
        """All display labels are unique."""
        labels = [label for _, _, _, label in FAMILIES]
        assert len(labels) == len(set(labels))

    def test_report_to_dict_handles_empty_report(self):
        """_report_to_dict works on a minimal StabilityReport."""
        from stencil_gen.brady2d_stability import StabilityReport
        report = StabilityReport()
        d = _report_to_dict(report)
        assert d["overall_verdict"] == "unknown"
        assert d["failed_layer"] is None
        assert "layer1" not in d

    def test_format_calibration_table_produces_markdown(self):
        """format_calibration_table returns a string with a markdown header."""
        sample = {"test_family": {
            "overall_verdict": "pass",
            "failed_layer": None,
            "failed_reason": "",
            "compute_time": 1.5,
            "layer1": {"boundary_gv_err": 0.01},
        }}
        table = format_calibration_table(sample)
        assert "| Family" in table
        assert "test_family" in table
        assert "pass" in table

    def test_run_calibration_at_layer_1(self):
        """run_calibration at max_layer=1 returns results for all families."""
        results = run_calibration(max_layer=1)
        assert len(results) == len(FAMILIES)
        for label, r in results.items():
            assert r["overall_verdict"] in ("pass", "fail", "error"), (
                f"{label}: unexpected verdict {r['overall_verdict']}"
            )
            assert "layer1" in r or r["overall_verdict"] == "error", (
                f"{label}: missing layer1 data"
            )


class TestBradyLivescu42:
    """Tests for the Brady & Livescu 2019 §4.2 reflecting-hyperbolic benchmark."""

    def test_initial_condition_matches_paper(self):
        """u(0) = 0, u(1) = -1.5*pi*sin(1.5*pi) ~ -1.5*pi, v(x) = 0."""
        assert initial_u(0.0) == pytest.approx(0.0, abs=1e-15)
        assert initial_u(1.0) == pytest.approx(
            -1.5 * np.pi * np.sin(1.5 * np.pi), rel=1e-14
        )
        xs = np.linspace(0, 1, 11)
        np.testing.assert_allclose(initial_v(xs), 0.0, atol=1e-15)

    def test_exact_solution_matches_initial_at_t_zero(self):
        """exact_solution(x, 0.0) returns (initial_u(x), initial_v(x))."""
        rng = np.random.default_rng(42)
        xs = rng.uniform(0.0, L_DOMAIN_42, 10)
        u, v = exact_solution_42(xs, 0.0)
        np.testing.assert_allclose(u, initial_u(xs), atol=1e-14)
        np.testing.assert_allclose(v, initial_v(xs), atol=1e-14)

    def test_exact_solution_satisfies_pde(self):
        """Central-difference check: |u_t - v_x| < 1e-6 and |v_t - u_x| < 1e-6."""
        rng = np.random.default_rng(123)
        h = 1e-6
        for _ in range(5):
            x_val = rng.uniform(0.1, 0.9)
            t_val = rng.uniform(0.1, 10.0)

            u_p, v_p = exact_solution_42(x_val, t_val + h)
            u_m, v_m = exact_solution_42(x_val, t_val - h)
            u_t = (u_p - u_m) / (2.0 * h)
            v_t = (v_p - v_m) / (2.0 * h)

            u_xp, v_xp = exact_solution_42(x_val + h, t_val)
            u_xm, v_xm = exact_solution_42(x_val - h, t_val)
            v_x = (v_xp - v_xm) / (2.0 * h)
            u_x = (u_xp - u_xm) / (2.0 * h)

            assert abs(u_t - v_x) < 1e-6, f"u_t - v_x = {u_t - v_x} at (x={x_val}, t={t_val})"
            assert abs(v_t - u_x) < 1e-6, f"v_t - u_x = {v_t - u_x} at (x={x_val}, t={t_val})"

    def test_continuous_eigenvalues_purely_imaginary(self):
        """Eigenvalues have zero real part, imag parts at +/-(2k-1)*pi/2."""
        eigs = continuous_eigenvalues(5)
        assert len(eigs) == 10
        np.testing.assert_allclose(eigs.real, 0.0, atol=1e-15)
        expected_pos = np.array([(2 * k - 1) * np.pi / 2.0 for k in range(1, 6)])
        expected_all = np.sort(np.concatenate([expected_pos, -expected_pos]))
        np.testing.assert_allclose(np.sort(eigs.imag), expected_all, rtol=1e-14)
