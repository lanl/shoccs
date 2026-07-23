"""Tests for scripts/stencil_gen/stencil_gen/cpp_bridge.py.

Plan 42.2a scope: make_brady2d_lua marker substitution and scheme-table
emission. Plan 42.2b scope: run_cpp_brady2d subprocess driver, verified with
a fake shoccs binary so the tests stay fast and hermetic. Plan 48 item 4.4:
embedded-sphere cut template, growth-rate metric, cut-capable L8 dispatch,
and the CFL knob.
"""

from __future__ import annotations

import math
import re
import stat
from pathlib import Path

import numpy as np
import pytest

from stencil_gen.brady2d_stability import (
    _L8_CUT_SCHEME_TYPE,
    _L8_SCHEME_TYPE,
    L8_RATE_TOL,
    layer8_cpp_simulation,
)
from stencil_gen.cpp_bridge import (
    BRADY_LIVESCU_CUT_TEMPLATE,
    BRADY_LIVESCU_TEMPLATE,
    CUT_SPHERE_CENTER,
    CUT_SPHERE_RADIUS,
    LUA_TEMPLATE_DIR,
    REPO_ROOT,
    SHOCCS_BINARY,
    BridgeResult,
    compute_growth_rate,
    make_brady2d_lua,
    run_cpp_brady2d,
)


class TestBridgePaths:
    def test_repo_root_contains_expected_markers(self):
        assert (REPO_ROOT / "CMakeLists.txt").exists()
        assert (REPO_ROOT / "lua-configs").is_dir()

    def test_lua_template_dir_matches_repo_root(self):
        assert LUA_TEMPLATE_DIR == REPO_ROOT / "lua-configs"

    def test_brady_livescu_template_exists(self):
        assert BRADY_LIVESCU_TEMPLATE.exists()
        text = BRADY_LIVESCU_TEMPLATE.read_text()
        for marker in ("--{{N}}--", "--{{T_FINAL}}--", "--{{SCHEME_TABLE}}--"):
            assert marker in text, f"template missing marker {marker!r}"

    def test_shoccs_binary_path_matches_plan(self):
        assert SHOCCS_BINARY == REPO_ROOT / "build" / "src" / "app" / "shoccs"


class TestMakeBrady2DLua:
    CLASSICAL_ALPHA = [-0.7733323791884821, 0.1623961700641681]

    def _render_classical(self, **overrides) -> str:
        kwargs = dict(
            scheme_type="E4u",
            params={"alpha": self.CLASSICAL_ALPHA},
            N=31,
            t_final=10.0,
        )
        kwargs.update(overrides)
        return make_brady2d_lua(**kwargs)

    def test_all_markers_replaced(self):
        rendered = self._render_classical()
        for marker in ("--{{N}}--", "--{{T_FINAL}}--", "--{{SCHEME_TABLE}}--"):
            assert marker not in rendered, f"marker {marker!r} survived substitution"

    def test_n_substituted_as_integer(self):
        rendered = self._render_classical(N=41)
        # index_extents = {41, 41}
        assert re.search(r"index_extents\s*=\s*\{41,\s*41\}", rendered)

    def test_t_final_substituted(self):
        rendered = self._render_classical(t_final=5.5)
        assert re.search(r"max_time\s*=\s*5\.5", rendered)

    def test_scheme_table_contains_type_and_alpha(self):
        rendered = self._render_classical()
        # Expect scheme = { order = 1, type = "E4u", alpha = {-0.77..., 0.16...} }
        assert re.search(r'type\s*=\s*"E4u"', rendered)
        for val in self.CLASSICAL_ALPHA:
            assert repr(val) in rendered, f"alpha coefficient {val!r} missing"
        assert re.search(r"alpha\s*=\s*\{", rendered)

    def test_scheme_table_with_sigma(self):
        rendered = make_brady2d_lua(
            scheme_type="tension_E4u",
            params={"sigma": 3.0},
            N=31,
            t_final=10.0,
        )
        assert re.search(r'type\s*=\s*"tension_E4u"', rendered)
        assert re.search(r"sigma\s*=\s*3\.0", rendered)
        assert "alpha" not in rendered.split("scheme")[1].split("system")[0]

    def test_scheme_table_with_epsilon(self):
        rendered = make_brady2d_lua(
            scheme_type="gaussian_E4u",
            params={"epsilon": 0.9},
            N=31,
            t_final=10.0,
        )
        assert re.search(r'type\s*=\s*"gaussian_E4u"', rendered)
        assert re.search(r"epsilon\s*=\s*0\.9", rendered)

    def test_rendered_output_is_lua_like(self):
        """Basic sanity: every Lua brace must be balanced after substitution."""
        rendered = self._render_classical()
        assert rendered.count("{") == rendered.count("}")
        assert "simulation" in rendered
        assert "mesh" in rendered
        assert "system" in rendered

    def test_int_like_float_for_t_final(self):
        # 10.0 should appear with a decimal point so Lua parses it as number
        rendered = self._render_classical(t_final=10.0)
        assert re.search(r"max_time\s*=\s*10\.0", rendered)


class TestMakeBrady2DLuaSpline:
    """Plan 42.7b — spline-family scheme-table emission.

    The three runtime-parameterized families registered in 42.5e/42.6c/42.6g
    each take a single scalar parameter (`sigma` for tension, `epsilon` for
    gaussian and multiquadric) and must round-trip cleanly through the Lua
    template.
    """

    @pytest.mark.parametrize(
        "scheme_type,params,expected_kv",
        [
            ("tension_E4u", {"sigma": 3.0}, ("sigma", 3.0)),
            ("tension_E4u", {"sigma": 7.5}, ("sigma", 7.5)),
            ("gaussian_E4u", {"epsilon": 0.9}, ("epsilon", 0.9)),
            ("gaussian_E4u", {"epsilon": 1.25}, ("epsilon", 1.25)),
            ("multiquadric_E4u", {"epsilon": 1.0}, ("epsilon", 1.0)),
            ("multiquadric_E4u", {"epsilon": 0.4}, ("epsilon", 0.4)),
        ],
    )
    def test_emits_spline_scheme_table(self, scheme_type, params, expected_kv):
        rendered = make_brady2d_lua(
            scheme_type=scheme_type, params=params, N=31, t_final=10.0
        )
        key, value = expected_kv
        # type must match verbatim
        assert re.search(rf'type\s*=\s*"{re.escape(scheme_type)}"', rendered)
        # parameter must be present with its repr (preserves full precision)
        assert re.search(rf"{key}\s*=\s*{re.escape(repr(float(value)))}", rendered)
        # the other scalar is never emitted
        other = "epsilon" if key == "sigma" else "sigma"
        scheme_slice = rendered.split("scheme")[1].split("system")[0]
        assert f"{other} =" not in scheme_slice
        assert "alpha" not in scheme_slice
        # all markers are replaced
        for marker in ("--{{N}}--", "--{{T_FINAL}}--", "--{{SCHEME_TABLE}}--"):
            assert marker not in rendered
        # order = 1 is present in the scheme table
        assert "order = 1" in scheme_slice

    def test_spline_rendered_output_has_balanced_braces(self):
        for scheme_type, params in [
            ("tension_E4u", {"sigma": 3.0}),
            ("gaussian_E4u", {"epsilon": 0.9}),
            ("multiquadric_E4u", {"epsilon": 1.0}),
        ]:
            rendered = make_brady2d_lua(
                scheme_type=scheme_type, params=params, N=31, t_final=10.0
            )
            assert rendered.count("{") == rendered.count("}"), scheme_type

    def test_spline_param_precision_preserved(self):
        """Sigma/epsilon are emitted via repr() so round-tripping is exact."""
        sigma = 0.1 + 0.2  # classic float that shows extra digits in repr
        rendered = make_brady2d_lua(
            scheme_type="tension_E4u",
            params={"sigma": sigma},
            N=31,
            t_final=10.0,
        )
        assert repr(sigma) in rendered


def _make_fake_shoccs(
    tmp_path: Path,
    *,
    csv_body: str | None = None,
    exit_code: int = 0,
    sleep_s: float = 0.0,
) -> Path:
    """Write a POSIX shell script that mimics the shoccs binary contract.

    The script writes the requested CSV body (stored alongside as a sibling
    file and copied into cwd) to `logs/system.csv` in its cwd, optionally
    sleeps, then exits with `exit_code`. Keeping the CSV in a separate file
    avoids all shell-quoting hazards.
    """
    csv_src = tmp_path / "fake_shoccs.csv"
    csv_src.write_text(csv_body if csv_body is not None else "")
    script = tmp_path / "fake_shoccs.sh"
    lines = [
        "#!/bin/sh",
        "mkdir -p logs",
        f"cp {csv_src} logs/system.csv",
    ]
    if sleep_s > 0.0:
        lines.append(f"sleep {sleep_s}")
    lines.append(f"exit {exit_code}")
    script.write_text("\n".join(lines) + "\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return script


_HEADER = "Timestamp,Time,Step,Linf,Min,Max,Domain_Linf,Domain_ic,Rx_Linf,Rx_ic,Ry_Linf,Ry_ic,Rz_Linf,Rz_ic,Wall_ms"


def _good_csv(final_linf: float, n_rows: int = 5) -> str:
    rows = [_HEADER]
    ts = "2026-04-13 19:51:40.000000"
    for step in range(n_rows):
        t = 0.1 * step
        linf = final_linf if step == n_rows - 1 else 0.001 * (step + 1)
        rows.append(f"{ts},{t},{step},{linf},-1.0,1.0,{linf},0,0,0,0,0,0,0,0.1")
    return "\n".join(rows)


class TestRunCppBrady2D:
    def test_success_parses_final_linf(self, tmp_path: Path):
        fake = _make_fake_shoccs(tmp_path, csv_body=_good_csv(0.01234))
        result = run_cpp_brady2d(
            scheme_type="E4u",
            params={"alpha": [-0.77, 0.16]},
            N=21,
            t_final=1.0,
            binary=fake,
        )
        assert result.exit_code == 0
        assert result.stable is True
        assert result.final_linf == pytest.approx(0.01234)
        assert result.linf_trace.shape == (5,)
        assert result.t_trace.shape == (5,)
        assert result.wall_time_s > 0.0

    def test_nonzero_exit_returns_nan_and_captures_stderr(self, tmp_path: Path):
        # Script that writes nothing useful and exits 2; also emits stderr.
        script = tmp_path / "failing_shoccs.sh"
        script.write_text("#!/bin/sh\necho boom >&2\nexit 2\n")
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        result = run_cpp_brady2d(
            scheme_type="E4u",
            params={"alpha": [-0.77, 0.16]},
            N=21,
            t_final=1.0,
            binary=script,
        )
        assert result.exit_code == 2
        assert result.stable is False
        assert np.isnan(result.final_linf)
        assert "boom" in result.stderr

    def test_unstable_when_final_linf_large(self, tmp_path: Path):
        fake = _make_fake_shoccs(tmp_path, csv_body=_good_csv(123.4))
        result = run_cpp_brady2d(
            scheme_type="E4u",
            params={"alpha": [-0.77, 0.16]},
            N=21,
            t_final=1.0,
            binary=fake,
        )
        assert result.exit_code == 0
        assert result.stable is False
        assert result.final_linf == pytest.approx(123.4)

    def test_missing_csv_returns_failure(self, tmp_path: Path):
        # Script that exits 0 but never writes logs/system.csv.
        script = tmp_path / "silent_shoccs.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        result = run_cpp_brady2d(
            scheme_type="E4u",
            params={"alpha": [-0.77, 0.16]},
            N=21,
            t_final=1.0,
            binary=script,
        )
        assert result.stable is False
        assert np.isnan(result.final_linf)
        assert "system.csv" in result.stderr

    def test_empty_csv_returns_failure(self, tmp_path: Path):
        fake = _make_fake_shoccs(tmp_path, csv_body=_HEADER)
        result = run_cpp_brady2d(
            scheme_type="E4u",
            params={"alpha": [-0.77, 0.16]},
            N=21,
            t_final=1.0,
            binary=fake,
        )
        assert result.stable is False
        assert np.isnan(result.final_linf)
        assert "no data rows" in result.stderr

    def test_timeout_is_graceful(self, tmp_path: Path):
        fake = _make_fake_shoccs(tmp_path, csv_body=_good_csv(0.1), sleep_s=2.0)
        result = run_cpp_brady2d(
            scheme_type="E4u",
            params={"alpha": [-0.77, 0.16]},
            N=21,
            t_final=1.0,
            timeout=0.2,
            binary=fake,
        )
        assert result.stable is False
        assert result.exit_code == -1
        assert "timeout" in result.stderr.lower()

    def test_run_is_isolated_to_tempdir(self, tmp_path: Path):
        """run_cpp_brady2d must not clobber REPO_ROOT/logs/system.csv."""
        fake = _make_fake_shoccs(tmp_path, csv_body=_good_csv(0.5))
        repo_logs = REPO_ROOT / "logs" / "system.csv"
        before = repo_logs.read_bytes() if repo_logs.exists() else None
        result = run_cpp_brady2d(
            scheme_type="E4u",
            params={"alpha": [-0.77, 0.16]},
            N=21,
            t_final=1.0,
            binary=fake,
        )
        assert result.stable is True
        after = repo_logs.read_bytes() if repo_logs.exists() else None
        assert before == after, "run_cpp_brady2d must not touch REPO_ROOT/logs/"


class TestCppBridgeSmoke:
    """End-to-end smoke test against the real shoccs binary.

    Marked @pytest.mark.slow — deselected unless --run-slow is passed.
    Skipped entirely if the binary hasn't been built.
    """

    @pytest.mark.slow
    def test_classical_e4u_short_run(self):
        if not SHOCCS_BINARY.exists():
            pytest.skip("shoccs binary not built")
        result = run_cpp_brady2d(
            scheme_type="E4u",
            params={"alpha": [-0.7733323791884821, 0.1623961700641681]},
            N=21,
            t_final=1.0,
        )
        assert result.exit_code == 0, f"shoccs failed: {result.stderr}"
        assert result.stable is True
        assert 0.0 < result.final_linf < 1.0, (
            f"final_linf out of expected band: {result.final_linf}"
        )
        assert result.linf_trace.size > 0
        assert result.t_trace.size == result.linf_trace.size


class TestBridgeResultDefaults:
    def test_default_construction(self):
        r = BridgeResult(final_linf=0.0)
        assert r.stable is False
        assert r.wall_time_s == 0.0
        assert r.exit_code == 0
        assert r.stderr == ""
        assert r.linf_trace.shape == (0,)
        assert r.t_trace.shape == (0,)
        assert r.growth_rate is None

    def test_final_linf_defaults_to_nan(self):
        r = BridgeResult()
        assert np.isnan(r.final_linf)

    def test_explicit_fields(self):
        import numpy as np

        r = BridgeResult(
            final_linf=0.123,
            linf_trace=np.array([0.0, 0.1, 0.123]),
            t_trace=np.array([0.0, 0.5, 1.0]),
            stable=True,
            wall_time_s=4.2,
            exit_code=0,
            stderr="",
        )
        assert r.stable is True
        assert r.final_linf == pytest.approx(0.123)
        assert r.linf_trace.shape == (3,)


# ---------------------------------------------------------------------------
# Plan 48 item 4.4 — embedded-sphere cut template, growth rate, cut dispatch
# ---------------------------------------------------------------------------


_DOMAIN_L = math.sqrt(2.0)


def _realized_psi_set(cx: float, cy: float, r: float, N: int) -> list[float]:
    """Replicate the shoccs cut-cell psi computation for a sphere.

    Mirrors src/mesh/object_geometry.cpp:70-95: for each grid line along each
    axis, ray-cast from the domain min; an entering hit (ray_outside=1) at
    position p gives psi = frac(p/h) (distance from the last fluid node to
    the wall over h), an exiting hit (ray_outside=0) gives psi = 1 - frac(p/h)
    (wall to first fluid node), with p/h snapped to the nearest integer when
    within 1e-12 and psi clamped to [1e-12, 1 - 1e-12].

    Cross-validated against the real binary: at N=31 with the template
    sphere, logs/geometry.csv reports 44 cut points with min/max psi matching
    this replication to ~1e-12 (0.0015736866200733 / 0.9992661602721697).
    """
    h = _DOMAIN_L / (N - 1)
    snap = 1e-12
    psis: list[float] = []
    for line_center, ray_center in ((cy, cx), (cx, cy)):
        for j in range(N):
            b = j * h
            d2 = r * r - (b - line_center) ** 2
            if d2 <= 0.0:
                continue
            half_chord = math.sqrt(d2)
            for p, entering in (
                (ray_center - half_chord, True),
                (ray_center + half_chord, False),
            ):
                if not 0.0 <= p <= _DOMAIN_L:
                    continue
                t_over_h = p / h
                i_cell = int(round(t_over_h))
                if abs(t_over_h - i_cell) > snap:
                    i_cell = int(t_over_h)
                psi = t_over_h - i_cell if entering else (i_cell + 1) - t_over_h
                psis.append(min(max(psi, snap), 1.0 - snap))
    return psis


class TestCutTemplate:
    """4.4.1 — the embedded-sphere template and its psi coverage."""

    def test_cut_template_exists_with_all_markers(self):
        assert BRADY_LIVESCU_CUT_TEMPLATE.exists()
        text = BRADY_LIVESCU_CUT_TEMPLATE.read_text()
        for marker in (
            "--{{N}}--",
            "--{{T_FINAL}}--",
            "--{{SCHEME_TABLE}}--",
            "--{{RADIUS}}--",
            "--{{CFL}}--",
        ):
            assert marker in text, f"cut template missing marker {marker!r}"

    def test_cut_template_sphere_block(self):
        text = BRADY_LIVESCU_CUT_TEMPLATE.read_text()
        assert re.search(r'type\s*=\s*"sphere"', text)
        assert re.search(r'boundary_condition\s*=\s*"dirichlet"', text)
        cx, cy = CUT_SPHERE_CENTER
        # the template's hardcoded center must match the module constant the
        # psi-span assertion below is computed from
        assert re.search(
            rf"center\s*=\s*\{{{re.escape(repr(cx))},\s*{re.escape(repr(cy))},\s*0\}}",
            text,
        ), "sphere center in cut template does not match CUT_SPHERE_CENTER"

    def test_cut_template_keeps_exact_solution_and_system(self):
        """system.center/radius take precedence over shapes for the exact
        solution (scalar_wave.cpp:143-147), so the cut run solves the same
        Brady-Livescu field as the uniform baseline."""
        text = BRADY_LIVESCU_CUT_TEMPLATE.read_text()
        assert re.search(r'type\s*=\s*"scalar wave"', text)
        assert re.search(r"center\s*=\s*\{-0\.25,\s*-0\.25,\s*0\}", text)
        assert re.search(r"max_error\s*=\s*10\.0", text)

    def test_uncut_template_untouched(self):
        """4.4.1: the original template stays the uncut regression baseline —
        no new markers, shapes empty, CFL hardcoded."""
        text = BRADY_LIVESCU_TEMPLATE.read_text()
        assert "--{{RADIUS}}--" not in text
        assert "--{{CFL}}--" not in text
        assert re.search(r"shapes\s*=\s*\{\}", text)
        assert re.search(r"hyperbolic\s*=\s*0\.8", text)

    @pytest.mark.parametrize("N", [31, 61])
    def test_realized_psi_spans_near_0_to_near_1(self, N):
        """4.4.1 acceptance: the realized psi set must have points < 0.05 and
        > 0.95 (here asserted at each N individually, stronger than the
        plan's across-N requirement)."""
        cx, cy = CUT_SPHERE_CENTER
        psis = _realized_psi_set(cx, cy, CUT_SPHERE_RADIUS, N)
        assert len(psis) >= 20, "sphere should generate a healthy cut set"
        assert all(0.0 < p < 1.0 for p in psis)
        assert min(psis) < 0.05, f"no near-0 psi at N={N}: min={min(psis):.4g}"
        assert max(psis) > 0.95, f"no near-1 psi at N={N}: max={max(psis):.4g}"

    def test_sphere_well_inside_domain(self):
        cx, cy = CUT_SPHERE_CENTER
        r = CUT_SPHERE_RADIUS
        margin = 0.3  # keep closure supports off the domain walls
        for c in (cx, cy):
            assert c - r > margin
            assert c + r < _DOMAIN_L - margin


class TestMakeBrady2DLuaCut:
    """4.4.1/4.4.4 — rendering the cut template; marker guard rails."""

    E2_PARAMS = {"alpha": [0.0, 0.0, 0.0, 0.0]}

    def _render_cut(self, **overrides) -> str:
        kwargs = dict(
            scheme_type="E2",
            params=self.E2_PARAMS,
            N=31,
            t_final=10.0,
            template=BRADY_LIVESCU_CUT_TEMPLATE,
            radius=CUT_SPHERE_RADIUS,
            cfl=0.8,
        )
        kwargs.update(overrides)
        return make_brady2d_lua(**kwargs)

    def test_cut_render_fills_all_markers(self):
        rendered = self._render_cut()
        assert "--{{" not in rendered
        assert re.search(r'type\s*=\s*"E2"', rendered)
        assert re.search(
            rf"radius\s*=\s*{re.escape(repr(float(CUT_SPHERE_RADIUS)))}", rendered
        )
        assert re.search(r"hyperbolic\s*=\s*0\.8", rendered)
        assert rendered.count("{") == rendered.count("}")

    def test_cut_render_custom_cfl(self):
        rendered = self._render_cut(cfl=1.0)
        assert re.search(r"hyperbolic\s*=\s*1\.0", rendered)

    def test_cfl_against_uncut_template_raises(self):
        """4.4.4: the uncut template hardcodes cfl with no marker — a silent
        str.replace no-op would run the wrong CFL, so this must be loud."""
        with pytest.raises(ValueError, match="CFL"):
            make_brady2d_lua(
                scheme_type="E4u",
                params={"alpha": [-0.77, 0.16]},
                N=31,
                t_final=10.0,
                template=BRADY_LIVESCU_TEMPLATE,
                cfl=0.8,
            )

    def test_radius_against_uncut_template_raises(self):
        with pytest.raises(ValueError, match="RADIUS"):
            make_brady2d_lua(
                scheme_type="E4u",
                params={"alpha": [-0.77, 0.16]},
                N=31,
                t_final=10.0,
                template=BRADY_LIVESCU_TEMPLATE,
                radius=0.25,
            )

    def test_cut_template_without_radius_raises(self):
        with pytest.raises(ValueError, match="unfilled markers"):
            self._render_cut(radius=None)

    def test_cut_template_without_cfl_raises(self):
        with pytest.raises(ValueError, match="unfilled markers"):
            self._render_cut(cfl=None)


class TestComputeGrowthRate:
    """4.4.2 — trailing-half least-squares slope of log(L-inf)."""

    def test_pure_exponential_recovers_rate(self):
        t = np.linspace(0.0, 10.0, 201)
        y = 1e-4 * np.exp(0.75 * t)
        assert compute_growth_rate(t, y) == pytest.approx(0.75, rel=1e-9)

    def test_decay_gives_negative_rate(self):
        t = np.linspace(0.0, 10.0, 101)
        y = np.exp(-1.5 * t)
        assert compute_growth_rate(t, y) == pytest.approx(-1.5, rel=1e-9)

    def test_only_trailing_half_is_fitted(self):
        """A decaying transient in the leading half must not contaminate the
        fitted rate: log(y) is piecewise linear with slope -3 then +0.3."""
        t = np.linspace(0.0, 10.0, 101)
        log_y = np.where(t < 5.0, -3.0 * t, -15.0 + 0.3 * (t - 5.0))
        rate = compute_growth_rate(t, np.exp(log_y))
        # trailing half = indices 50..100 = t in [5, 10], exactly the +0.3 leg
        assert rate == pytest.approx(0.3, rel=1e-9)

    def test_none_on_nonfinite_linf(self):
        t = np.linspace(0.0, 1.0, 10)
        y = np.ones(10)
        y[7] = np.inf
        assert compute_growth_rate(t, y) is None
        y[7] = np.nan
        assert compute_growth_rate(t, y) is None

    def test_none_on_nonfinite_time(self):
        t = np.linspace(0.0, 1.0, 10)
        y = np.ones(10)
        t2 = t.copy()
        t2[3] = np.nan
        assert compute_growth_rate(t2, y) is None

    def test_none_when_linf_below_floor_throughout(self):
        t = np.linspace(0.0, 1.0, 10)
        y = np.full(10, 1e-15)
        assert compute_growth_rate(t, y) is None

    def test_none_on_too_short_series(self):
        assert compute_growth_rate(np.array([]), np.array([])) is None
        assert compute_growth_rate(np.array([0.0]), np.array([1.0])) is None
        # size 2: trailing half is a single point — no slope
        assert compute_growth_rate(np.array([0.0, 1.0]), np.array([1.0, 2.0])) is None

    def test_none_on_zero_time_span(self):
        t = np.zeros(10)
        y = np.ones(10)
        assert compute_growth_rate(t, y) is None

    def test_isolated_exact_zero_is_clamped_not_fatal(self):
        t = np.linspace(0.0, 1.0, 11)
        y = np.ones(11)
        y[8] = 0.0  # isolated zero in the trailing half
        rate = compute_growth_rate(t, y)
        assert rate is not None
        assert math.isfinite(rate)

    def test_run_cpp_brady2d_populates_growth_rate(self, tmp_path: Path):
        """The bridge computes growth_rate from the parsed trace."""
        rows = [_HEADER]
        ts = "2026-07-14 08:00:00.000000"
        t = np.linspace(0.0, 2.0, 21)
        y = 1e-3 * np.exp(1.5 * t)
        for step, (ti, yi) in enumerate(zip(t, y)):
            rows.append(f"{ts},{ti},{step},{yi},-1.0,1.0,{yi},0,0,0,0,0,0,0,0.1")
        fake = _make_fake_shoccs(tmp_path, csv_body="\n".join(rows))
        result = run_cpp_brady2d(
            scheme_type="E4u",
            params={"alpha": [-0.77, 0.16]},
            N=21,
            t_final=2.0,
            binary=fake,
        )
        assert result.exit_code == 0
        assert result.growth_rate == pytest.approx(1.5, rel=1e-6)
        assert result.growth_rate == pytest.approx(
            compute_growth_rate(result.t_trace, result.linf_trace)
        )


class TestL8CutDispatch:
    """4.4.3/4.4.4 — separate cut dispatch table, template selection, CFL knob.

    All hermetic: run_cpp_brady2d is monkeypatched in the brady2d_stability
    namespace (the existing TestLayer8 pattern).
    """

    def _patch_bridge(self, monkeypatch, result: BridgeResult) -> list[dict]:
        calls: list[dict] = []

        def fake(scheme_type, params, **kwargs):
            calls.append({"scheme_type": scheme_type, "params": params, **kwargs})
            return result

        monkeypatch.setattr(
            "stencil_gen.brady2d_stability.run_cpp_brady2d", fake,
        )
        return calls

    def test_cut_table_contents_exact(self):
        assert _L8_CUT_SCHEME_TYPE == {
            ("E2", "classical"): "E2",
            ("E4", "classical"): "E4",
        }

    def test_uniform_table_not_mutated(self):
        """Regression: 4.4.3 must NOT touch _L8_SCHEME_TYPE — its
        ("E4", "classical") → "E4u" entry is the uniform-validation path."""
        assert _L8_SCHEME_TYPE == {
            ("E4", "classical"): "E4u",
            ("E4", "tension"): "tension_E4u",
            ("E4", "gaussian"): "gaussian_E4u",
            ("E4", "multiquadric"): "multiquadric_E4u",
        }

    def test_cut_dispatch_e2(self, monkeypatch):
        stub = BridgeResult(final_linf=1e-3, stable=True, growth_rate=0.001)
        calls = self._patch_bridge(monkeypatch, stub)
        out = layer8_cpp_simulation(
            "E2", "classical", {"alpha": [0.0]}, template="cut",
        )
        assert len(calls) == 1
        call = calls[0]
        assert call["scheme_type"] == "E2"
        assert call["template"] == BRADY_LIVESCU_CUT_TEMPLATE
        assert call["radius"] == CUT_SPHERE_RADIUS
        assert call["cfl"] == 0.8  # default CFL for the cut template
        assert out["stable"] is True
        assert out["growth_rate"] == pytest.approx(0.001)

    def test_cut_dispatch_e4_uses_cut_capable_type(self, monkeypatch):
        """("E4", "classical") maps to "E4" (make_E4_1) on the cut template,
        NOT "E4u" — the uniform-only type silently accepts shapes and
        produces a garbage cut discretization (4.3 caveat 2). The
        paper-aligned E4_1 takes FOUR free boundary alphas; they pass through
        verbatim in ``params["alpha"]``."""
        stub = BridgeResult(final_linf=1e-3, stable=True, growth_rate=0.001)
        calls = self._patch_bridge(monkeypatch, stub)
        four_alphas = [
            -0.0228413686378388,
            -0.0582315114712024,
            0.0258161783004396,
            -0.0111107179663226,
        ]
        layer8_cpp_simulation(
            "E4", "classical", {"alpha": four_alphas}, template="cut",
        )
        assert calls[0]["scheme_type"] == "E4"
        assert calls[0]["params"]["alpha"] == four_alphas

    def test_cut_rejects_uniform_only_kernels(self, monkeypatch):
        self._patch_bridge(monkeypatch, BridgeResult())
        with pytest.raises(NotImplementedError, match="cut"):
            layer8_cpp_simulation(
                "E4", "tension", {"sigma": 3.0}, template="cut",
            )

    def test_unknown_template_raises(self, monkeypatch):
        self._patch_bridge(monkeypatch, BridgeResult())
        with pytest.raises(ValueError, match="unknown L8 template"):
            layer8_cpp_simulation(
                "E2", "classical", {"alpha": [0.0]}, template="staircase",
            )

    def test_uniform_template_keeps_blowup_verdict(self, monkeypatch):
        """Uniform rung regression: verdict remains final_linf-based
        (bridge stable flag), growth_rate merely reported."""
        stub = BridgeResult(final_linf=1.75e-3, stable=True, growth_rate=0.4)
        calls = self._patch_bridge(monkeypatch, stub)
        out = layer8_cpp_simulation(
            "E4", "classical", {"alpha": [-0.77, 0.16]},
        )
        assert calls[0]["scheme_type"] == "E4u"
        assert calls[0]["template"] == BRADY_LIVESCU_TEMPLATE
        assert "radius" not in calls[0]
        assert "cfl" not in calls[0]
        assert out["stable"] is True  # despite growth_rate > rate_tol
        assert out["growth_rate"] == pytest.approx(0.4)

    def test_cut_verdict_is_rate_based_not_blowup_based(self, monkeypatch):
        """4.4.2: growth_rate < rate_tol replaces final_linf < 10 on the cut
        rung — slow exponential growth with a small final error must FAIL,
        and a bounded-but-large final error with flat rate must PASS."""
        slow_growth = BridgeResult(final_linf=1e-3, stable=True, growth_rate=0.2)
        calls = self._patch_bridge(monkeypatch, slow_growth)
        out = layer8_cpp_simulation(
            "E2", "classical", {"alpha": [0.0]}, template="cut",
        )
        assert out["stable"] is False

    def test_cut_verdict_flat_rate_passes(self, monkeypatch):
        flat = BridgeResult(final_linf=42.0, stable=False, growth_rate=0.0)
        self._patch_bridge(monkeypatch, flat)
        out = layer8_cpp_simulation(
            "E2", "classical", {"alpha": [0.0]}, template="cut",
        )
        assert out["stable"] is True

    def test_cut_verdict_none_rate_fails(self, monkeypatch):
        none_rate = BridgeResult(final_linf=float("nan"), stable=False)
        self._patch_bridge(monkeypatch, none_rate)
        out = layer8_cpp_simulation(
            "E2", "classical", {"alpha": [0.0]}, template="cut",
        )
        assert out["stable"] is False

    def test_cut_rate_tol_parameter(self, monkeypatch):
        assert L8_RATE_TOL == 0.05
        stub = BridgeResult(final_linf=1e-3, stable=True, growth_rate=0.03)
        self._patch_bridge(monkeypatch, stub)
        loose = layer8_cpp_simulation(
            "E2", "classical", {"alpha": [0.0]}, template="cut",
        )
        tight = layer8_cpp_simulation(
            "E2", "classical", {"alpha": [0.0]}, template="cut", rate_tol=0.01,
        )
        assert loose["stable"] is True
        assert tight["stable"] is False

    def test_explicit_cfl_forwarded(self, monkeypatch):
        stub = BridgeResult(final_linf=1e-3, stable=True, growth_rate=0.0)
        calls = self._patch_bridge(monkeypatch, stub)
        layer8_cpp_simulation(
            "E2", "classical", {"alpha": [0.0]}, template="cut", cfl=0.4,
        )
        assert calls[0]["cfl"] == 0.4

    def test_cfl_values_sequential_runs(self, monkeypatch):
        stub = BridgeResult(final_linf=1e-3, stable=True, growth_rate=0.0,
                            wall_time_s=0.5)
        calls = self._patch_bridge(monkeypatch, stub)
        out = layer8_cpp_simulation(
            "E2", "classical", {"alpha": [0.0]},
            template="cut", cfl_values=[0.4, 0.8, 1.0],
        )
        assert [c["cfl"] for c in calls] == [0.4, 0.8, 1.0]
        assert sorted(out["cfl_results"]) == [0.4, 0.8, 1.0]
        assert out["stable"] is True
        assert out["wall_time_s"] == pytest.approx(1.5)
        for entry in out["cfl_results"].values():
            assert entry["stable"] is True
            assert entry["bridge_result"] is stub

    def test_cfl_and_cfl_values_mutually_exclusive(self, monkeypatch):
        self._patch_bridge(monkeypatch, BridgeResult())
        with pytest.raises(ValueError, match="not both"):
            layer8_cpp_simulation(
                "E2", "classical", {"alpha": [0.0]},
                template="cut", cfl=0.8, cfl_values=[0.8],
            )


class TestCutBridgeSmoke:
    """End-to-end against the real shoccs binary (plan 48 item 4.4).

    Marked @pytest.mark.slow — deselected unless --run-slow is passed.
    Skipped entirely if the binary hasn't been built.
    """

    # Paper-aligned E4_1 (Brady & Livescu 2021) id=1 free boundary alphas,
    # order alpha_0..alpha_3 = (alphau_04, alphau_14, alphau_24, alphau_25);
    # from tests/fixtures/e4_1_paper_reference.json ("alpha_id1").
    _E4_1_FIXTURE_ALPHA = [
        -0.0228413686378388,
        -0.0582315114712024,
        0.0258161783004396,
        -0.0111107179663226,
    ]

    @pytest.mark.slow
    def test_e4_1_cut_run_at_fixture_alphas(self):
        """The corrected 4-alpha E4_1 runs end-to-end on the cut template at
        the paper id=1 alphas and yields a finite growth rate. The mis-port's
        197/288 std::invalid_argument guard is gone, so no alpha vector is
        infeasible — this must run to a numeric verdict, never abort."""
        if not SHOCCS_BINARY.exists():
            pytest.skip("shoccs binary not built")
        out = layer8_cpp_simulation(
            "E4", "classical", {"alpha": self._E4_1_FIXTURE_ALPHA},
            N=31, t_final=1.0, template="cut",
        )
        br = out["bridge_result"]
        assert br.exit_code == 0, f"stderr: {br.stderr}"
        assert out["growth_rate"] is not None
        assert math.isfinite(out["growth_rate"])

    @pytest.mark.slow
    def test_e2_1_cut_run_reports_growth_rate(self):
        """E2_1 alpha=0 on the cut template runs end-to-end and yields a
        finite growth rate. No verdict asserted: this template's raw rate is
        dominated by the default-floating xmax/ymax walls (4.3 caveat 1),
        uniform and cut alike."""
        if not SHOCCS_BINARY.exists():
            pytest.skip("shoccs binary not built")
        out = layer8_cpp_simulation(
            "E2", "classical", {"alpha": [0.0, 0.0, 0.0, 0.0]},
            N=31, t_final=1.0, template="cut",
        )
        br = out["bridge_result"]
        assert br.exit_code == 0, f"stderr: {br.stderr}"
        assert out["growth_rate"] is not None
        assert math.isfinite(out["growth_rate"])
