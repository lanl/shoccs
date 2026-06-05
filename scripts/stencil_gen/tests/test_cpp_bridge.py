"""Tests for scripts/stencil_gen/stencil_gen/cpp_bridge.py.

Plan 42.2a scope: make_brady2d_lua marker substitution and scheme-table
emission. Plan 42.2b scope: run_cpp_brady2d subprocess driver, verified with
a fake shoccs binary so the tests stay fast and hermetic.
"""

from __future__ import annotations

import re
import stat
from pathlib import Path

import numpy as np
import pytest

from stencil_gen.cpp_bridge import (
    BRADY_LIVESCU_TEMPLATE,
    LUA_TEMPLATE_DIR,
    REPO_ROOT,
    SHOCCS_BINARY,
    BridgeResult,
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
