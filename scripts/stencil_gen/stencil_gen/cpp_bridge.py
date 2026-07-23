"""Python → C++ bridge for the Brady-Livescu 2D stability validator.

Builds Lua configs from a template and drives the compiled shoccs binary so
plan 41's analytical stability stack can validate survivors end-to-end in the
real solver.
"""

from __future__ import annotations

import csv
import math
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT: Path = Path(__file__).resolve().parents[3]
LUA_TEMPLATE_DIR: Path = REPO_ROOT / "lua-configs"
BRADY_LIVESCU_TEMPLATE: Path = LUA_TEMPLATE_DIR / "brady_livescu_4_3.lua"
BRADY_LIVESCU_CUT_TEMPLATE: Path = LUA_TEMPLATE_DIR / "brady_livescu_4_3_cut.lua"
SHOCCS_BINARY: Path = REPO_ROOT / "build" / "src" / "app" / "shoccs"

# Embedded-sphere geometry baked into brady_livescu_4_3_cut.lua (plan 48 item
# 4.4.1). The center is hardcoded in the template (must match — asserted in
# tests/test_cpp_bridge.py); the radius fills the --{{RADIUS}}-- marker and
# this is its default. Chosen so the realized cut-cell psi set spans near-0
# to near-1 at BOTH N=31 (44 cut points, psi in [1.6e-3, 0.99927]) and N=61
# (84 cut points, psi in [5.7e-6, 0.99853]); cross-validated against the
# shoccs binary's logs/geometry.csv (matches the Python ray-cast replication
# in tests/test_cpp_bridge.py to ~1e-12).
CUT_SPHERE_CENTER: tuple[float, float] = (0.8441, 0.6177)
CUT_SPHERE_RADIUS: float = 0.2422

# Growth-rate floor: traces with L-inf below this throughout carry no signal
# for a log-linear fit (the solution is numerically zero).
_GROWTH_RATE_LINF_FLOOR = 1e-12


@dataclass
class BridgeResult:
    final_linf: float = float("nan")
    linf_trace: np.ndarray = field(default_factory=lambda: np.empty(0))
    t_trace: np.ndarray = field(default_factory=lambda: np.empty(0))
    stable: bool = False
    wall_time_s: float = 0.0
    exit_code: int = 0
    stderr: str = ""
    #: Least-squares slope of log(L-inf) over the trailing half of the trace
    #: (plan 48 item 4.4.2); None when undefined (non-finite trace, L-inf
    #: below 1e-12 throughout, or too few points).
    growth_rate: float | None = None


def _format_lua_number(x: float) -> str:
    """Format a Python float as a Lua-parseable numeric literal.

    Uses repr() to preserve full double precision, which matters for the alpha
    coefficients that must match known_values.json exactly.
    """
    return repr(float(x))


def _format_lua_array(values: list[float]) -> str:
    return "{" + ", ".join(_format_lua_number(v) for v in values) + "}"


def _scheme_table_for(scheme_type: str, params: dict[str, Any]) -> str:
    """Emit a Lua scheme sub-table for the requested type and parameters.

    Classical schemes pass `alpha` as an array. Spline families (added in
    42.7) pass a scalar `sigma` or `epsilon`. The shape of params dictates
    which parameter is emitted.
    """
    entries: list[str] = ["order = 1", f'type = "{scheme_type}"']
    if "alpha" in params:
        entries.append(f"alpha = {_format_lua_array(list(params['alpha']))}")
    if "sigma" in params:
        entries.append(f"sigma = {_format_lua_number(params['sigma'])}")
    if "epsilon" in params:
        entries.append(f"epsilon = {_format_lua_number(params['epsilon'])}")
    return "{ " + ", ".join(entries) + " }"


_ALL_MARKERS = (
    "--{{N}}--",
    "--{{T_FINAL}}--",
    "--{{SCHEME_TABLE}}--",
    "--{{RADIUS}}--",
    "--{{CFL}}--",
)


def make_brady2d_lua(
    scheme_type: str,
    params: dict[str, Any],
    *,
    N: int,
    t_final: float,
    template: Path = BRADY_LIVESCU_TEMPLATE,
    radius: float | None = None,
    cfl: float | None = None,
) -> str:
    """Render the Brady-Livescu 2D Lua config with the supplied parameters.

    Substitutes the explicit markers --{{N}}--, --{{T_FINAL}}--, and
    --{{SCHEME_TABLE}}-- (plus --{{RADIUS}}-- / --{{CFL}}-- on the cut
    template) with their runtime values. No regex, no Lua AST parsing — just
    str.replace().

    Passing ``radius`` or ``cfl`` against a template lacking the matching
    marker raises ValueError: the uniform template hardcodes
    ``cfl = { hyperbolic = 0.8 }`` with no marker, so a silent str.replace
    no-op would run the wrong CFL (plan 48 item 4.4.4). Conversely, any
    marker left unfilled (e.g. the cut template rendered without ``radius``)
    also raises, since the rendered Lua would not parse.
    """
    src = template.read_text()
    src = src.replace("--{{N}}--", str(int(N)))
    src = src.replace("--{{T_FINAL}}--", _format_lua_number(t_final))
    src = src.replace("--{{SCHEME_TABLE}}--", _scheme_table_for(scheme_type, params))
    if radius is not None:
        if "--{{RADIUS}}--" not in src:
            raise ValueError(
                f"radius={radius} passed but template {template.name!r} has no "
                "--{{RADIUS}}-- marker"
            )
        src = src.replace("--{{RADIUS}}--", _format_lua_number(radius))
    if cfl is not None:
        if "--{{CFL}}--" not in src:
            raise ValueError(
                f"cfl={cfl} passed but template {template.name!r} has no "
                "--{{CFL}}-- marker; the uniform template hardcodes "
                "cfl = { hyperbolic = 0.8 } and a silent no-op replace would "
                "run the wrong CFL"
            )
        src = src.replace("--{{CFL}}--", _format_lua_number(cfl))
    leftover = [m for m in _ALL_MARKERS if m in src]
    if leftover:
        raise ValueError(
            f"unfilled markers {leftover} in template {template.name!r}; the cut "
            "template requires radius= and cfl= arguments"
        )
    return src


def compute_growth_rate(
    t_trace: np.ndarray, linf_trace: np.ndarray
) -> float | None:
    """Least-squares growth rate of log(L-inf) over the trailing half.

    Plan 48 item 4.4.2: fits ``log(linf) ~ a + rate * t`` on the trailing
    half of the series (the leading half is discarded as an initial
    transient — see the 4.3 findings caveat 3: a short window's trailing
    half can still sit inside the error transient, so callers should use
    the standard t_final = 10 window) and returns the slope, in units of
    1/time.

    Returns None when the rate is undefined:
    - any non-finite entry in either trace (a NaN/inf blow-up gives no
      meaningful log-linear fit; ``final_linf``/``stable`` already flag it),
    - L-inf below 1e-12 throughout (numerically zero solution, no signal),
    - fewer than two trailing-half points or zero time span.
    """
    t = np.asarray(t_trace, dtype=float)
    y = np.asarray(linf_trace, dtype=float)
    if t.size != y.size or y.size == 0:
        return None
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
        return None
    if np.all(y < _GROWTH_RATE_LINF_FLOOR):
        return None
    start = y.size // 2
    t_tail = t[start:]
    y_tail = y[start:]
    if t_tail.size < 2 or not float(t_tail[-1] - t_tail[0]) > 0.0:
        return None
    # Clamp to avoid log(0) on isolated exact zeros; the all-below-floor
    # case (no signal at all) already returned None above.
    log_y = np.log(np.maximum(y_tail, 1e-300))
    slope, _ = np.polyfit(t_tail, log_y, 1)
    return float(slope)


def _parse_system_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse `logs/system.csv` written by the shoccs scalar-wave system.

    Columns (0-indexed) are: Timestamp=0, Time=1, Step=2, Linf=3, Min=4, ...
    Returns (t_trace, linf_trace). Both are empty if the file has no data rows.
    """
    times: list[float] = []
    linfs: list[float] = []
    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0] or not row[0][0].isdigit():
                continue
            try:
                times.append(float(row[1]))
                linfs.append(float(row[3]))
            except (IndexError, ValueError):
                continue
    return np.asarray(times), np.asarray(linfs)


def run_cpp_brady2d(
    scheme_type: str,
    params: dict[str, Any],
    *,
    N: int = 31,
    t_final: float = 10.0,
    timeout: float = 300.0,
    template: Path = BRADY_LIVESCU_TEMPLATE,
    binary: Path = SHOCCS_BINARY,
    radius: float | None = None,
    cfl: float | None = None,
) -> BridgeResult:
    """Run the shoccs Brady-Livescu 2D test and return the parsed result.

    Renders the Lua template with `make_brady2d_lua`, writes it to a
    NamedTemporaryFile, and invokes `binary` with cwd set to a per-call
    tempdir. The shoccs binary writes `logs/system.csv` under its cwd, so
    using a private tempdir isolates concurrent invocations — callers can
    run this in parallel without racing on the same CSV.

    ``radius`` and ``cfl`` fill the cut template's --{{RADIUS}}-- and
    --{{CFL}}-- markers (plan 48 item 4.4); passing them against a template
    without the marker raises (see `make_brady2d_lua`).

    On nonzero exit, timeout, or parse failure, returns a BridgeResult with
    `final_linf=nan`, `stable=False`, and the diagnostic captured in
    `exit_code`/`stderr`. On success, `growth_rate` carries the trailing-half
    log-L-inf slope (see `compute_growth_rate`).
    """
    lua_source = make_brady2d_lua(
        scheme_type,
        params,
        N=N,
        t_final=t_final,
        template=template,
        radius=radius,
        cfl=cfl,
    )

    with tempfile.TemporaryDirectory(prefix="shoccs-brady2d-") as tmpdir:
        tmp = Path(tmpdir)
        lua_path = tmp / "brady_livescu_4_3.lua"
        lua_path.write_text(lua_source)
        (tmp / "logs").mkdir(exist_ok=True)

        start = time.perf_counter()
        try:
            completed = subprocess.run(
                [str(binary), str(lua_path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            wall = time.perf_counter() - start
            return BridgeResult(
                final_linf=float("nan"),
                stable=False,
                wall_time_s=wall,
                exit_code=-1,
                stderr=f"timeout after {timeout}s: {e}",
            )
        wall = time.perf_counter() - start

        if completed.returncode != 0:
            return BridgeResult(
                final_linf=float("nan"),
                stable=False,
                wall_time_s=wall,
                exit_code=completed.returncode,
                stderr=completed.stderr,
            )

        csv_path = tmp / "logs" / "system.csv"
        if not csv_path.exists():
            return BridgeResult(
                final_linf=float("nan"),
                stable=False,
                wall_time_s=wall,
                exit_code=completed.returncode,
                stderr=f"logs/system.csv not found under {tmp}",
            )

        t_trace, linf_trace = _parse_system_csv(csv_path)
        if linf_trace.size == 0:
            return BridgeResult(
                final_linf=float("nan"),
                stable=False,
                wall_time_s=wall,
                exit_code=completed.returncode,
                stderr="system.csv has no data rows",
            )

        final_linf = float(linf_trace[-1])
        stable = (
            math.isfinite(final_linf) and not math.isnan(final_linf) and final_linf < 10.0
        )
        return BridgeResult(
            final_linf=final_linf,
            linf_trace=linf_trace,
            t_trace=t_trace,
            stable=stable,
            wall_time_s=wall,
            exit_code=completed.returncode,
            stderr=completed.stderr,
            growth_rate=compute_growth_rate(t_trace, linf_trace),
        )
