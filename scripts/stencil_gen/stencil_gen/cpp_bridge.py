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
SHOCCS_BINARY: Path = REPO_ROOT / "build" / "src" / "app" / "shoccs"


@dataclass
class BridgeResult:
    final_linf: float = float("nan")
    linf_trace: np.ndarray = field(default_factory=lambda: np.empty(0))
    t_trace: np.ndarray = field(default_factory=lambda: np.empty(0))
    stable: bool = False
    wall_time_s: float = 0.0
    exit_code: int = 0
    stderr: str = ""


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


def make_brady2d_lua(
    scheme_type: str,
    params: dict[str, Any],
    *,
    N: int,
    t_final: float,
    template: Path = BRADY_LIVESCU_TEMPLATE,
) -> str:
    """Render the Brady-Livescu 2D Lua config with the supplied parameters.

    Substitutes the three explicit markers --{{N}}--, --{{T_FINAL}}--, and
    --{{SCHEME_TABLE}}-- with their runtime values. No regex, no Lua AST
    parsing — just str.replace().
    """
    src = template.read_text()
    src = src.replace("--{{N}}--", str(int(N)))
    src = src.replace("--{{T_FINAL}}--", _format_lua_number(t_final))
    src = src.replace("--{{SCHEME_TABLE}}--", _scheme_table_for(scheme_type, params))
    return src


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
) -> BridgeResult:
    """Run the shoccs Brady-Livescu 2D test and return the parsed result.

    Renders the Lua template with `make_brady2d_lua`, writes it to a
    NamedTemporaryFile, and invokes `binary` with cwd set to a per-call
    tempdir. The shoccs binary writes `logs/system.csv` under its cwd, so
    using a private tempdir isolates concurrent invocations — callers can
    run this in parallel without racing on the same CSV.

    On nonzero exit, timeout, or parse failure, returns a BridgeResult with
    `final_linf=nan`, `stable=False`, and the diagnostic captured in
    `exit_code`/`stderr`.
    """
    lua_source = make_brady2d_lua(
        scheme_type, params, N=N, t_final=t_final, template=template
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
        )
