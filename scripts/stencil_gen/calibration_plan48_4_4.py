#!/usr/bin/env python
"""Plan 48 item 4.4.5 — L8 cut-rung calibration (findings script).

Run once from ``scripts/stencil_gen/`` (needs ``build/src/app/shoccs``):

    uv run python calibration_plan48_4_4.py

This is a standalone findings script (NOT a test, NOT production code);
its numbers are appended to the Findings log of
``plans/48-cutcell-normal-modes.md``.

Parts (plan 48 item 4.4.5, recalibrated per the 4.3 findings caveat 1):

(a) E2_1 alpha = 0 on the CUT template, N in {31, 61}, CFL 0.8,
    t_final = 10.  The plan's original expectation (growth_rate ~ 0)
    was contradicted by the 4.3 gate findings: this template's
    xmax/ymax walls default to FLOATING closures, which the Python
    certification chain deliberately excludes ("the floating block is
    never analyzed for stability"), and E2_1 alpha=0's floating closure
    drives a semi-discrete blow-up at fixed physical time, uniform and
    cut alike.  Following the 4.3 option (b), the cut-cell signal is
    therefore the cut-vs-uniform DELTA of the growth rate, measured
    here against part (b).
(b) E2_1 alpha = 0 on the UNCUT template at the same N (uniform
    baseline for the delta).
(c) Gaussian epsilon = 0.1 on the UNCUT template — the known-unstable
    anchor from known_values.json; expect a positive growth_rate.
(d) E4_1 (alpha_0, w_4) = (0.1, 0.16) on the cut template — expect
    infeasible=True via the std::invalid_argument guard
    (E4_1.cpp:37-41, w_4 >= 197/288).
(e) Classical E4u on the UNCUT template (the stable anchor) — the
    growth-rate noise floor that rate_tol must sit safely above
    (4.3 caveat 3: only meaningful over the standard t_final = 10
    window).
(f) CFL sweep on the cut template (the 4.4.4 knob): CFL-independence
    discriminates a semi-discrete mechanism from an RK4-region /
    cut-CFL violation.
(g) Control sphere: the 4.3 mild-psi probe geometry (center
    (0.7071, 0.7071), r = 0.25; realized psi in [0.088, 0.79] at N=31,
    [0.031, 0.83] at N=61) through the same pipeline — isolates
    whether the cut-vs-uniform delta tracks the realized psi set or is
    an artifact of the new code path.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from stencil_gen.brady2d_stability import layer8_cpp_simulation
from stencil_gen.cpp_bridge import (
    BRADY_LIVESCU_CUT_TEMPLATE,
    BRADY_LIVESCU_TEMPLATE,
    SHOCCS_BINARY,
    run_cpp_brady2d,
)

E2_1_ALPHA0 = {"alpha": [0.0, 0.0, 0.0, 0.0]}
E4U_CLASSICAL = {"alpha": [-0.7733323791884821, 0.1623961700641681]}


def fmt(result_dict: dict) -> str:
    br = result_dict["bridge_result"]
    gr = result_dict["growth_rate"]
    return (
        f"growth_rate={gr if gr is None else f'{gr:+.4f}'} "
        f"final_linf={result_dict['final_linf']:.4g} "
        f"stable={result_dict['stable']} infeasible={result_dict['infeasible']} "
        f"exit={br.exit_code} t_end={br.t_trace[-1] if br.t_trace.size else float('nan'):.4f} "
        f"wall={result_dict['wall_time_s']:.2f}s"
    )


def as_dict(br) -> dict:
    return {
        "final_linf": br.final_linf,
        "stable": br.stable,
        "growth_rate": br.growth_rate,
        "infeasible": br.infeasible,
        "reason": br.reason,
        "wall_time_s": br.wall_time_s,
        "bridge_result": br,
    }


def main() -> int:
    if not SHOCCS_BINARY.exists():
        print(f"shoccs binary not built at {SHOCCS_BINARY}; aborting")
        return 1

    records: dict[str, dict] = {}

    print("== (a) E2_1 alpha=0, CUT template, CFL 0.8, t_final=10 ==")
    for N in (31, 61):
        out = layer8_cpp_simulation(
            "E2", "classical", E2_1_ALPHA0, N=N, t_final=10.0, template="cut",
        )
        records[f"a_cut_E2_1_N{N}"] = out
        print(f"  N={N}: {fmt(out)}")

    print("== (b) E2_1 alpha=0, UNCUT template (uniform baseline) ==")
    for N in (31, 61):
        # _L8_SCHEME_TYPE has no E2 entry (uniform validation is E4-only),
        # so drive the bridge directly: Lua type "E2" (make_E2_1) on the
        # uncut template is the shapes-free baseline of the same stencil.
        br = run_cpp_brady2d(
            "E2", E2_1_ALPHA0, N=N, t_final=10.0,
            template=BRADY_LIVESCU_TEMPLATE,
        )
        out = as_dict(br)
        records[f"b_uniform_E2_1_N{N}"] = out
        print(f"  N={N}: {fmt(out)}")

    for N in (31, 61):
        ga = records[f"a_cut_E2_1_N{N}"]["growth_rate"]
        gb = records[f"b_uniform_E2_1_N{N}"]["growth_rate"]
        delta = None if (ga is None or gb is None) else ga - gb
        print(f"  N={N}: cut-vs-uniform delta = "
              f"{delta if delta is None else f'{delta:+.4f}'}")
        records[f"delta_E2_1_N{N}"] = {"delta": delta}

    print("== (c) Gaussian eps=0.1, UNCUT template (known-unstable anchor) ==")
    out = layer8_cpp_simulation(
        "E4", "gaussian", {"epsilon": 0.1}, N=31, t_final=10.0,
    )
    records["c_gaussian_eps01_N31"] = out
    print(f"  N=31: {fmt(out)}")

    print("== (d) E4_1 (0.1, 0.16), CUT template (expect infeasible) ==")
    out = layer8_cpp_simulation(
        "E4", "classical", {"alpha": [0.1, 0.16]}, N=31, t_final=10.0,
        template="cut",
    )
    records["d_E4_1_below_floor"] = out
    print(f"  N=31: {fmt(out)}")
    print(f"  reason: {out['reason']}")

    print("== (e) classical E4u, UNCUT template (stable anchor / noise floor) ==")
    for N in (31, 61):
        out = layer8_cpp_simulation(
            "E4", "classical", E4U_CLASSICAL, N=N, t_final=10.0,
        )
        records[f"e_uniform_E4u_N{N}"] = out
        print(f"  N={N}: {fmt(out)}")

    print("== (f) E2_1 alpha=0, CUT template, CFL sweep (4.4.4 knob) ==")
    for N in (31, 61):
        out = layer8_cpp_simulation(
            "E2", "classical", E2_1_ALPHA0, N=N, t_final=10.0,
            template="cut", cfl_values=[0.8, 0.4, 0.2, 0.1],
        )
        for cfl_value, entry in out["cfl_results"].items():
            records[f"f_cut_E2_1_N{N}_cfl{cfl_value}"] = entry
            print(f"  N={N} CFL={cfl_value}: {fmt(entry)}")

    print("== (g) control: 4.3 mild-psi sphere through the same pipeline ==")
    mild_src = BRADY_LIVESCU_CUT_TEMPLATE.read_text().replace(
        "center = {0.8441, 0.6177, 0}", "center = {0.7071, 0.7071, 0}"
    )
    assert "0.7071" in mild_src, "cut template center line changed; update (g)"
    with tempfile.NamedTemporaryFile(
        "w", suffix=".lua", delete=False, prefix="mild-psi-"
    ) as f:
        f.write(mild_src)
        mild_template = Path(f.name)
    try:
        for N in (31, 61):
            br = run_cpp_brady2d(
                "E2", E2_1_ALPHA0, N=N, t_final=10.0,
                template=mild_template, radius=0.25, cfl=0.8,
            )
            out = as_dict(br)
            records[f"g_mildpsi_cut_E2_1_N{N}"] = out
            print(f"  N={N}: {fmt(out)}")
    finally:
        mild_template.unlink()

    # JSON summary (bridge_result stripped) for the findings log
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "bridge_result"}
        for k, v in records.items()
    }
    print("\nJSON summary:")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
