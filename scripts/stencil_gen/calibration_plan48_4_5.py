#!/usr/bin/env python
"""Plan 48 item 4.5 — End-to-end acceptance comparison (findings script).

Run once from ``scripts/stencil_gen/`` (needs ``build/src/app/shoccs``):

    SYMPY_CACHE_SIZE=50000 uv run python calibration_plan48_4_5.py

Standalone findings script (NOT a test, NOT production code); its numbers
are appended to the Findings log of ``plans/48-cutcell-normal-modes.md``.

For E2_1 alpha = 0 (config A) and one deliberately perturbed alpha the
Python margins call marginal (config B: alpha_3 = +0.01, picked after the
1.4-style neighborhood scan in part 0 — the E2_1 stability region around
alpha = 0 has ZERO width in every single-alpha direction, so "marginal"
means weakly unstable: max Re(witness s) = 0.43, dense-confirmed), compare:

  (i)   kreiss_margin_over_psi          (item 1.3, half-line inflow tier)
  (ii)  max_reflection / round-trip     (items 2.4 / 3.3)
  (iii) rk4_containment_over_psi        (item 4.1) at CFL in {0.8, 1.0},
        plus the item-3.2 interval scan it consumes
  (iv)  cut-template L8 growth_rate     (item 4.4) at N in {31, 61},
        CFL in {0.8, 1.0}, with uniform-template baselines at the same
        (N, CFL) so the cut-vs-uniform DELTA is the cut-cell signal
        (4.3 caveat 1 option (b); the raw rate is contaminated by the
        uncertified default-floating xmax/ymax walls)

plus the discriminating probes recommended by the 4.4.5 findings entry:
probe spheres realizing only-large / mid / only-small min realized psi at
N = 31 (a per-psi-threshold bisection of the delta), and an N = 41 fresh
psi draw of the standard sphere.  Probe geometries were located with the
ray-cast replication ``tests.test_cpp_bridge._realized_psi_set`` (validated
against logs/geometry.csv to ~1e-12) and are hardcoded below.

Acceptance (plan): no Python-pass/C++-fail disagreement; Python-fail/
C++-pass is acceptable (proxies may be conservative) but must be logged.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from stencil_gen import temo
from stencil_gen.brady2d_stability import L8_RATE_TOL
from stencil_gen.cpp_bridge import (
    BRADY_LIVESCU_CUT_TEMPLATE,
    BRADY_LIVESCU_TEMPLATE,
    CUT_SPHERE_CENTER,
    CUT_SPHERE_RADIUS,
    SHOCCS_BINARY,
    run_cpp_brady2d,
)
from stencil_gen.cutcell_normal_modes import (
    CutCellInfeasible,
    default_psi_grid,
    kreiss_margin_over_psi,
)
from stencil_gen.fully_discrete import rk4_amplification, rk4_containment_over_psi
from stencil_gen.interval_operator import (
    default_interval_psi_pairs,
    interval_stability_scan,
)
from stencil_gen.reflection import cutcell_round_trip, reflection_over_psi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCHEME = temo.E2_1
H31 = math.sqrt(2.0) / 30.0  # h cancels in dt*L = -cfl*D; kept for fidelity
CFLS = (0.8, 1.0)
NS = (31, 61)

CONFIGS = {
    # (python alpha dict, C++ Lua alpha table)
    "A_alpha0": ({f"alpha_{i}": 0.0 for i in range(4)}, [0.0, 0.0, 0.0, 0.0]),
    "B_alpha3_+0.01": (
        {"alpha_0": 0.0, "alpha_1": 0.0, "alpha_2": 0.0, "alpha_3": 0.01},
        [0.0, 0.0, 0.0, 0.01],
    ),
}

# Verdict thresholds (documented in the findings entry):
# - DEEP_AMPL: an eigenvalue of dt*L is a DEEP RK4-region violation iff
#   max |R(zeta)| - 1 > 1e-3 (the A-config finite-N noise sits at ~1e-7;
#   genuine outflow-law / half-line violations sit at |R|-1 ~ 0.5-2).
# - DELTA_TOL: cut-vs-uniform growth-rate delta signalling a cut-side
#   mechanism: > 1.0 (noise floor of the rate is 0.0013 per 4.4.5; measured
#   deltas cluster at ~0 or >= +8).
DEEP_AMPL = 1e-3
DELTA_TOL = 1.0

# Probe spheres (part 3): located by fixed-seed random search over
# (cx, cy, r) with the tests._realized_psi_set replication at N = 31;
# min/max realized psi and cut-point counts quoted from that replication.
PROBES = {
    #                 cx      cy      r       min psi   max psi  n
    "P_large": (0.5430, 0.5422, 0.1834),  # 0.501,   0.930,  32
    "P_mid": (0.7357, 0.7294, 0.1953),  # 0.0700,  0.757,  32
    "P_small": (0.6823, 0.7606, 0.2271),  # 5.4e-6,  0.866,  38
}
# Standard template sphere realized psi (replication): N=31 [1.57e-3,
# 0.99927] (44 pts), N=41 [2.10e-3, 0.98534] (54 pts), N=61 [5.70e-6,
# 0.99853] (84 pts).


# ---------------------------------------------------------------------------
# Part 0 — perturbed-alpha pick: neighborhood scan (1.4-style)
# ---------------------------------------------------------------------------


def kreiss_stats(alpha: dict) -> dict:
    sweep = kreiss_margin_over_psi(SCHEME, alpha)
    unstable = sorted(
        p
        for p, r in sweep.per_psi.items()
        if not isinstance(r, CutCellInfeasible) and not r.is_stable
    )
    max_re, arg_psi, arg_s = 0.0, None, None
    for p in unstable:
        s = sweep.per_psi[p].witness_s
        if s is not None and np.isfinite(np.real(s)) and np.real(s) > max_re:
            max_re, arg_psi, arg_s = float(np.real(s)), p, complex(s)
    return {
        "any_unstable": sweep.any_unstable,
        "n_unstable": len(unstable),
        "n_psi": len(sweep.per_psi),
        "unstable_psi": unstable,
        "worst_sigma_min": sweep.worst_sigma_min,
        "worst_psi": sweep.worst_psi,
        "max_re_witness_s": max_re,
        "argmax_psi": arg_psi,
        "argmax_s": None if arg_s is None else str(arg_s),
        "infeasible_psi": sweep.infeasible_psi,
    }


def part0_neighborhood() -> dict:
    print("== Part 0: perturbed-alpha pick (E2_1 stability region width) ==")
    out = {}
    for name, av in [
        ("alpha_3=+0.002", [0, 0, 0, 0.002]),
        ("alpha_3=+0.005", [0, 0, 0, 0.005]),
        ("alpha_3=+0.01", [0, 0, 0, 0.01]),
        ("alpha_0=+0.01", [0.01, 0, 0, 0]),
        ("alpha_0=-0.01", [-0.01, 0, 0, 0]),
    ]:
        st = kreiss_stats({f"alpha_{i}": float(v) for i, v in enumerate(av)})
        out[name] = st
        print(
            f"  {name}: unstable {st['n_unstable']}/{st['n_psi']} psi, "
            f"max Re(witness s)={st['max_re_witness_s']:.4f} at "
            f"psi={st['argmax_psi']}"
        )
    print(
        "  -> stability region around alpha=0 has zero width in each "
        "single-alpha direction; B := alpha_3=+0.01 (weakly unstable == "
        "'marginal')."
    )
    return out


# ---------------------------------------------------------------------------
# Part 1 — Python margin suite per config (tiers i-iii)
# ---------------------------------------------------------------------------


def python_suite(name: str, alpha: dict) -> dict:
    print(f"== Part 1: Python margins for {name} ==")
    res: dict = {}

    # (i) half-line inflow Dirichlet tier
    res["tier_i_kreiss"] = kreiss_stats(alpha)
    ti = res["tier_i_kreiss"]
    print(
        f"  (i)   kreiss_margin_over_psi: unstable {ti['n_unstable']}/"
        f"{ti['n_psi']}, worst sigma={ti['worst_sigma_min']:.3e} at "
        f"psi={ti['worst_psi']}, max Re(witness s)="
        f"{ti['max_re_witness_s']:.4f}"
    )

    # (ii) per-end reflection + two-boundary round trip
    refl = {}
    for side in ("left", "right"):
        r = reflection_over_psi(SCHEME, alpha, side=side)
        refl[side] = {
            "max_reflection": r.max_reflection,
            "argmax_psi": r.argmax_psi,
            "non_clean": bool(r.any_unstable),
        }
        print(
            f"  (ii)  reflection[{side}]: max ||R||={r.max_reflection:.3e} "
            f"at psi={r.argmax_psi}, non_clean={r.any_unstable}"
        )
    pairs = default_interval_psi_pairs(default_psi_grid())
    radii, products, non_finite_pairs = [], [], []
    worst_pair = None
    for pl, pr in pairs:
        rt = cutcell_round_trip(SCHEME, alpha, float(pl), float(pr))
        if isinstance(rt, CutCellInfeasible):
            non_finite_pairs.append((float(pl), float(pr), "infeasible"))
            continue
        if not rt.finite:
            non_finite_pairs.append((float(pl), float(pr), "non_finite"))
        if not np.isnan(rt.round_trip_radius):
            radii.append(float(rt.round_trip_radius))
            if rt.round_trip_radius >= max(radii):
                worst_pair = (float(pl), float(pr))
        if not np.isnan(rt.reflection_product):
            products.append(float(rt.reflection_product))
    res["tier_ii_reflection"] = refl
    res["tier_ii_round_trip"] = {
        "n_pairs": len(pairs),
        "worst_radius": max(radii) if radii else float("nan"),
        "worst_radius_pair": worst_pair,
        "worst_product": max(products) if products else float("nan"),
        "non_finite_pairs": non_finite_pairs,
    }
    tr = res["tier_ii_round_trip"]
    print(
        f"  (ii)  round trip over {tr['n_pairs']} default pairs: worst "
        f"radius={tr['worst_radius']:.6f} at {tr['worst_radius_pair']}, "
        f"worst product={tr['worst_product']:.3e}, "
        f"non-finite pairs={len(tr['non_finite_pairs'])}"
    )

    # (iii) interval scan + RK4 containment (scan computed once, reused)
    scan = interval_stability_scan(SCHEME, alpha)
    res["tier_iii_scan"] = {
        "worst_abscissa": float(scan.worst_abscissa),
        "worst_abscissa_case": scan.worst_abscissa_case,
        "worst_kreiss_constant": float(scan.worst_kreiss_constant),
        "worst_kreiss_case": scan.worst_kreiss_case,
        "n_infeasible_pairs": len(scan.infeasible_pairs),
    }
    ts = res["tier_iii_scan"]
    print(
        f"  (iii) interval scan: worst abscissa={ts['worst_abscissa']:+.4f}"
        f" at {ts['worst_abscissa_case']}, worst Kreiss const="
        f"{ts['worst_kreiss_constant']:.3e} at {ts['worst_kreiss_case']}"
    )

    def family_ampl(cases: dict) -> tuple[float, object]:
        """Max over cases of (max |R(eig(dt L))| - 1) and its argmax case."""
        worst, arg = -np.inf, None
        for key, c in cases.items():
            a = float(np.max(np.abs(rk4_amplification(c.eigenvalues)))) - 1.0
            if a > worst:
                worst, arg = a, key
        return worst, arg

    res["tier_iii_containment"] = {}
    for cfl in CFLS:
        ct = rk4_containment_over_psi(SCHEME, alpha, cfl, H31, scan=scan)
        sweep_ampl, sweep_arg = family_ampl(ct.per_psi)
        scan_ampl, scan_arg = family_ampl(ct.scan_cases)
        entry = {
            "worst_margin": min(
                c.margin for c in (*ct.per_psi.values(), *ct.scan_cases.values())
            ),
            "worst_margin_certified": float(ct.worst_margin_certified),
            "any_eigs_outside": bool(ct.any_eigs_outside),
            "sweep_family_max_ampl_minus1": sweep_ampl,
            "sweep_family_argmax": sweep_arg,
            "scan_family_max_ampl_minus1": scan_ampl,
            "scan_family_argmax": scan_arg,
            "sweep_deep_violation": sweep_ampl > DEEP_AMPL,
            "scan_deep_violation": scan_ampl > DEEP_AMPL,
        }
        res["tier_iii_containment"][cfl] = entry
        print(
            f"  (iii) containment CFL={cfl}: grid margin="
            f"{entry['worst_margin']:.3e} (certified "
            f"{entry['worst_margin_certified']:+.4f}; degenerate — exact "
            f"zero eigenvalue on dS at z=0 for every interval operator), "
            f"eigs_outside={entry['any_eigs_outside']}; max|R|-1: sweep "
            f"family {sweep_ampl:+.3e} at {sweep_arg}, scan family "
            f"{scan_ampl:+.3e} at {scan_arg}"
        )

    # Rollup
    contain = res["tier_iii_containment"]
    res["rollup"] = {
        "half_line_pass": not ti["any_unstable"],
        "reflection_clean": not (
            refl["left"]["non_clean"] or refl["right"]["non_clean"]
        ),
        "round_trip_pass": bool(tr["worst_radius"] < 1.0),
        "two_ended_scan_pass": ts["worst_abscissa"] < 1e-6,
        "containment_deep_violation": {
            cfl: (
                contain[cfl]["sweep_deep_violation"]
                or contain[cfl]["scan_deep_violation"]
            )
            for cfl in CFLS
        },
    }
    ru = res["rollup"]
    ru["overall_python_pass"] = bool(
        ru["half_line_pass"]
        and ru["reflection_clean"]
        and ru["round_trip_pass"]
        and ru["two_ended_scan_pass"]
        and not any(ru["containment_deep_violation"].values())
    )
    print(f"  rollup: {ru}")
    return res


# ---------------------------------------------------------------------------
# Part 2 — C++ L8: cut + uniform x N{31,61} x CFL{0.8,1.0}
# ---------------------------------------------------------------------------


def make_uniform_cfl_template() -> Path:
    """Uniform template with the hardcoded CFL replaced by the marker."""
    src = BRADY_LIVESCU_TEMPLATE.read_text()
    needle = "hyperbolic = 0.8"
    assert needle in src, "uniform template CFL line changed; update 4.5 script"
    with tempfile.NamedTemporaryFile(
        "w", suffix=".lua", delete=False, prefix="uniform-cfl-"
    ) as f:
        f.write(src.replace(needle, "hyperbolic = --{{CFL}}--"))
        return Path(f.name)


def make_probe_template(cx: float, cy: float) -> Path:
    """Cut template with the sphere center moved (radius stays a marker)."""
    src = BRADY_LIVESCU_CUT_TEMPLATE.read_text()
    needle = "center = {0.8441, 0.6177, 0}"
    assert needle in src, "cut template center line changed; update 4.5 script"
    with tempfile.NamedTemporaryFile(
        "w", suffix=".lua", delete=False, prefix="probe-"
    ) as f:
        f.write(src.replace(needle, f"center = {{{cx}, {cy}, 0}}"))
        return Path(f.name)


def bridge_row(br) -> dict:
    return {
        "growth_rate": br.growth_rate,
        "final_linf": br.final_linf,
        "t_end": float(br.t_trace[-1]) if br.t_trace.size else float("nan"),
        "infeasible": br.infeasible,
        "exit_code": br.exit_code,
        "raw_fail": (br.growth_rate is None) or (br.growth_rate >= L8_RATE_TOL),
    }


def part2_cpp(uniform_cfl_tmpl: Path) -> dict:
    print("== Part 2: C++ L8 cut vs uniform (both configs) ==")
    rows: dict = {}
    for name, (_, lua_alpha) in CONFIGS.items():
        for N in NS:
            for cfl in CFLS:
                cut = run_cpp_brady2d(
                    "E2",
                    {"alpha": lua_alpha},
                    N=N,
                    t_final=10.0,
                    template=BRADY_LIVESCU_CUT_TEMPLATE,
                    radius=CUT_SPHERE_RADIUS,
                    cfl=cfl,
                )
                uni = run_cpp_brady2d(
                    "E2",
                    {"alpha": lua_alpha},
                    N=N,
                    t_final=10.0,
                    template=uniform_cfl_tmpl,
                    cfl=cfl,
                )
                rc, ru = bridge_row(cut), bridge_row(uni)
                delta = (
                    None
                    if rc["growth_rate"] is None or ru["growth_rate"] is None
                    else rc["growth_rate"] - ru["growth_rate"]
                )
                rows[f"{name}_N{N}_cfl{cfl}"] = {
                    "cut": rc,
                    "uniform": ru,
                    "delta": delta,
                    "cut_signal": delta is not None and delta > DELTA_TOL,
                }
                print(
                    f"  {name} N={N} CFL={cfl}: cut rate="
                    f"{rc['growth_rate']:+.3f} (t_end {rc['t_end']:.3f}) | "
                    f"uniform rate={ru['growth_rate']:+.3f} "
                    f"(t_end {ru['t_end']:.3f}) | delta="
                    f"{delta:+.3f} cut_signal="
                    f"{rows[f'{name}_N{N}_cfl{cfl}']['cut_signal']}"
                )
    return rows


# ---------------------------------------------------------------------------
# Part 3 — discriminating probes (config A only; 4.4.5 recommendations)
# ---------------------------------------------------------------------------


def part3_probes(uniform_cfl_tmpl: Path) -> dict:
    print("== Part 3: discriminating probes (A_alpha0, CFL 0.8) ==")
    _, lua_alpha = CONFIGS["A_alpha0"]
    out: dict = {}

    # Uniform baselines for the probe N values
    baselines = {}
    for N in (31, 41):
        br = run_cpp_brady2d(
            "E2",
            {"alpha": lua_alpha},
            N=N,
            t_final=10.0,
            template=uniform_cfl_tmpl,
            cfl=0.8,
        )
        baselines[N] = bridge_row(br)
        print(f"  uniform baseline N={N}: rate={baselines[N]['growth_rate']:+.3f}")
    out["uniform_baselines"] = baselines

    # (a) min-psi ladder at N=31 (P_mid and P_small sit at nearly the same
    # sphere location, isolating min-psi from the local-wave-speed confound)
    for pname, (cx, cy, r) in PROBES.items():
        tmpl = make_probe_template(cx, cy)
        try:
            br = run_cpp_brady2d(
                "E2",
                {"alpha": lua_alpha},
                N=31,
                t_final=10.0,
                template=tmpl,
                radius=r,
                cfl=0.8,
            )
        finally:
            tmpl.unlink()
        row = bridge_row(br)
        row["delta"] = row["growth_rate"] - baselines[31]["growth_rate"]
        out[pname] = row
        print(
            f"  {pname} (center=({cx},{cy}), r={r}): rate="
            f"{row['growth_rate']:+.3f} (t_end {row['t_end']:.3f}), "
            f"delta={row['delta']:+.3f}"
        )

    # (b) CFL jitter on the weakest probe signal: P_large's delta (+1.6 on
    # the first run) is close to the blow-up-window rate jitter scale
    # (4.4.5 measured 15.4-16.3 over an 8x dt range), so re-measure at two
    # more CFL values — a real semi-discrete signal is CFL-insensitive.
    cx, cy, r = PROBES["P_large"]
    tmpl = make_probe_template(cx, cy)
    jitter = {}
    try:
        for cfl in (0.4, 1.0):
            br = run_cpp_brady2d(
                "E2",
                {"alpha": lua_alpha},
                N=31,
                t_final=10.0,
                template=tmpl,
                radius=r,
                cfl=cfl,
            )
            jitter[cfl] = bridge_row(br)
            print(
                f"  P_large CFL={cfl}: rate={jitter[cfl]['growth_rate']:+.3f}"
                f" (t_end {jitter[cfl]['t_end']:.3f})"
            )
    finally:
        tmpl.unlink()
    out["P_large_cfl_jitter"] = jitter

    # (c) N=41 fresh psi draw of the standard sphere
    br = run_cpp_brady2d(
        "E2",
        {"alpha": lua_alpha},
        N=41,
        t_final=10.0,
        template=BRADY_LIVESCU_CUT_TEMPLATE,
        radius=CUT_SPHERE_RADIUS,
        cfl=0.8,
    )
    row = bridge_row(br)
    row["delta"] = row["growth_rate"] - baselines[41]["growth_rate"]
    out["standard_N41"] = row
    print(
        f"  standard sphere N=41: rate={row['growth_rate']:+.3f} "
        f"(t_end {row['t_end']:.3f}), delta={row['delta']:+.3f}"
    )
    return out


# ---------------------------------------------------------------------------
# Part 4 — acceptance matrix
# ---------------------------------------------------------------------------


def part4_acceptance(python_res: dict, cpp_rows: dict) -> dict:
    print("== Part 4: acceptance ==")
    verdicts: dict = {}
    for name in CONFIGS:
        py = python_res[name]["rollup"]
        cpp_fail_raw = any(
            v["cut"]["raw_fail"]
            for k, v in cpp_rows.items()
            if k.startswith(name)
        )
        cpp_fail_signal = any(
            v["cut_signal"] for k, v in cpp_rows.items() if k.startswith(name)
        )
        verdicts[name] = {
            "python_overall_pass": py["overall_python_pass"],
            "python_half_line_pass": py["half_line_pass"],
            "python_two_ended_pass": py["two_ended_scan_pass"],
            "cpp_cut_raw_fail": cpp_fail_raw,
            "cpp_cut_signal_fail": cpp_fail_signal,
            "python_pass_cpp_fail": (
                py["overall_python_pass"] and (cpp_fail_raw or cpp_fail_signal)
            ),
        }
        print(f"  {name}: {verdicts[name]}")
    acceptance = not any(v["python_pass_cpp_fail"] for v in verdicts.values())
    verdicts["acceptance_no_python_pass_cpp_fail"] = acceptance
    print(f"  ACCEPTANCE (no Python-pass/C++-fail): {acceptance}")
    return verdicts


def main() -> int:
    if not SHOCCS_BINARY.exists():
        print(f"shoccs binary not built at {SHOCCS_BINARY}; aborting")
        return 1

    records: dict = {"part0_neighborhood": part0_neighborhood()}

    python_res = {}
    for name, (alpha, _) in CONFIGS.items():
        python_res[name] = python_suite(name, alpha)
    records["part1_python"] = python_res

    uniform_cfl_tmpl = make_uniform_cfl_template()
    try:
        cpp_rows = part2_cpp(uniform_cfl_tmpl)
        records["part2_cpp"] = cpp_rows
        records["part3_probes"] = part3_probes(uniform_cfl_tmpl)
    finally:
        uniform_cfl_tmpl.unlink()

    records["part4_acceptance"] = part4_acceptance(python_res, cpp_rows)

    print("\nJSON summary:")
    print(json.dumps(records, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
