#!/usr/bin/env python
"""Plan 48 item 1.4 — calibration / control experiment (findings script).

Run once from ``scripts/stencil_gen/``:

    SYMPY_CACHE_SIZE=50000 uv run python calibration_plan48_1_4.py

This is a standalone findings script (NOT a test, NOT production code);
its numbers are appended to the Findings log of
``plans/48-cutcell-normal-modes.md``.

Parts (plan 48 item 1.4):

(a) E2_1 alpha = 0 across the full default psi grid via
    ``kreiss_margin_over_psi`` — expected stable everywhere (plan-35
    history).
(b) E4_1 in its OWN legacy zeros-path coordinates
    (``alpha_2 -> alpha_0`` boundary shape, ``w_4 -> alpha_1`` free
    conservation quadrature weight; the parameterization that generated
    ``src/stencils/E4_1.cpp`` — see temo.py Step 6 of the zeros path) at
    ``alpha_1 in {0.60, 197/288 + 0.05}`` with ``alpha_0 = 0.1`` (the
    test-suite default).  Records per-psi feasibility (denominator
    guard) and Kreiss verdicts; probes the interior denominator pole of
    the 0.60 case directly.
(c) The same configurations through 40-point dense eigenvalues of the
    assembled half-line operator (test-local assembly, the plan-48
    item-1.2 pattern), plus the shipped default ``alpha_1 = 0.7`` as a
    dense control (exonerates the uniform right-end closure).
(d) Independent symbolic re-derivation of the ``alpha_1`` (w_4)
    denominator floor from the derived Dirichlet/floating blocks;
    reconciliation of the commit-15c0ddd cubic vs the E4_1.cpp degree-6
    denominator record.
(e) Explanation diagnostics for any Kreiss-stable/dense-unstable
    contradiction (the acceptance criterion requires contradictions
    explained): mirrored right-end half-line Kreiss check (rules the
    uniform tension-sigma=0 right closure of the dense assembly in or
    out), eigenvector localization of every unstable dense mode, and
    sigma_min(M(s*), equilibrated) at the left-localized dense
    eigenvalues of each contradiction config.

NOTE on parameterizations: ``make_cutcell_evaluator(E4_1)`` (item 1.1)
dispatches to ``derive_cut_cell_mathematica``, whose parameterization
keeps the four uniform-boundary alphas free — the 197/288 floor does not
exist in those coordinates.  Parts (b)/(d) therefore construct a
``CutCellEvaluator`` directly on the legacy ``derive_cut_cell_scheme``
result (2 free parameters), exactly the coordinates the plan prescribes.
Part (d) also records the Mathematica-path denominators for contrast.

No time integration anywhere (plan 48 hard constraint): Kreiss checks
are resolvent/root-locus based; the dense cross-check is a single
eigendecomposition of the semi-discrete operator.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from sympy import Poly, Rational, Symbol, cancel, simplify, solve

from stencil_gen.cutcell_eval import (
    CutCellEvaluator,
    _collect_denominators,
    make_cutcell_evaluator,
)
from stencil_gen.cutcell_normal_modes import (
    FEASIBILITY_TOL,
    _vanishing_denominators,
    default_psi_grid,
    kreiss_margin_over_psi,
)
from stencil_gen.gks_kreiss import kreiss_stability_check, min_singular_value_ex
from stencil_gen.phs import uniform_boundary_weights_rbf
from stencil_gen.temo import E2_1, E4_1, derive_cut_cell_scheme

OUT = Path(
    os.environ.get(
        "PLAN48_1_4_OUT",
        "/tmp/claude-1000/-workspace/68b521c9-1550-4631-b32c-af28cc646933/"
        "scratchpad/plan48_1_4_results.json",
    )
)

# The item-0.3/0.5/1.2 calibration s-grid (odd n_imag puts Im = 0 on grid).
_CUT_GRID = {"s_max": 10.0, "n_radial": 30, "n_imag": 81, "imag_max": 15.0}

FLOOR = Rational(197, 288)
ALPHA0 = 0.1  # test-suite default boundary-shape parameter
W4_CASES = {
    "w4=0.60": 0.60,
    "w4=197/288+0.05": float(FLOOR) + 0.05,
    "w4=0.70 (shipped default, control)": 0.70,
}

# The two records to reconcile in part (d):
_PSI = Symbol("psi")
_A1 = Symbol("alpha_1")
CUBIC_15C0DDD = 288 * _A1 + 648 * _PSI + 90 * _PSI**2 + 12 * _PSI**3 - 197
DEG6_E4_1_CPP = (
    1728 * _A1 + 1584 * _A1 * _PSI + 864 * _A1 * _PSI**2 + 144 * _A1 * _PSI**3
    + 12 * _PSI**6 + 162 * _PSI**5 + 1464 * _PSI**4 + 5617 * _PSI**3
    + 8070 * _PSI**2 + 1721 * _PSI - 1182
)

RESULTS: dict = {"s_grid_params": _CUT_GRID}


def log(msg: str) -> None:
    print(msg, flush=True)


def dump() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(RESULTS, indent=1, default=str))


def kres_to_dict(res) -> dict:
    return {
        "is_stable": bool(res.is_stable),
        "witness_s": str(res.witness_s),
        "witness_sigma_min": float(res.witness_sigma_min),
        "imag_axis_verdict": res.imaginary_axis_perturbation_verdict,
        "near_origin_verdict": res.near_origin_verdict,
        "defective": bool(res.defective_kappa_detected),
    }


def halfline_operator(rows, iw, io, q, nu, t, n=40):
    """N x N cut-cell half-line operator (item-1.2 test-local pattern).

    Cut Dirichlet closure rows at the left (wall column already
    eliminated by the evaluator), centered interior, reflected uniform
    tension-sigma=0 closure at the right.  Semi-discrete convention:
    u_t = -D u, so stability is max Re eig(-D) <~ 0.
    """
    r = len(rows)
    D = np.zeros((n, n))
    for i, (w, cols) in enumerate(rows):
        D[i, np.asarray(cols, dtype=int)] = w
    for i in range(r, n - r):
        D[i, i + io] = iw
    sign = (-1.0) ** nu
    for i in range(r):
        w = uniform_boundary_weights_rbf(i, t, nu, q, 0.0, kernel="tension")
        for j in range(t):
            D[n - 1 - i, n - 1 - j] = sign * float(w[j])
    return D


def max_re_eig(D) -> float:
    return float(np.max(np.real(np.linalg.eigvals(-D))))


def real_roots_in_unit_interval(expr, w4: float):
    """Real roots of expr(psi; alpha_1=w4) inside (0, 1)."""
    poly = Poly(expr.subs(_A1, w4), _PSI)
    roots = np.roots([float(c) for c in poly.all_coeffs()])
    out = [
        float(r.real)
        for r in roots
        if abs(r.imag) < 1e-10 and 0.0 < r.real < 1.0
    ]
    return sorted(out)


# =====================================================================
# Part (a): E2_1 alpha = 0 across the full default psi grid
# =====================================================================

def part_a():
    log("== Part (a): E2_1 alpha=0 Kreiss margin over the default psi grid ==")
    ev2 = make_cutcell_evaluator(E2_1)
    alpha2 = {s: 0.0 for s in ev2.alpha_symbols}
    t0 = time.perf_counter()
    sweep = kreiss_margin_over_psi(E2_1, alpha2, s_grid_params=_CUT_GRID)
    elapsed = time.perf_counter() - t0
    RESULTS["a_e2_1_alpha0"] = {
        "elapsed_s": elapsed,
        "any_unstable": bool(sweep.any_unstable),
        "infeasible_psi": list(sweep.infeasible_psi),
        "worst_sigma_min": float(sweep.worst_sigma_min),
        "worst_psi": sweep.worst_psi,
        "per_psi": {
            f"{p:.6g}": kres_to_dict(r) for p, r in sweep.per_psi.items()
        },
    }
    log(f"   any_unstable={sweep.any_unstable} "
        f"worst_sigma_min={sweep.worst_sigma_min:.3e} at psi={sweep.worst_psi} "
        f"({elapsed:.1f} s)")
    dump()
    return ev2, alpha2, sweep


# =====================================================================
# Part (c1): E2_1 dense-eigenvalue cross-check (N = 40)
# =====================================================================

def part_c1(ev2, alpha2, sweep):
    log("== Part (c1): E2_1 alpha=0 dense eigenvalues, N=40, all grid psi ==")
    iw, io = ev2.interior()
    dense = {}
    contradictions = []
    for p in [float(x) for x in default_psi_grid()]:
        rows, _ = ev2.rows(p, alpha2)
        D = halfline_operator(rows, iw, io, q=E2_1.q, nu=E2_1.nu,
                              t=ev2.dims.t, n=40)
        mre = max_re_eig(D)
        dense[f"{p:.6g}"] = mre
        kstable = sweep.per_psi[p].is_stable
        if kstable and mre > 1e-4:
            contradictions.append((p, "kreiss_stable_but_dense_unstable", mre))
        if not kstable and mre < 1e-6:
            contradictions.append((p, "kreiss_unstable_but_dense_stable", mre))
    RESULTS["c1_e2_1_dense_maxre_N40"] = dense
    RESULTS["c1_e2_1_contradictions"] = contradictions
    log(f"   max over grid of max Re eig(-D) = {max(dense.values()):.3e}; "
        f"contradictions: {contradictions}")
    dump()


# =====================================================================
# Part (b): E4_1 legacy (alpha_2, w_4) coordinates
# =====================================================================

def part_b_derive():
    log("== Part (b): deriving E4_1 legacy zeros-path closure "
        "(derive_cut_cell_scheme, conserve=True) ==")
    psi_sym = Symbol("psi")
    t0 = time.perf_counter()
    cc = derive_cut_cell_scheme(E4_1, psi_sym)
    t_derive = time.perf_counter() - t0
    ev4 = CutCellEvaluator(E4_1, "dirichlet", cc, psi_sym)
    names = [s.name for s in ev4.alpha_symbols]
    assert names == ["alpha_0", "alpha_1"], names
    RESULTS["b_derivation"] = {
        "path": "derive_cut_cell_scheme (legacy zeros path, conserve=True)",
        "elapsed_s": t_derive,
        "alpha_symbols": names,
        "dims": str(ev4.dims),
    }
    log(f"   derived in {t_derive:.1f} s; free params {names} "
        f"(alpha_0 = boundary shape alpha_2, alpha_1 = quadrature weight w_4)")
    dump()
    return cc, ev4


def part_b_sweeps(ev4):
    grid = [float(x) for x in default_psi_grid()]
    iw, io = ev4.interior()
    RESULTS["b_e4_1"] = {}
    for label, w4 in W4_CASES.items():
        log(f"== Part (b): E4_1 legacy sweep at alpha_0={ALPHA0}, {label} ==")
        alpha = {"alpha_0": ALPHA0, "alpha_1": w4}
        case: dict = {"alpha_0": ALPHA0, "alpha_1": w4}

        # Denominator pole locations inside (0,1) for every alpha_1
        # dependent structural denominator factor.
        poles = {}
        for den in sorted(ev4.denominators, key=str):
            if _A1 in den.free_symbols or any(
                s.name == "alpha_1" for s in den.free_symbols
            ):
                poles[str(den)] = real_roots_in_unit_interval(den, w4)
        case["denominator_poles_in_(0,1)"] = poles

        per_psi = {}
        unstable_psi = []
        infeasible_psi = []
        worst = (float("inf"), None)
        t0 = time.perf_counter()
        for p in grid:
            vanishing = _vanishing_denominators(ev4, p, alpha, FEASIBILITY_TOL)
            if vanishing:
                infeasible_psi.append(p)
                per_psi[f"{p:.6g}"] = {
                    "infeasible": True,
                    "vanishing": [(str(d), v) for d, v in vanishing],
                }
                continue
            rows, ridx = ev4.rows(p, alpha)
            res = kreiss_stability_check(
                iw, io, rows,
                row_indices=[int(i) for i in ridx],
                s_grid_params=_CUT_GRID,
            )
            per_psi[f"{p:.6g}"] = kres_to_dict(res)
            if not res.is_stable:
                unstable_psi.append(p)
            sig = float(res.witness_sigma_min)
            if np.isfinite(sig) and sig < worst[0]:
                worst = (sig, p)
        elapsed = time.perf_counter() - t0
        case["per_psi"] = per_psi
        case["unstable_psi"] = unstable_psi
        case["infeasible_psi"] = infeasible_psi
        case["worst_sigma_min"] = worst[0]
        case["worst_psi"] = worst[1]
        case["elapsed_s"] = elapsed
        log(f"   unstable_psi={unstable_psi} infeasible_psi={infeasible_psi} "
            f"worst_sigma_min={worst[0]:.3e} at psi={worst[1]} "
            f"({elapsed:.1f} s)")

        # Pole probe: evaluate the feasibility guard AT the interior pole
        # (if any) and the Kreiss verdict just off it.
        probe = {}
        all_poles = sorted({r for rs in poles.values() for r in rs})
        if all_poles:
            p_star = all_poles[0]
            probe["psi_pole"] = p_star
            vanish_at = _vanishing_denominators(
                ev4, p_star, alpha, FEASIBILITY_TOL
            )
            probe["guard_flags_at_pole"] = [
                (str(d), v) for d, v in vanish_at
            ]
            for dp in (-2e-3, 2e-3):
                p_off = p_star + dp
                rows, ridx = ev4.rows(p_off, alpha)
                res = kreiss_stability_check(
                    iw, io, rows,
                    row_indices=[int(i) for i in ridx],
                    s_grid_params=_CUT_GRID,
                )
                probe[f"kreiss_at_pole{'+' if dp > 0 else '-'}2e-3"] = (
                    kres_to_dict(res)
                )
            log(f"   pole probe at psi*={p_star:.6f}: guard flags "
                f"{probe['guard_flags_at_pole']}")
        case["pole_probe"] = probe

        RESULTS["b_e4_1"][label] = case
        dump()


# =====================================================================
# Part (c2): E4_1 dense-eigenvalue cross-check (N = 40)
# =====================================================================

def part_c2(ev4):
    log("== Part (c2): E4_1 dense eigenvalues, N=40, all grid psi ==")
    grid = [float(x) for x in default_psi_grid()]
    iw, io = ev4.interior()
    RESULTS["c2_e4_1_dense_maxre_N40"] = {}
    RESULTS["c2_contradictions"] = {}
    for label, w4 in W4_CASES.items():
        alpha = {"alpha_0": ALPHA0, "alpha_1": w4}
        dense = {}
        for p in grid:
            rows, _ = ev4.rows(p, alpha)
            D = halfline_operator(rows, iw, io, q=E4_1.q, nu=E4_1.nu,
                                  t=ev4.dims.t, n=40)
            dense[f"{p:.6g}"] = max_re_eig(D)
        RESULTS["c2_e4_1_dense_maxre_N40"][label] = dense

        # N-refinement at the dense-worst psi.
        worst_key = max(dense, key=dense.get)
        p_worst = float(worst_key)
        refine = {}
        for n in (20, 40, 80):
            rows, _ = ev4.rows(p_worst, alpha)
            D = halfline_operator(rows, iw, io, q=E4_1.q, nu=E4_1.nu,
                                  t=ev4.dims.t, n=n)
            refine[n] = max_re_eig(D)
        RESULTS["c2_e4_1_dense_maxre_N40"][label + "_refine_at_worst"] = {
            "psi": p_worst, **{str(k): v for k, v in refine.items()},
        }

        # Contradiction bookkeeping against the part-(b) Kreiss verdicts.
        contradictions = []
        for pkey, mre in dense.items():
            b = RESULTS["b_e4_1"][label]["per_psi"][pkey]
            if b.get("infeasible"):
                continue
            if b["is_stable"] and mre > 1e-4:
                contradictions.append(
                    (pkey, "kreiss_stable_but_dense_unstable", mre)
                )
            if not b["is_stable"] and mre < 1e-6:
                contradictions.append(
                    (pkey, "kreiss_unstable_but_dense_stable", mre)
                )
        RESULTS["c2_contradictions"][label] = contradictions
        log(f"   {label}: max over grid = {max(dense.values()):.3e} at "
            f"psi={worst_key}; refine {refine}; "
            f"contradictions: {contradictions}")
        dump()


# =====================================================================
# Part (d): symbolic denominator-floor re-derivation + reconciliation
# =====================================================================

def part_d(cc, ev4):
    log("== Part (d): symbolic re-derivation of the w_4 denominator floor ==")
    d: dict = {}
    dens_dirichlet = ev4.denominators
    dens_floating = _collect_denominators(cc.floating)
    d["dirichlet_denominator_factors"] = sorted(str(x) for x in dens_dirichlet)
    d["floating_denominator_factors"] = sorted(str(x) for x in dens_floating)

    def a1_dependent(dens):
        return [
            den for den in dens
            if any(s.name == "alpha_1" for s in den.free_symbols)
        ]

    analysis = {}
    for block_name, dens in (("dirichlet", dens_dirichlet),
                             ("floating", dens_floating)):
        for den in sorted(a1_dependent(dens), key=str):
            key = f"{block_name}: {den}"
            entry: dict = {}
            # Canonicalize symbols (evaluator may use its own instances).
            den_c = den.subs(
                {s: _A1 for s in den.free_symbols if s.name == "alpha_1"}
            )
            den_c = den_c.subs(
                {s: _PSI for s in den_c.free_symbols if s.name == "psi"}
            )
            entry["degree_in_psi"] = Poly(den_c, _PSI).degree()
            entry["degree_in_alpha_1"] = Poly(den_c, _A1).degree()
            # Match against the two historical records (up to a constant).
            for rec_name, rec in (("cubic_15c0ddd", CUBIC_15C0DDD),
                                  ("deg6_E4_1_cpp", DEG6_E4_1_CPP)):
                ratio = simplify(cancel(den_c / rec))
                entry[f"equals_{rec_name}_up_to_constant"] = (
                    not ratio.free_symbols
                )
                if not ratio.free_symbols:
                    entry[f"constant_ratio_{rec_name}"] = str(ratio)
            # Floor: solve F(psi, a1) = 0 for a1, maximize over psi in [0,1].
            sols = solve(den_c, _A1)
            if len(sols) == 1:
                a1_of_psi = sols[0]
                floor0 = simplify(a1_of_psi.subs(_PSI, 0))
                entry["a1_root_at_psi0"] = str(floor0)
                entry["a1_root_at_psi0_equals_197/288"] = bool(
                    simplify(floor0 - FLOOR) == 0
                )
                samples = [
                    float(a1_of_psi.subs(_PSI, x))
                    for x in np.linspace(0.0, 1.0, 201)
                ]
                entry["a1_root_max_on_[0,1]"] = max(samples)
                entry["a1_root_argmax_is_psi0"] = bool(
                    np.argmax(samples) == 0
                )
            analysis[key] = entry
    d["alpha_1_dependent_factor_analysis"] = analysis

    # Direct reconciliation of the two records.
    q, r = np.polynomial.polynomial, None  # noqa: F841 (clarity only)
    from sympy import div as sym_div, factor
    quot, rem = sym_div(Poly(DEG6_E4_1_CPP, _PSI), Poly(CUBIC_15C0DDD, _PSI))
    d["records"] = {
        "cubic_15c0ddd": str(CUBIC_15C0DDD),
        "deg6_E4_1_cpp": str(DEG6_E4_1_CPP),
        "deg6_factor": str(factor(DEG6_E4_1_CPP)),
        "cubic_divides_deg6": bool(rem.as_expr() == 0),
        "deg6_div_cubic_remainder": str(rem.as_expr()),
        "deg6_at_psi0": str(simplify(DEG6_E4_1_CPP.subs(_PSI, 0))),
        "cubic_at_psi0": str(simplify(CUBIC_15C0DDD.subs(_PSI, 0))),
        "same_floor_at_psi0": bool(
            simplify(
                DEG6_E4_1_CPP.subs(_PSI, 0) / 6 - CUBIC_15C0DDD.subs(_PSI, 0)
            ) == 0
        ),
    }
    # Monotonicity of both records in psi for a1 >= 0 (coefficient check).
    for rec_name, rec in (("cubic_15c0ddd", CUBIC_15C0DDD),
                          ("deg6_E4_1_cpp", DEG6_E4_1_CPP)):
        dpsi = Poly(rec.diff(_PSI), _PSI)
        coeffs = dpsi.all_coeffs()
        nonneg = all(
            (Poly(c, _A1).all_coeffs() and
             all(float(x) >= 0 for x in Poly(c, _A1).all_coeffs()))
            for c in coeffs
        )
        d["records"][f"{rec_name}_dpsi_coeffs_nonneg_for_a1>=0"] = bool(nonneg)

    RESULTS["d_denominator_floor"] = d
    dump()
    log("   part (d) core done; deriving Mathematica-path evaluator for "
        "denominator contrast (slow, > 170 s)...")

    t0 = time.perf_counter()
    ev4m = make_cutcell_evaluator(E4_1)
    RESULTS["d_mathematica_path"] = {
        "elapsed_s": time.perf_counter() - t0,
        "alpha_symbols": [s.name for s in ev4m.alpha_symbols],
        "denominator_factors": sorted(str(x) for x in ev4m.denominators),
    }
    dump()
    log("   Mathematica-path denominators recorded.")


# =====================================================================
# Part (e): contradiction-explanation diagnostics
# =====================================================================

def part_e(ev4):
    log("== Part (e): contradiction diagnostics ==")
    t = ev4.dims.t
    q, nu = E4_1.q, E4_1.nu
    iw4, io4 = ev4.interior()

    # (e1) Mirrored right-end closure as a left half-line problem.
    # Pure index reversal (the plan-48 item-2.3 convention): the assembled
    # right rows D[n-1-i, n-1-j] = -w_i[j] become, in mirrored coordinates
    # m = n-1-index, closure rows with weights -w_i at nodes j = 0..t-1 and
    # row_indices 0..r-1; the interior maps to (weights unchanged,
    # offsets negated).
    mirror_rows = []
    for i in range(len(ev4.rows(0.5, {"alpha_0": ALPHA0, "alpha_1": 0.7})[0])):
        w = uniform_boundary_weights_rbf(i, t, nu, q, 0.0, kernel="tension")
        mirror_rows.append(
            (-np.array([float(x) for x in w]), np.arange(t))
        )
    res_right = kreiss_stability_check(
        iw4[::-1].copy(), (-io4)[::-1].copy(), mirror_rows,
        row_indices=list(range(len(mirror_rows))), s_grid_params=_CUT_GRID,
    )
    RESULTS["e1_mirrored_right_end"] = {
        "is_stable": bool(res_right.is_stable),
        "witness_s": str(res_right.witness_s),
        "witness_sigma_min": float(res_right.witness_sigma_min),
    }
    log(f"   (e1) right-end closure: stable={res_right.is_stable} "
        f"sigma={res_right.witness_sigma_min:.2e}")

    # (e2) Localization of every unstable dense mode (Re > 1e-3): the
    # decontaminated cross-check quantity is max Re over LEFT-localized
    # modes (>50% of |v|^2 mass on nodes 0..9 of the N=40 operator).
    grid = [float(x) for x in default_psi_grid()]
    loc = {}
    for label, w4 in W4_CASES.items():
        alpha = {"alpha_0": ALPHA0, "alpha_1": w4}
        per_psi = {}
        n_right_total = 0
        for p in grid:
            rows, _ = ev4.rows(p, alpha)
            D = halfline_operator(rows, iw4, io4, q=q, nu=nu, t=t, n=40)
            lam, V = np.linalg.eig(-D)
            left_modes, right_modes = [], []
            for k in np.where(lam.real > 1e-3)[0]:
                v = np.abs(V[:, k]) ** 2
                v = v / v.sum()
                if float(v[:10].sum()) > 0.5:
                    left_modes.append(complex(lam[k]))
                elif float(v[-10:].sum()) > 0.5:
                    right_modes.append(complex(lam[k]))
            n_right_total += len(right_modes)
            per_psi[f"{p:.6g}"] = {
                "left_modes": [str(x) for x in left_modes],
                "right_modes": [str(x) for x in right_modes],
                "max_re_left": max((x.real for x in left_modes),
                                   default=float("-inf")),
            }
        loc[label] = per_psi
        n_left = sum(1 for v in per_psi.values() if v["left_modes"])
        log(f"   (e2) {label}: left-localized unstable modes at "
            f"{n_left}/{len(grid)} psi; right-localized total "
            f"{n_right_total}")
    RESULTS["e2_localization"] = loc

    # (e3) sigma_min(M(s*), equilibrated) at the left-localized dense
    # eigenvalues of every Kreiss-stable/dense-unstable contradiction:
    # ~0 means the determinant criterion agrees with dense and the
    # witness SEARCH missed the zero (search gap, not criterion defect).
    e3 = {}
    key_to_psi = {f"{p:.6g}": p for p in grid}  # full-precision lookup
    for label, contras in RESULTS["c2_contradictions"].items():
        w4 = W4_CASES[label]
        alpha = {"alpha_0": ALPHA0, "alpha_1": w4}
        for pkey, kind, _mre in contras:
            if kind != "kreiss_stable_but_dense_unstable":
                continue
            rows, ridx = ev4.rows(key_to_psi[pkey], alpha)
            sig_at = {}
            for ms in loc[label][pkey]["left_modes"]:
                s_val = complex(ms)
                sig, status = min_singular_value_ex(
                    iw4, io4, rows, s_val,
                    row_indices=[int(i) for i in ridx], equilibrate=True,
                )
                sig_at[ms] = (float(sig), status)
            e3[f"{label} @ psi={pkey}"] = sig_at
            log(f"   (e3) {label} @ psi={pkey}: sigma_min at left dense "
                f"modes: {sig_at}")
    RESULTS["e3_sigma_at_contradiction_modes"] = e3
    dump()


def main():
    t_start = time.perf_counter()
    ev2, alpha2, sweep = part_a()
    part_c1(ev2, alpha2, sweep)
    cc, ev4 = part_b_derive()
    part_b_sweeps(ev4)
    part_c2(ev4)
    part_e(ev4)
    part_d(cc, ev4)
    RESULTS["total_elapsed_s"] = time.perf_counter() - t_start
    dump()
    log(f"DONE in {RESULTS['total_elapsed_s']:.1f} s; results at {OUT}")


if __name__ == "__main__":
    main()
