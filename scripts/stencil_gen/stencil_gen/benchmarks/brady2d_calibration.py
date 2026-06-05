"""Calibration of all stencil families against the Brady-Livescu 2D benchmark.

Runs ``brady2d_stability_score`` on every (scheme, kernel, parameter)
combination currently supported by ``stencil_gen`` and collects per-layer
results.  Plan 42 uses these scores to prioritise which families to port
to C++ first.

Reference: Brady & Livescu 2019 §4.3, pp. 92–94.

Calibration results (max_layer=6, 2026-04-12):
  All E4 families and E2_phs_k2 pass through L6.  Three E2 families fail
  at L1 (boundary group-velocity error > 5%):
    - E2_tension_6:     bndGV=9.75e-02 (sigma=6 pushes boundary dispersion)
    - E2_gaussian_2:    bndGV=2.70e-01 (Gaussian ε=2 poor boundary dispersion)
    - E2_multiquadric_1: bndGV=6.00e-02 (barely over threshold)
  These are expected: E2 is 2nd-order, so boundary closures have larger
  dispersion error.  Only E2_phs_k2 (sigma=0, fully-determined weights)
  passes L1.  Plan 42 should prioritise the six passing families.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any

from stencil_gen.brady2d_stability import brady2d_stability_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Production alpha values for classical boundary closures
# ---------------------------------------------------------------------------
# From src/operators/gradient.t.cpp (E4u_1) — the only classical family
# with free boundary alphas in the E2/E4 scope.
# E2 classical is excluded per 41.5c: E2 has no free alpha parameters.
_E4_CLASSICAL_ALPHA = [-0.7733323791884821, 0.1623961700641681]

# ---------------------------------------------------------------------------
# Family enumeration
# ---------------------------------------------------------------------------
# Each entry: (scheme, kernel, params, display_label)
# The display_label disambiguates stored results (e.g. sigma=0.0 tension
# is really PHS k=2, not actual tension).
#
# NOTE: E2 classical removed per 41.5c — E2 has no free alpha parameters.
# The E2 PHS k=2 entry ("E2", "tension", {"sigma": 0.0}) already covers E2
# with fully-determined boundary weights.
FAMILIES: list[tuple[str, str, dict, str]] = [
    ("E4", "classical", {"alpha": _E4_CLASSICAL_ALPHA}, "E4_classical"),
    # sigma=0 → dispatches to PHS k=2 (see phs.py:407–423)
    ("E2", "tension", {"sigma": 0.0}, "E2_phs_k2"),
    ("E4", "tension", {"sigma": 0.0}, "E4_phs_k2"),
    # actual tension-spline at optimal sigma
    ("E2", "tension", {"sigma": 6.0}, "E2_tension_6"),
    ("E4", "tension", {"sigma": 3.0}, "E4_tension_3"),
    # Gaussian
    ("E2", "gaussian", {"epsilon": 2.0}, "E2_gaussian_2"),
    ("E4", "gaussian", {"epsilon": 0.9}, "E4_gaussian_09"),
    # Multiquadric
    ("E2", "multiquadric", {"epsilon": 1.0}, "E2_multiquadric_1"),
    ("E4", "multiquadric", {"epsilon": 1.0}, "E4_multiquadric_1"),
]


def _report_to_dict(report) -> dict[str, Any]:
    """Extract per-layer scalars from a StabilityReport into a plain dict."""
    d: dict[str, Any] = {
        "overall_verdict": report.overall_verdict,
        "failed_layer": report.failed_layer,
        "failed_reason": report.failed_reason,
        "compute_time": report.compute_time,
    }

    if report.layer1 is not None:
        d["layer1"] = {
            "interior_gv_err_x": report.layer1.get("interior_gv_err_x"),
            "interior_gv_err_y": report.layer1.get("interior_gv_err_y"),
            "boundary_gv_err": report.layer1.get("boundary_gv_err"),
            "cutoff_fraction": report.layer1.get("cutoff_fraction"),
        }

    if report.layer2 is not None:
        d["layer2"] = {"is_stable": report.layer2.is_stable}

    if report.layer3 is not None:
        d["layer3"] = {
            "max_stab_eig": report.layer3.get("max_stab_eig"),
        }

    if report.layer4 is not None:
        d["layer4"] = {
            "max_local_gv_error": report.layer4.get("max_local_gv_error"),
        }

    if report.layer5 is not None:
        d["layer5"] = {
            "max_aligned_error": report.layer5.get("max_aligned_error"),
        }

    if report.layer6 is not None:
        d["layer6"] = {
            "spectral_abscissa": report.layer6.get("spectral_abscissa"),
            "transient_growth_bound": report.layer6.get("transient_growth_bound"),
        }

    if report.layer7 is not None:
        d["layer7"] = {
            "max_spectral_abscissa": report.layer7.get("max_spectral_abscissa"),
        }

    if report.layer8 is not None:
        d["layer8"] = {
            "final_linf": float(report.layer8["final_linf"]),
            "stable": bool(report.layer8["stable"]),
            "wall_time_s": float(report.layer8.get("wall_time_s", float("nan"))),
        }

    if report.layer_bl42 is not None:
        d["layer_bl42"] = {
            "max_spectral_abscissa": report.layer_bl42.get("max_spectral_abscissa"),
            "purely_imaginary": report.layer_bl42.get("purely_imaginary"),
        }

    if report.non_normality is not None:
        d["non_normality"] = dataclasses.asdict(report.non_normality)

    if report.kreiss is not None:
        kr_dict = dataclasses.asdict(report.kreiss)
        kr_dict.pop("sigma_min_field", None)
        kr_dict.pop("s_grid", None)
        d["kreiss"] = kr_dict

    return d


def run_calibration(
    max_layer: int = 7,
    short_circuit: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run ``brady2d_stability_score`` on every family in ``FAMILIES``.

    Parameters
    ----------
    max_layer : int
        Maximum layer to evaluate (1–7).
    short_circuit : bool
        If True, stop evaluating a family as soon as it fails a layer.

    Returns
    -------
    dict[str, dict]
        ``{display_label: {"layer1": {...}, ..., "overall_verdict": ...}}``.
    """
    results: dict[str, dict[str, Any]] = {}
    total_start = time.perf_counter()

    for scheme, kernel, params, label in FAMILIES:
        logger.info("Scoring %s (scheme=%s, kernel=%s, params=%s)", label, scheme, kernel, params)
        try:
            report = brady2d_stability_score(
                scheme, kernel, params,
                max_layer=max_layer,
                short_circuit=short_circuit,
            )
            results[label] = _report_to_dict(report)
        except Exception as exc:
            logger.error("Family %s raised %s: %s", label, type(exc).__name__, exc)
            results[label] = {
                "overall_verdict": "error",
                "failed_layer": None,
                "failed_reason": f"{type(exc).__name__}: {exc}",
                "compute_time": 0.0,
            }

    total_time = time.perf_counter() - total_start
    logger.info("Calibration complete in %.1f s (%d families)", total_time, len(results))
    return results


def format_calibration_table(results: dict[str, dict[str, Any]]) -> str:
    """Format calibration results as a markdown table for display."""
    lines = []
    lines.append("| Family | Verdict | Failed | L1 bndGV | L3 maxEig | L4 localGV | L5 aniso | Time |")
    lines.append("|--------|---------|--------|----------|-----------|------------|----------|------|")

    for label, r in results.items():
        verdict = r.get("overall_verdict", "?")
        failed = str(r.get("failed_layer", "-"))
        l1_gv = r.get("layer1", {}).get("boundary_gv_err")
        l3_eig = r.get("layer3", {}).get("max_stab_eig")
        l4_gv = r.get("layer4", {}).get("max_local_gv_error")
        l5_an = r.get("layer5", {}).get("max_aligned_error")
        ct = r.get("compute_time", 0.0)

        def _fmt(v):
            return f"{v:.2e}" if v is not None else "-"

        lines.append(
            f"| {label:24s} | {verdict:7s} | {failed:6s} "
            f"| {_fmt(l1_gv):8s} | {_fmt(l3_eig):9s} "
            f"| {_fmt(l4_gv):10s} | {_fmt(l5_an):8s} | {ct:5.1f}s |"
        )

    return "\n".join(lines)
