"""Brady-Livescu 2D stability sweep subcommand.

Drives :func:`stencil_gen.brady2d_stability.brady2d_stability_score` over a
scalar parameter range (sigma for tension, epsilon for gaussian/multiquadric)
and prints a per-layer markdown results table.  With ``--validate-with-cpp``,
re-runs the top survivors at ``max_layer=8`` to validate end-to-end against
the compiled shoccs binary via the Python → C++ bridge.

Usage:
    uv run python -m sweeps brady2d --scheme E4 --kernel tension --param-range 2 4 3 --max-layer 3
    uv run python -m sweeps brady2d --scheme E4 --kernel gaussian --param-range 0.5 1.5 5 --validate-with-cpp
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from stencil_gen.brady2d_stability import (
    L8_FINAL_LINF_TOL,
    StabilityReport,
    brady2d_stability_score,
)

from ._common import load_known_values, save_known_values

# Classical E4 alpha pair matching scripts/stencil_gen/tests/test_cpp_bridge.py
# and the E4u_1.t.cpp reference values.  Used when --kernel=classical so the
# sweep can run a single-point check without requiring --param-range.
CLASSICAL_E4_ALPHA = [-0.7733323791884821, 0.1623961700641681]

# How many passing sweep points to re-run at max_layer=8 when
# --validate-with-cpp is set.
TOP_K_FOR_L8 = 3


@dataclass
class SweepPoint:
    """One row of the brady2d sweep table."""

    param: float
    params_dict: dict
    report: StabilityReport

    def verdict(self) -> str:
        return self.report.overall_verdict

    def failed_layer(self) -> int | None:
        return self.report.failed_layer

    def layer1_gv_err(self) -> float:
        d = self.report.layer1
        return float(d["boundary_gv_err"]) if d is not None else float("nan")

    def layer3_max_stab_eig(self) -> float:
        d = self.report.layer3
        return float(d["max_stab_eig"]) if d is not None else float("nan")

    def layer6_tgb(self) -> float:
        d = self.report.layer6
        return float(d.get("transient_growth_bound", float("nan"))) if d is not None else float("nan")

    def layer7_msa(self) -> float:
        d = self.report.layer7
        return float(d["max_spectral_abscissa"]) if d is not None else float("nan")

    def layer8_linf(self) -> float:
        d = self.report.layer8
        return float(d["final_linf"]) if d is not None else float("nan")


def _param_name(kernel: str) -> str:
    if kernel == "tension":
        return "sigma"
    if kernel in ("gaussian", "multiquadric"):
        return "epsilon"
    return "param"


def _params_for(kernel: str, value: float) -> dict:
    """Build the params dict passed to brady2d_stability_score."""
    if kernel == "classical":
        return {"alpha": list(CLASSICAL_E4_ALPHA)}
    if kernel == "tension":
        return {"sigma": float(value)}
    if kernel in ("gaussian", "multiquadric"):
        return {"epsilon": float(value)}
    raise ValueError(f"unknown kernel: {kernel}")


def _build_param_values(kernel: str, param_range: tuple[float, float, int] | None) -> np.ndarray:
    """Resolve the scalar parameter grid for the sweep."""
    if kernel == "classical":
        # Classical has a vector alpha, not a scalar — sweep degenerates to a
        # single point using CLASSICAL_E4_ALPHA.  We return a length-1 array
        # with nan so the printer records 'classical' once.
        return np.array([float("nan")])
    if param_range is None:
        raise ValueError(
            f"--param-range lo hi n is required for kernel={kernel}"
        )
    lo, hi, n = param_range
    if n < 1:
        raise ValueError(f"--param-range n must be >= 1, got {n}")
    if n == 1:
        return np.array([float(lo)])
    return np.linspace(float(lo), float(hi), int(n))


def _fmt(x: float, *, fmt: str = "{:.4e}") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return fmt.format(x)


def print_markdown_table(
    points: list[SweepPoint],
    *,
    scheme: str,
    kernel: str,
    max_layer: int,
    param_name: str,
) -> None:
    """Print a markdown table summarising each sweep point."""
    print(f"\n## brady2d sweep — scheme={scheme}, kernel={kernel}, max_layer={max_layer}")
    print()

    cols: list[tuple[str, str]] = [(param_name, ">10s")]
    if max_layer >= 1:
        cols.append(("L1_gv_err", ">12s"))
    if max_layer >= 3:
        cols.append(("L3_max_se", ">12s"))
    if max_layer >= 6:
        cols.append(("L6_tgb", ">10s"))
    if max_layer >= 7:
        cols.append(("L7_max_sa", ">12s"))
    if max_layer >= 8:
        cols.append(("L8_linf", ">12s"))
    cols.append(("verdict", ">8s"))
    cols.append(("failed_at", ">10s"))

    header = "| " + " | ".join(h for h, _ in cols) + " |"
    sep = "|" + "|".join("-" * (len(h) + 2) for h, _ in cols) + "|"
    print(header)
    print(sep)

    points_sorted = sorted(points, key=lambda p: (math.inf if math.isnan(p.param) else p.param))
    for p in points_sorted:
        row: list[str] = []
        row.append(_fmt(p.param, fmt="{:.4f}"))
        if max_layer >= 1:
            row.append(_fmt(p.layer1_gv_err()))
        if max_layer >= 3:
            row.append(_fmt(p.layer3_max_stab_eig()))
        if max_layer >= 6:
            row.append(_fmt(p.layer6_tgb(), fmt="{:.2f}"))
        if max_layer >= 7:
            row.append(_fmt(p.layer7_msa()))
        if max_layer >= 8:
            row.append(_fmt(p.layer8_linf()))
        row.append(p.verdict())
        fl = p.failed_layer()
        row.append(str(fl) if fl is not None else "—")
        print("| " + " | ".join(row) + " |")
    print()


def rank_for_l8(points: list[SweepPoint], *, max_layer: int) -> list[SweepPoint]:
    """Return top-K passing points ordered by a stability ranking metric.

    Ranking preference (first available wins):
      1. layer6.transient_growth_bound ascending (if max_layer >= 6)
      2. layer3.max_stab_eig ascending (more-negative = more-stable)
      3. natural order
    """
    passing = [p for p in points if p.verdict() == "pass"]
    if not passing:
        return []

    if max_layer >= 6 and all(p.report.layer6 is not None for p in passing):
        key = lambda p: p.layer6_tgb()  # noqa: E731
    elif max_layer >= 3 and all(p.report.layer3 is not None for p in passing):
        key = lambda p: p.layer3_max_stab_eig()  # noqa: E731
    else:
        warnings.warn(
            f"rank_for_l8: max_layer={max_layer} too shallow (or layer3 reports missing) "
            "for meaningful ranking; returning insertion order. Run with max_layer >= 3 "
            "and ensure layer3 is populated for stability-ordered selection.",
            UserWarning,
            stacklevel=2,
        )
        key = lambda p: 0.0  # noqa: E731

    return sorted(passing, key=key)[:TOP_K_FOR_L8]


def _report_to_dict(report: StabilityReport) -> dict[str, Any]:
    """Serialise a StabilityReport to a JSON-friendly dict for known_values."""
    out: dict[str, Any] = {
        "overall_verdict": report.overall_verdict,
        "failed_layer": report.failed_layer,
        "failed_reason": report.failed_reason,
        "compute_time": float(report.compute_time),
    }
    if report.layer1 is not None:
        out["layer1"] = {k: float(v) for k, v in report.layer1.items() if isinstance(v, (int, float))}
    if report.layer2 is not None:
        out["layer2"] = {"is_stable": bool(report.layer2.is_stable)}
    if report.layer3 is not None:
        out["layer3"] = {"max_stab_eig": float(report.layer3["max_stab_eig"])}
    if report.layer_bl42 is not None:
        out["layer_bl42"] = {
            k: float(v)
            for k, v in report.layer_bl42.items()
            if isinstance(v, (int, float))
        }
    if report.layer4 is not None:
        out["layer4"] = {"max_local_gv_error": float(report.layer4["max_local_gv_error"])}
    if report.layer5 is not None:
        out["layer5"] = {"max_aligned_error": float(report.layer5["max_aligned_error"])}
    if report.layer6 is not None:
        out["layer6"] = {
            "spectral_abscissa": float(report.layer6.get("spectral_abscissa", float("nan"))),
            "transient_growth_bound": float(report.layer6.get("transient_growth_bound", float("nan"))),
        }
    if report.layer7 is not None:
        out["layer7"] = {
            "max_spectral_abscissa": float(report.layer7["max_spectral_abscissa"]),
        }
    if report.layer8 is not None:
        out["layer8"] = {
            "final_linf": float(report.layer8["final_linf"]),
            "stable": bool(report.layer8["stable"]),
            "wall_time_s": float(report.layer8.get("wall_time_s", float("nan"))),
        }
    if report.non_normality is not None:
        out["non_normality"] = dataclasses.asdict(report.non_normality)
    if report.kreiss is not None:
        kr_dict = dataclasses.asdict(report.kreiss)
        kr_dict.pop("sigma_min_field", None)
        kr_dict.pop("s_grid", None)
        out["kreiss"] = kr_dict
    return out


def run_brady2d_sweep(
    *,
    scheme: str,
    kernel: str,
    param_range: tuple[float, float, int] | None,
    max_layer: int,
    validate_with_cpp: bool,
    layer8_N: int = 21,
    layer8_t_final: float = 1.0,
) -> list[SweepPoint]:
    """Execute the sweep and return the list of sweep points."""
    param_values = _build_param_values(kernel, param_range)
    param_name = _param_name(kernel)

    points: list[SweepPoint] = []
    for value in param_values:
        params = _params_for(kernel, float(value) if not math.isnan(float(value)) else 0.0)
        print(f"[brady2d] {param_name}={value:.4f}  scheme={scheme}  kernel={kernel}  max_layer={max_layer}")
        report = brady2d_stability_score(
            scheme=scheme,
            kernel=kernel,
            params=params,
            max_layer=max_layer,
        )
        points.append(SweepPoint(param=float(value), params_dict=params, report=report))

    if validate_with_cpp and max_layer < 8:
        winners = rank_for_l8(points, max_layer=max_layer)
        if not winners:
            print("\n[brady2d] --validate-with-cpp: no passing points to re-run at L8")
        else:
            print(f"\n[brady2d] --validate-with-cpp: re-running {len(winners)} top survivor(s) at max_layer=8")
            for w in winners:
                print(
                    f"  re-run {param_name}={w.param:.4f}  "
                    f"(L6 tgb={w.layer6_tgb():.2f}, L3 max_se={w.layer3_max_stab_eig():.4e})"
                )
                w.report = brady2d_stability_score(
                    scheme=scheme,
                    kernel=kernel,
                    params=w.params_dict,
                    max_layer=8,
                    layer8_N=layer8_N,
                    layer8_t_final=layer8_t_final,
                )

    return points


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.brady2d_sweep",
        description="Brady-Livescu 2D layered stability sweep",
    )
    parser.add_argument("--scheme", choices=["E2", "E4"], required=True)
    parser.add_argument(
        "--kernel",
        choices=["classical", "tension", "gaussian", "multiquadric"],
        required=True,
    )
    parser.add_argument(
        "--param-range",
        nargs=3,
        metavar=("LO", "HI", "N"),
        default=None,
        help=(
            "Three values: low, high, and number of sweep points. "
            "Sweeps sigma for tension, epsilon for gaussian/multiquadric. "
            "Ignored when --kernel=classical."
        ),
    )
    parser.add_argument(
        "--max-layer", type=int, default=3, choices=list(range(1, 9)),
        help="Highest stability layer to run (1-8). Default: 3.",
    )
    parser.add_argument(
        "--validate-with-cpp", action="store_true",
        help=(
            f"Re-run the top-{TOP_K_FOR_L8} passing points at max_layer=8, "
            "invoking the compiled shoccs binary via the Python → C++ bridge."
        ),
    )
    parser.add_argument(
        "--layer8-N", type=int, default=21,
        help="Grid resolution forwarded to layer 8 (default: 21, fast).",
    )
    parser.add_argument(
        "--layer8-t-final", type=float, default=1.0,
        help="Simulation end time forwarded to layer 8 (default: 1.0, fast).",
    )
    parser.add_argument(
        "--update-known-values", action="store_true",
        help='Persist results to known_values.json["brady2d_sweep"][scheme][kernel].',
    )

    args = parser.parse_args(argv)

    param_range: tuple[float, float, int] | None = None
    if args.param_range is not None:
        lo, hi, n = args.param_range
        param_range = (float(lo), float(hi), int(n))

    try:
        points = run_brady2d_sweep(
            scheme=args.scheme,
            kernel=args.kernel,
            param_range=param_range,
            max_layer=args.max_layer,
            validate_with_cpp=args.validate_with_cpp,
            layer8_N=args.layer8_N,
            layer8_t_final=args.layer8_t_final,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    effective_max_layer = 8 if args.validate_with_cpp else args.max_layer
    print_markdown_table(
        points,
        scheme=args.scheme,
        kernel=args.kernel,
        max_layer=effective_max_layer,
        param_name=_param_name(args.kernel),
    )

    # Summary line
    n_pass = sum(1 for p in points if p.verdict() == "pass")
    print(f"[brady2d] {n_pass}/{len(points)} point(s) passed (max_layer={effective_max_layer})")
    if args.validate_with_cpp:
        n_l8_pass = sum(
            1 for p in points
            if p.report.layer8 is not None and bool(p.report.layer8["stable"])
            and float(p.report.layer8["final_linf"]) <= L8_FINAL_LINF_TOL
        )
        n_l8_total = sum(1 for p in points if p.report.layer8 is not None)
        print(f"[brady2d] L8 validation: {n_l8_pass}/{n_l8_total} C++ run(s) stable")

    if args.update_known_values:
        kv = load_known_values()
        sweep_root = kv.setdefault("brady2d_sweep", {})
        scheme_bucket = sweep_root.setdefault(args.scheme, {})
        kernel_bucket: dict[str, Any] = {
            "scheme": args.scheme,
            "kernel": args.kernel,
            "max_layer": effective_max_layer,
            "param_name": _param_name(args.kernel),
            "points": [
                {
                    "param": p.param,
                    "params_dict": p.params_dict,
                    "report": _report_to_dict(p.report),
                }
                for p in points
            ],
        }
        scheme_bucket[args.kernel] = kernel_bucket
        save_known_values(kv)
        print(
            f'\n[brady2d] Updated known_values.json: brady2d_sweep.{args.scheme}.{args.kernel} '
            f"({len(points)} point(s))"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
