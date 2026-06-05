"""CLI entry point for Brady-Livescu 2D stability scoring.

Usage:
    # Score a single scheme/kernel combination
    uv run python -m stencil_gen.brady2d_cli --scheme E4 --kernel tension --sigma 3.0 --max-layer 3
    uv run python -m stencil_gen.brady2d_cli --scheme E4 --kernel classical --alpha 0.3487 --max-layer 6
    uv run python -m stencil_gen.brady2d_cli --scheme E4 --kernel tension --sigma 3.0 --max-layer 7 --json-output result.json

    # Run calibration across all families
    uv run python -m stencil_gen.brady2d_cli --run-calibration --max-layer 3
    uv run python -m stencil_gen.brady2d_cli --run-calibration --max-layer 6 --update-known-values

    # Multi-seed E4 classical-α basin survey (plan 43.9c)
    uv run python -m stencil_gen.brady2d_cli --alpha-basin-survey --n-seeds 20
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict

import numpy as np


def _json_serializer(obj):
    """Custom JSON serializer for numpy types and dataclass fields."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _build_params(args: argparse.Namespace) -> dict:
    """Build the params dict from CLI arguments."""
    params: dict = {}
    if args.sigma is not None:
        params["sigma"] = args.sigma
    if args.epsilon is not None:
        params["epsilon"] = args.epsilon
    if args.alpha is not None:
        params["alpha"] = [float(a) for a in args.alpha.split(",")]
    return params


_KNOWN_VALUES_PATH = pathlib.Path(__file__).resolve().parent.parent / "sweeps" / "known_values.json"


def _run_calibration_mode(args: argparse.Namespace) -> int:
    """Run calibration across all families and optionally update known_values.json."""
    from stencil_gen.benchmarks.brady2d_calibration import (
        format_calibration_table,
        run_calibration,
    )

    results = run_calibration(
        max_layer=args.max_layer,
        short_circuit=args.short_circuit,
    )

    print(format_calibration_table(results))

    if args.update_known_values:
        # Load existing known_values.json, add/overwrite the brady2d_calibration key
        if _KNOWN_VALUES_PATH.exists():
            with open(_KNOWN_VALUES_PATH) as f:
                known = json.load(f)
        else:
            known = {}

        known["brady2d_calibration"] = results

        with open(_KNOWN_VALUES_PATH, "w") as f:
            json.dump(known, f, indent=2)
            f.write("\n")

        print(f"\nCalibration results written to: {_KNOWN_VALUES_PATH}")

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2, default=_json_serializer)
        print(f"\nJSON output written to: {args.json_output}")

    # Return 0 if all families passed, 1 if any failed
    any_fail = any(r.get("overall_verdict") != "pass" for r in results.values())
    return 1 if any_fail else 0


def _run_alpha_basin_survey_mode(args: argparse.Namespace) -> int:
    """Run the E4 classical-α multi-seed basin survey (plan 43.9c)."""
    from stencil_gen.benchmarks.alpha_basin_survey import (
        format_survey_table,
        run_survey,
    )

    kwargs: dict = {"n_seeds": args.n_seeds, "base_seed": args.base_seed}
    if args.n_restarts is not None:
        kwargs["n_restarts"] = args.n_restarts
    if args.survey_max_evals is not None:
        kwargs["max_evals"] = args.survey_max_evals
    survey = run_survey(**kwargs)

    print(format_survey_table(survey))

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(survey, f, indent=2, default=_json_serializer)
        print(f"\nJSON output written to: {args.json_output}")

    return 0 if survey["n_feasible_seeds"] > 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="brady2d_cli",
        description="Brady-Livescu 2D analytical stability scoring pipeline",
    )
    parser.add_argument(
        "--scheme",
        choices=["E2", "E4"],
        default=None,
        help="Scheme name (required unless --run-calibration)",
    )
    parser.add_argument(
        "--kernel",
        choices=["classical", "tension", "gaussian", "multiquadric", "phs"],
        default=None,
        help="Kernel type (required unless --run-calibration)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Tension/PHS sigma parameter",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Gaussian/multiquadric epsilon parameter",
    )
    parser.add_argument(
        "--alpha",
        type=str,
        default=None,
        help="Classical alpha values (comma-separated, e.g. '0.3487' or '0.1,0.2')",
    )
    parser.add_argument(
        "--max-layer",
        type=int,
        default=7,
        help="Highest layer to run (1-7, default: 7)",
    )
    parser.add_argument(
        "--short-circuit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop at first failing layer (default: --short-circuit)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Path to write JSON output",
    )
    parser.add_argument(
        "--run-calibration",
        action="store_true",
        default=False,
        help="Run calibration across all families instead of scoring a single scheme",
    )
    parser.add_argument(
        "--update-known-values",
        action="store_true",
        default=False,
        help="Write calibration results to known_values.json under 'brady2d_calibration' key",
    )
    parser.add_argument(
        "--alpha-basin-survey",
        action="store_true",
        default=False,
        help="Run multi-seed E4 classical-α basin survey (plan 43.9c) instead of scoring",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="Number of Sobol seeds for --alpha-basin-survey (default: 20)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Starting seed offset for --alpha-basin-survey (default: 0)",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=None,
        help="Override inner multi-start restarts for --alpha-basin-survey",
    )
    parser.add_argument(
        "--survey-max-evals",
        type=int,
        default=None,
        help="Override per-restart max objective evaluations for --alpha-basin-survey",
    )

    args = parser.parse_args(argv)

    if args.alpha_basin_survey:
        return _run_alpha_basin_survey_mode(args)

    if args.run_calibration:
        return _run_calibration_mode(args)

    # --- Single-scheme scoring mode ---
    if args.scheme is None:
        print("Error: --scheme is required (unless using --run-calibration)", file=sys.stderr)
        return 1
    if args.kernel is None:
        print("Error: --kernel is required (unless using --run-calibration)", file=sys.stderr)
        return 1

    params = _build_params(args)

    # Validate that at least one parameter is provided for non-classical kernels
    if args.kernel == "classical" and args.alpha is None:
        print("Error: --alpha is required for classical kernel", file=sys.stderr)
        return 1
    if args.kernel in ("tension", "phs") and args.sigma is None:
        print(
            f"Error: --sigma is required for {args.kernel} kernel",
            file=sys.stderr,
        )
        return 1
    if args.kernel in ("gaussian", "multiquadric") and args.epsilon is None:
        print(
            f"Error: --epsilon is required for {args.kernel} kernel",
            file=sys.stderr,
        )
        return 1

    # Lazy import to avoid loading numpy/scipy at parse time
    from stencil_gen.brady2d_stability import brady2d_stability_score

    report = brady2d_stability_score(
        scheme=args.scheme,
        kernel=args.kernel,
        params=params,
        max_layer=args.max_layer,
        short_circuit=args.short_circuit,
    )

    print(report)

    if args.json_output:
        result = asdict(report)
        with open(args.json_output, "w") as f:
            json.dump(result, f, indent=2, default=_json_serializer)
        print(f"\nJSON output written to: {args.json_output}")

    return 0 if report.overall_verdict == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
