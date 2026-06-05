"""Optimization CLI for Brady-Livescu 2D stability objectives.

Wraps the drivers in :mod:`stencil_gen.optimizer` behind a ``sweeps optimize``
subcommand.  Picks one of Nelder-Mead, COBYQA, SHGO, differential_evolution,
or the staged cheap-inner + expensive-validator pipeline, runs it against a
kernel-specific parameter vector, and prints a summary.

See ``plans/43-stability-optimization-framework.md`` items 43.7 and 43.8 for
the argparse surface and persistence schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from typing import Any

import numpy as np

from stencil_gen.brady2d_stability import (
    L8_FINAL_LINF_TOL,
    brady2d_stability_score,
)
from stencil_gen.cpp_bridge import SHOCCS_BINARY
from stencil_gen.optimizer import (
    DEFAULT_BOUNDS,
    OptimizeResult,
    _infer_max_layer,
    make_objective,
    multi_start_optimize,
    params_from_vector,
    run_scipy_de,
    run_scipy_shgo,
    run_staged_optimize,
)

from ._common import load_known_values, save_known_values

_METHOD_CHOICES = ("Nelder-Mead", "COBYQA", "SHGO", "DE", "staged")
_KERNEL_CHOICES = ("tension", "gaussian", "multiquadric", "classical")
_KERNEL_DIM = {"tension": 1, "gaussian": 1, "multiquadric": 1, "classical": 2}
_CPP_SUPPORTED_KERNELS = ("classical", "tension", "gaussian", "multiquadric")
_CPP_SUPPORTED_SCHEMES = ("E4",)
_CPP_VALIDATION_N_DEFAULT = 31
_CPP_VALIDATION_T_FINAL_DEFAULT = 5.0


def _parse_bounds(raw: list[float] | None) -> list[tuple[float, float]] | None:
    if raw is None:
        return None
    if len(raw) == 0 or len(raw) % 2 != 0:
        raise ValueError(
            f"--bounds expects pairs of values (lo hi [lo hi ...]); got {len(raw)} value(s)"
        )
    return [(float(raw[2 * i]), float(raw[2 * i + 1])) for i in range(len(raw) // 2)]


def _resolve_bounds(
    scheme: str,
    kernel: str,
    raw: list[float] | None,
) -> list[tuple[float, float]]:
    parsed = _parse_bounds(raw)
    if parsed is not None:
        return parsed
    key = (scheme, kernel)
    if key not in DEFAULT_BOUNDS:
        raise ValueError(
            f"no default bounds for scheme={scheme!r}, kernel={kernel!r}; "
            "pass --bounds explicitly"
        )
    return list(DEFAULT_BOUNDS[key])


def _validate_kernel_bounds_dim(
    kernel: str,
    bounds: list[tuple[float, float]],
) -> None:
    """Reject bounds whose pair count disagrees with ``kernel``'s dimensionality.

    Without this gate, a mismatch (e.g. ``--kernel classical --bounds 0.5 20``)
    would feed a 1D Sobol start into a 2D ``params_from_vector``, the exception
    would be swallowed by ``make_objective``'s try/except, and every evaluation
    would silently return ``+inf`` — the CLI would exit 0 with an infeasible
    result instead of flagging the user error.
    """
    expected = _KERNEL_DIM[kernel]
    if len(bounds) != expected:
        raise ValueError(
            f"kernel={kernel!r} expects {expected} bound pair(s); "
            f"got {len(bounds)}"
        )


def _resolve_persisted_layers(
    args: argparse.Namespace,
) -> tuple[int, int, int | None]:
    """Resolve the layer-configuration triple to persist with an optimum.

    Returns ``(gate_layer, max_layer, validator_max_layer)`` where

    - ``gate_layer`` and ``max_layer`` are always ``int`` (never ``None``):
      ``max_layer`` falls back to ``_infer_max_layer(args.objective)`` for
      non-staged methods when ``--max-layer`` is absent, and to the staged
      inner-depth default of ``3`` for ``method == "staged"`` when absent.
      ``gate_layer`` falls back to ``max(max_layer - 1, 0)`` when
      ``--gate-layer`` is absent, matching ``make_objective``'s auto-infer
      behaviour so the persisted record reflects the actual gate the
      optimiser used.
    - ``validator_max_layer`` is ``args.validator_max_layer`` iff
      ``method == "staged"``, else ``None`` (and therefore omitted from the
      persisted dict).

    Raises ``ValueError`` if ``max_layer`` cannot be inferred (e.g. a
    non-layer-prefixed objective without ``--max-layer``).
    """
    if args.method == "staged":
        max_layer = args.max_layer if args.max_layer is not None else 3
        gate_layer = (
            args.gate_layer if args.gate_layer is not None else max(max_layer - 1, 0)
        )
        return int(gate_layer), int(max_layer), int(args.validator_max_layer)
    if args.max_layer is not None:
        max_layer = int(args.max_layer)
    else:
        inferred = _infer_max_layer(args.objective)
        if inferred is None:
            raise ValueError(
                f"cannot infer max_layer from objective={args.objective!r}; "
                "pass --max-layer explicitly"
            )
        max_layer = int(inferred)
    gate_layer = (
        args.gate_layer if args.gate_layer is not None else max(max_layer - 1, 0)
    )
    return int(gate_layer), max_layer, None


def _run_cpp_validation(
    scheme: str,
    kernel: str,
    best_params: dict,
    best_objective: float,
    *,
    N: int = _CPP_VALIDATION_N_DEFAULT,
    t_final: float = _CPP_VALIDATION_T_FINAL_DEFAULT,
) -> dict[str, Any] | None:
    """Re-run the optimizer winner at ``max_layer=8`` via the shoccs bridge.

    Returns a JSON-friendly ``{stable, final_linf, wall_time_s}`` dict, or
    ``None`` when the L8 run is skipped (unsupported scheme/kernel, no
    feasible winner, missing shoccs binary).  Plan 43.10a: a failing L8
    prints a warning and still returns the dict, but the caller does **not**
    alter ``best_objective`` — the analytical verdict stands; L8 disagreement
    is purely diagnostic.
    """
    if not best_params:
        print(
            "\n[optimize] --validate-with-cpp: skipping — optimizer did not "
            "find a feasible winner (best_params is empty)."
        )
        return None
    if not np.isfinite(best_objective):
        print(
            "\n[optimize] --validate-with-cpp: skipping — best_objective is "
            "non-finite; the analytical stack did not produce a feasible winner."
        )
        return None
    if scheme not in _CPP_SUPPORTED_SCHEMES or kernel not in _CPP_SUPPORTED_KERNELS:
        print(
            f"\n[optimize] --validate-with-cpp: skipping — (scheme={scheme}, "
            f"kernel={kernel}) is not wired through the L8 bridge."
        )
        return None
    if not SHOCCS_BINARY.exists():
        print(
            f"\n[optimize] --validate-with-cpp: skipping — shoccs binary not "
            f"found at {SHOCCS_BINARY}. Build it with `cmake --build build`."
        )
        return None

    print(
        f"\n[optimize] --validate-with-cpp: running L8 ({scheme}/{kernel}) "
        f"at N={N}, t_final={t_final}..."
    )
    try:
        report = brady2d_stability_score(
            scheme,
            kernel,
            best_params,
            max_layer=8,
            short_circuit=False,
            layer8_N=N,
            layer8_t_final=t_final,
        )
    except Exception as exc:
        print(
            f"[optimize] L8 raised ({type(exc).__name__}): {exc}; "
            "treating as failure."
        )
        return {"stable": False, "final_linf": float("nan"), "wall_time_s": 0.0}

    if report.layer8 is None:
        print(
            "[optimize] L8 did not populate a report (unexpected with "
            "short_circuit=False); skipping."
        )
        return None

    l8 = report.layer8
    cpp_validation = {
        "stable": bool(l8["stable"]),
        "final_linf": float(l8["final_linf"]),
        "wall_time_s": float(l8["wall_time_s"]),
    }
    passed = cpp_validation["stable"] and (
        cpp_validation["final_linf"] <= L8_FINAL_LINF_TOL
    )
    if passed:
        print(
            f"[optimize] L8 PASS: final_linf={cpp_validation['final_linf']:.4e} "
            f"(wall={cpp_validation['wall_time_s']:.1f}s)"
        )
    elif cpp_validation["stable"]:
        print(
            f"[optimize] WARNING: L8 soft-failure: "
            f"final_linf={cpp_validation['final_linf']:.4e} > "
            f"L8_FINAL_LINF_TOL={L8_FINAL_LINF_TOL} "
            f"(wall={cpp_validation['wall_time_s']:.1f}s). "
            "Analytical best_objective unchanged — L8 disagreement is diagnostic."
        )
    else:
        print(
            f"[optimize] WARNING: L8 FAIL "
            f"(stable=False, final_linf={cpp_validation['final_linf']:.4e}, "
            f"wall={cpp_validation['wall_time_s']:.1f}s). "
            "Analytical best_objective unchanged — L8 disagreement is diagnostic."
        )
    return cpp_validation


def _result_to_persist_dict(
    result: OptimizeResult,
    *,
    scheme: str,
    kernel: str,
    objective: str,
    bounds: list[tuple[float, float]],
    gate_layer: int,
    max_layer: int,
    validator_max_layer: int | None = None,
    cpp_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialise ``result`` to a JSON-friendly dict, dropping ``history``.

    ``gate_layer`` and ``max_layer`` are the resolved feasibility-gate and
    objective-evaluation depths (see :func:`_resolve_persisted_layers`).  They
    are recorded explicitly so that :class:`TestRegressionBrady2DOptima` can
    rebuild ``make_objective`` at the exact configuration the CLI used, rather
    than silently relying on defaults that may drift.  ``validator_max_layer``
    is only present for ``method == "staged"``.

    Plan 43.9b-r2: a short allow-list of scalar diagnostics from
    ``result.extras`` is copied into the persisted entry so downstream
    consumers of ``known_values.json`` can read them without recomputing.
    Currently just ``cpp_cutcell_violates_197_288`` (the E4 classical-alpha
    C++ cut-cell floor flag); keys absent in ``extras`` are not written.
    """
    d: dict[str, Any] = {
        "scheme": scheme,
        "kernel": kernel,
        "objective": objective,
        "bounds": [list(b) for b in bounds],
        "best_x": [float(v) for v in np.asarray(result.best_x, dtype=float).ravel()],
        "best_params": result.best_params,
        "best_objective": float(result.best_objective),
        "method": result.method,
        "gate_layer": int(gate_layer),
        "max_layer": int(max_layer),
        "n_evals": int(result.n_evals),
        "compute_time": float(result.compute_time),
        "converged": bool(result.converged),
        "best_report": result.best_report,
    }
    if validator_max_layer is not None:
        d["validator_max_layer"] = int(validator_max_layer)
    extras = result.extras or {}
    if "cpp_cutcell_violates_197_288" in extras:
        d["cpp_cutcell_violates_197_288"] = bool(
            extras["cpp_cutcell_violates_197_288"]
        )
    if cpp_validation is not None:
        d["cpp_validation"] = {
            "stable": bool(cpp_validation["stable"]),
            "final_linf": float(cpp_validation["final_linf"]),
            "wall_time_s": float(cpp_validation["wall_time_s"]),
        }
    return d


def _print_summary(
    result: OptimizeResult,
    *,
    scheme: str,
    kernel: str,
    objective: str,
    bounds: list[tuple[float, float]],
) -> None:
    print(f"\n{'=' * 64}")
    print(f"  [optimize] scheme={scheme}  kernel={kernel}  method={result.method}")
    print(f"  [optimize] objective={objective}")
    print(f"  [optimize] bounds={bounds}")
    print(f"{'=' * 64}")
    best_x = np.asarray(result.best_x, dtype=float).ravel()
    print(f"  best_x         = {np.array2string(best_x, precision=6)}")
    print(f"  best_params    = {result.best_params}")
    print(f"  best_objective = {result.best_objective:.6e}")
    print(f"  converged      = {result.converged}")
    print(f"  n_evals        = {result.n_evals}")
    print(f"  compute_time   = {result.compute_time:.3f} s")
    extras = result.extras or {}
    for k, v in extras.items():
        if k in ("validator_ranking", "local_minima"):
            try:
                n = len(v)
            except TypeError:
                n = "?"
            print(f"  extras.{k:<20s} (len={n})")
        elif isinstance(v, np.ndarray):
            print(f"  extras.{k:<20s} = {np.array2string(v, precision=6)}")
        else:
            print(f"  extras.{k:<20s} = {v}")


def _run_method(
    args: argparse.Namespace,
    bounds: list[tuple[float, float]],
) -> OptimizeResult:
    method = args.method
    if method == "staged":
        inner_max_layer = args.max_layer if args.max_layer is not None else 3
        # ``make_objective`` auto-infers ``gate_layer = max(max_layer - 1, 0)``
        # when ``None``; ``run_staged_optimize``'s ``inner_gate`` is a plain
        # ``int`` so resolve the default here to keep its signature untouched.
        inner_gate = (
            args.gate_layer
            if args.gate_layer is not None
            else max(inner_max_layer - 1, 0)
        )
        return run_staged_optimize(
            scheme=args.scheme,
            kernel=args.kernel,
            report_field=args.objective,
            bounds=bounds,
            inner_gate=inner_gate,
            inner_max_layer=inner_max_layer,
            validator_max_layer=args.validator_max_layer,
            top_k=args.top_k,
            method=args.inner_method,
            n_restarts=args.n_restarts,
            seed=args.seed,
            max_evals=args.max_evals,
        )

    f = make_objective(
        scheme=args.scheme,
        kernel=args.kernel,
        report_field=args.objective,
        gate_layer=args.gate_layer,
        max_layer=args.max_layer,
    )

    if method in ("Nelder-Mead", "COBYQA"):
        result = multi_start_optimize(
            f,
            bounds=bounds,
            n_restarts=args.n_restarts,
            method=method,
            seed=args.seed,
            max_evals=args.max_evals,
        )
        # multi_start_optimize is kernel-agnostic, so fill in best_params here.
        if np.isfinite(result.best_objective):
            return replace(
                result,
                best_params=params_from_vector(args.kernel, result.best_x),
            )
        return result

    if method == "SHGO":
        result = run_scipy_shgo(f, bounds=bounds, n=args.shgo_n, iters=args.shgo_iters)
        if np.isfinite(result.best_objective):
            return replace(
                result,
                best_params=params_from_vector(args.kernel, result.best_x),
            )
        return result

    if method == "DE":
        result = run_scipy_de(
            f,
            bounds=bounds,
            popsize=args.de_popsize,
            maxiter=args.de_maxiter,
            seed=args.seed,
        )
        if np.isfinite(result.best_objective):
            return replace(
                result,
                best_params=params_from_vector(args.kernel, result.best_x),
            )
        return result

    raise ValueError(f"unknown method: {method}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.optimize",
        description=(
            "Optimize boundary-closure parameters against a Brady-Livescu 2D "
            "stability objective (layered cascade)."
        ),
    )
    parser.add_argument("--scheme", choices=["E2", "E4"], required=True)
    parser.add_argument("--kernel", choices=list(_KERNEL_CHOICES), required=True)
    parser.add_argument(
        "--objective",
        required=True,
        help='Dotted-path report field (e.g. "layer3.max_stab_eig", "layer6.transient_growth_bound").',
    )
    parser.add_argument(
        "--gate-layer",
        type=int,
        default=None,
        help=(
            "Highest layer whose failure forces +inf (the feasibility gate). "
            "Default: max_layer-1 (auto-inferred from --objective; floored at 0)."
        ),
    )
    parser.add_argument(
        "--max-layer",
        type=int,
        default=None,
        help="Highest layer run by the objective (default: inferred from --objective).",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs="+",
        default=None,
        metavar="VAL",
        help="Flat list of bound pairs (lo hi [lo hi ...]).  If absent, falls back to DEFAULT_BOUNDS.",
    )
    parser.add_argument(
        "--method",
        choices=list(_METHOD_CHOICES),
        default="staged",
    )
    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument(
        "--max-evals",
        type=int,
        default=200,
        help="Max objective evaluations per local run (Nelder-Mead / COBYQA / staged inner).",
    )
    parser.add_argument("--seed", type=int, default=0)

    # Staged-specific knobs
    parser.add_argument(
        "--validator-max-layer",
        type=int,
        default=6,
        help="Validator stage max_layer (staged method only; default: 6).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of inner-stage survivors re-evaluated by the validator (staged method only).",
    )
    parser.add_argument(
        "--inner-method",
        choices=["Nelder-Mead", "COBYQA"],
        default="Nelder-Mead",
        help="Local method used inside the staged inner multi-start (default: Nelder-Mead).",
    )

    # SHGO-specific knobs
    parser.add_argument("--shgo-n", type=int, default=100)
    parser.add_argument("--shgo-iters", type=int, default=3)

    # DE-specific knobs
    parser.add_argument("--de-popsize", type=int, default=15)
    parser.add_argument("--de-maxiter", type=int, default=100)

    # Post-run knobs
    parser.add_argument(
        "--validate-with-cpp",
        action="store_true",
        help="Re-run the winner at max_layer=8 via the C++ bridge (plan 43.10a).",
    )
    parser.add_argument(
        "--update-known-values",
        action="store_true",
        help='Persist the result to known_values.json["brady2d_optima"][scheme][kernel][objective].',
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to write the full result as JSON.",
    )

    args = parser.parse_args(argv)

    try:
        bounds = _resolve_bounds(args.scheme, args.kernel, args.bounds)
        _validate_kernel_bounds_dim(args.kernel, bounds)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        result = _run_method(args, bounds=bounds)
    except ValueError as exc:
        parser.error(str(exc))

    _print_summary(
        result,
        scheme=args.scheme,
        kernel=args.kernel,
        objective=args.objective,
        bounds=bounds,
    )

    try:
        gate_layer, max_layer, validator_max_layer = _resolve_persisted_layers(args)
    except ValueError as exc:
        parser.error(str(exc))

    cpp_validation: dict[str, Any] | None = None
    if args.validate_with_cpp:
        cpp_validation = _run_cpp_validation(
            args.scheme,
            args.kernel,
            result.best_params,
            result.best_objective,
        )

    persisted = _result_to_persist_dict(
        result,
        scheme=args.scheme,
        kernel=args.kernel,
        objective=args.objective,
        bounds=bounds,
        gate_layer=gate_layer,
        max_layer=max_layer,
        validator_max_layer=validator_max_layer,
        cpp_validation=cpp_validation,
    )

    if args.update_known_values:
        kv = load_known_values()
        optima_root = kv.setdefault("brady2d_optima", {})
        scheme_bucket = optima_root.setdefault(args.scheme, {})
        kernel_bucket = scheme_bucket.setdefault(args.kernel, {})
        kernel_bucket[args.objective] = persisted
        save_known_values(kv)
        print(
            f'\n[optimize] Updated known_values.json: '
            f"brady2d_optima.{args.scheme}.{args.kernel}.{args.objective}"
        )

    if args.json_output is not None:
        with open(args.json_output, "w") as fp:
            json.dump(persisted, fp, indent=2)
            fp.write("\n")
        print(f"[optimize] Wrote JSON result to {args.json_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
