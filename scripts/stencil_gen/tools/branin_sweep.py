#!/usr/bin/env python3
"""AugmentedBranin sweep harness for plan 47.3k.4b/c/d empirical measurement.

Runs ``run_mfbo`` against BoTorch's :class:`AugmentedBranin` two-fidelity test
function under one of three named kwarg presets and prints a Markdown table
suitable for pasting into the parent plan item's Done note.

The cascade is bypassed: ``run_mfbo`` accepts an ``objective=`` injection
hook that takes ``(x, m) -> (value, wall_time, report)`` and we wire it
directly to ``AugmentedBranin``.

Configurations
--------------
* ``47.3j-equiv``: pre-47.3k defaults restored explicitly so the loop's
  behaviour matches the 47.3j empirical-fallback table verbatim
  (``min_acquisition_iterations=5``, ``variance_guard_relative_threshold=1e-3``).
* ``47.3k-default``: the new 47.3k.1+47.3k.2 defaults are used (no overrides);
  isolates the effect of the default changes (bonus still off).
* ``47.3k-bonus``: 47.3k defaults + ``hf_acquisition_bonus=<--bonus>``;
  the recommended composition under test.

All three configurations share the layered HF-coverage knobs
(``hf_priority_warmup=3``, ``adaptive_hf_explore_bias=0.5``,
``hf_explore_bias=0.0``), the literal 47.6a budget
(``n_init=8``, ``hf_anchors=4``, ``budget_evals=30``), the canonical
Branin bounds ``[(-5, 10), (0, 15)]``, and the 100x cost ratio
``{1: 0.01, 7: 1.0}``.

Usage
-----
::

    python tools/branin_sweep.py --config 47.3j-equiv --seeds 0 1 2 3 4
    python tools/branin_sweep.py --config 47.3k-bonus --seeds 0 --bonus 1.0
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
from botorch.test_functions.multi_fidelity import AugmentedBranin

from stencil_gen.bo import BOResult, run_mfbo

# Two-fidelity hook: m=1 -> s=0.5 (cheap), m=7 -> s=1.0 (HF).
_FIDELITY_S = {1: 0.5, 7: 1.0}
_HF_LEVEL = 7
_BRANIN_BOUNDS = [(-5.0, 10.0), (0.0, 15.0)]
_COST_TABLE = {1: 0.01, 7: 1.0}
_REPORT_FIELDS = {1: "branin.lf", 7: "branin.hf"}
_FIDELITY_LEVELS = (1, 7)
_BUDGET_EVALS = 30
_N_INIT = 8
_HF_ANCHORS = 4
_WARMUP = 3
_BETA = 0.5  # adaptive_hf_explore_bias
_STATIC_BIAS = 0.0  # hf_explore_bias

_CONFIG_CHOICES = ("47.3j-equiv", "47.3k-default", "47.3k-bonus")


def _make_branin_objective():
    """Return an ``(x, m) -> (value, wall_time, report)`` closure."""
    bran = AugmentedBranin(negate=False)

    def objective(x: np.ndarray, m: int) -> tuple[float, float, dict]:
        s = _FIDELITY_S[m]
        # AugmentedBranin expects a (..., 3) tensor: [x0, x1, s].
        xs = torch.tensor([[float(x[0]), float(x[1]), s]], dtype=torch.double)
        v = float(bran(xs))
        # Synthetic wall-time placeholder; the real timing is recorded by
        # ``run_mfbo`` via ``time.perf_counter`` around this call, not from
        # this returned scalar (which lands in ``BOEval.wall_time`` only when
        # the loop's own measurement is unavailable — see make_multi_fidelity_
        # objective for the parallel pattern).
        return (v, 0.05, {})

    return objective


def _kwargs_for_config(config: str, bonus: float | None) -> dict[str, Any]:
    """Map a named config to ``run_mfbo`` keyword overrides."""
    shared = {
        "hf_priority_warmup": _WARMUP,
        "adaptive_hf_explore_bias": _BETA,
        "hf_explore_bias": _STATIC_BIAS,
        "n_init": _N_INIT,
        "hf_anchors": _HF_ANCHORS,
        "budget_evals": _BUDGET_EVALS,
    }
    if config == "47.3j-equiv":
        return {
            **shared,
            "min_acquisition_iterations": 5,
            "variance_guard_relative_threshold": 1e-3,
            "hf_acquisition_bonus": None,
        }
    if config == "47.3k-default":
        return {
            **shared,
            "hf_acquisition_bonus": None,
        }
    if config == "47.3k-bonus":
        if bonus is None:
            raise ValueError(
                "--bonus is required when --config 47.3k-bonus"
            )
        return {
            **shared,
            "hf_acquisition_bonus": bonus,
        }
    raise ValueError(f"unknown config: {config!r}")


def _count_hf_evals(result: BOResult) -> int:
    """Count BOEval entries at the HF fidelity in the result's eval history."""
    return sum(1 for ev in result.eval_history if ev.fidelity == _HF_LEVEL)


def _run_one_seed(
    config: str, seed: int, bonus: float | None
) -> dict[str, Any]:
    """Execute one ``run_mfbo`` call; return a dict with the per-row fields.

    On any exception, returns ``{"seed": seed, "error": "<type>: <msg>"}``
    so the harness can continue with the next seed.
    """
    try:
        # Set global RNGs in addition to ``run_mfbo``'s internal seeding
        # so any out-of-band torch draws inside BoTorch internals are also
        # deterministic across reruns.
        torch.manual_seed(seed)
        np.random.seed(seed)

        objective = _make_branin_objective()
        kwargs = _kwargs_for_config(config, bonus)

        t0 = time.perf_counter()
        result = run_mfbo(
            scheme="synthetic",
            kernel="synthetic",
            report_fields_by_layer=_REPORT_FIELDS,
            bounds=_BRANIN_BOUNDS,
            cost_table=_COST_TABLE,
            seed=seed,
            objective=objective,
            **kwargs,
        )
        elapsed = time.perf_counter() - t0

        return {
            "seed": seed,
            "best_obj": result.best_objective,
            "stop_reason": result.stop_reason,
            "n_HF": _count_hf_evals(result),
            "elapsed_s": elapsed,
        }
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        return {
            "seed": seed,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _format_row(row: dict[str, Any]) -> str:
    """Format a result dict as a Markdown table row."""
    seed = row["seed"]
    if "error" in row:
        return f"| {seed} | error | {row['error']} | - | - |"
    return (
        f"| {seed} "
        f"| {row['best_obj']:.4f} "
        f"| {row['stop_reason']} "
        f"| {row['n_HF']} "
        f"| {row['elapsed_s']:.1f} |"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "AugmentedBranin sweep harness for plan 47.3k.4 empirical"
            " measurement."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        choices=_CONFIG_CHOICES,
        help="Named kwarg preset for run_mfbo.",
    )
    parser.add_argument(
        "--seeds",
        required=True,
        type=int,
        nargs="+",
        help="One or more integer seeds to sweep.",
    )
    parser.add_argument(
        "--bonus",
        type=float,
        default=None,
        help="hf_acquisition_bonus value (required for --config 47.3k-bonus).",
    )
    args = parser.parse_args(argv)

    if args.config == "47.3k-bonus" and args.bonus is None:
        parser.error("--bonus is required when --config 47.3k-bonus")

    print("| seed | best_obj | stop_reason | n_HF | elapsed_s |")
    print("|------|---------:|:-----------:|-----:|----------:|")
    for seed in args.seeds:
        row = _run_one_seed(args.config, seed, args.bonus)
        print(_format_row(row), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
