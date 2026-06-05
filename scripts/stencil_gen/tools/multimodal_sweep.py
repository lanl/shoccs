#!/usr/bin/env python3
"""E4-classical α multi-modal sweep harness for plan 47.6b.3.2a / 47.6b.3.2c.4.

Runs ``run_mfbo`` against the real cascade objective on E4-classical α
(``brady2d_stability_score(scheme="E4", kernel="classical", ..., max_layer=3)``)
across a user-supplied list of seeds, recording per-seed basin-proximity
data for the BL and DE basins from ``scientific_findings.md`` finding #2.
Prints a Markdown table suitable for pasting into the parent plan item's
Done note.

Mirrors :file:`tools/branin_sweep.py` (47.3k.4a) in shape but exercises
the *real* cascade rather than ``AugmentedBranin``: ``run_mfbo`` is
invoked WITHOUT ``objective=`` injection so the actual
``make_multi_fidelity_objective`` factory + ``brady2d_stability_score``
pipeline runs at every eval.

Configuration matches ``TestMultiModal::test_classical_alpha_finds_a_basin``
verbatim except for ``budget_evals``, which is exposed as a CLI argument
so 47.6b.3.2a can sweep ``budget_evals=60`` (path (a) from the
47.6b.3.1 empirical-fallback list) for the basin-proximity gap closure.

Per 47.6b.3.2c.4a, three additional flags forward
``recommendation_strategy`` / ``voronoi_radius`` / ``ucb_beta`` through to
``run_mfbo`` so 47.6b.3.2c.4b/c can sweep the ``"voronoi"`` and ``"ucb"``
feasibility-aware recommendation strategies under the same harness.

Usage
-----
::

    python tools/multimodal_sweep.py --seeds 0 1 2 --budget-evals 60
    python tools/multimodal_sweep.py --seeds 3 4 --budget-evals 60
    python tools/multimodal_sweep.py --seeds 0 1 2 --budget-evals 60 \\
        --recommendation-strategy voronoi --voronoi-radius 0.1
    python tools/multimodal_sweep.py --seeds 0 1 2 --budget-evals 60 \\
        --recommendation-strategy ucb --ucb-beta 2.0
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch

from stencil_gen.bo import DEFAULT_COST_TABLE, BOResult, run_mfbo

# Known basins from scientific_findings.md finding #2.
_BL = np.array([-0.7733, 0.1624])
_DE = np.array([-1.399, 0.293])

_HF_LEVEL = 3
_BOUNDS = [(-2.0, 2.0), (0.05, 2.0)]
_REPORT_FIELDS = {1: "layer1.boundary_gv_err", 3: "layer3.max_stab_eig"}
_COST_TABLE = {1: DEFAULT_COST_TABLE[1], 3: DEFAULT_COST_TABLE[3]}
_N_INIT = 8
_HF_ANCHORS = 4
_WARMUP = 3
_BETA = 0.5  # adaptive_hf_explore_bias
_STATIC_BIAS = 0.0  # hf_explore_bias
_BONUS = 2.0  # hf_acquisition_bonus


def _count_hf_evals(result: BOResult) -> int:
    """Count BOEval entries at the HF fidelity in the result's eval history."""
    return sum(1 for ev in result.eval_history if ev.fidelity == _HF_LEVEL)


def _run_one_seed(
    seed: int,
    budget_evals: int,
    recommendation_strategy: str,
    voronoi_radius: float,
    ucb_beta: float,
) -> dict[str, Any]:
    """Execute one ``run_mfbo`` call; return a dict with the per-row fields.

    On any exception, returns ``{"seed": seed, "error": "<type>: <msg>"}``
    so the harness can continue with the next seed.
    """
    try:
        # Set global RNGs in addition to ``run_mfbo``'s internal seeding so
        # any out-of-band torch draws inside BoTorch internals are also
        # deterministic across reruns.
        torch.manual_seed(seed)
        np.random.seed(seed)

        t0 = time.perf_counter()
        result = run_mfbo(
            scheme="E4",
            kernel="classical",
            report_fields_by_layer=_REPORT_FIELDS,
            bounds=_BOUNDS,
            cost_table=_COST_TABLE,
            seed=seed,
            n_init=_N_INIT,
            hf_anchors=_HF_ANCHORS,
            budget_evals=budget_evals,
            hf_priority_warmup=_WARMUP,
            adaptive_hf_explore_bias=_BETA,
            hf_explore_bias=_STATIC_BIAS,
            hf_acquisition_bonus=_BONUS,
            recommendation_strategy=recommendation_strategy,
            voronoi_radius=voronoi_radius,
            ucb_beta=ucb_beta,
        )
        elapsed = time.perf_counter() - t0

        d_bl = float(np.linalg.norm(result.best_x - _BL))
        d_de = float(np.linalg.norm(result.best_x - _DE))
        d_min = min(d_bl, d_de)

        return {
            "seed": seed,
            "best_x": (float(result.best_x[0]), float(result.best_x[1])),
            "best_obj": float(result.best_objective),
            "d_BL": d_bl,
            "d_DE": d_de,
            "found": d_min < 0.1,
            "stop_reason": result.stop_reason,
            "n_HF": _count_hf_evals(result),
            "elapsed_s": elapsed,
        }
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}


def _format_row(row: dict[str, Any]) -> str:
    """Format a result dict as a Markdown table row."""
    seed = row["seed"]
    if "error" in row:
        return f"| {seed} | error | {row['error']} | - | - | - | - | - |"
    bx = row["best_x"]
    return (
        f"| {seed} "
        f"| ({bx[0]:.3f}, {bx[1]:.3f}) "
        f"| {row['best_obj']:.4g} "
        f"| {row['d_BL']:.3f} "
        f"| {row['d_DE']:.3f} "
        f"| {str(row['found'])} "
        f"| {row['stop_reason']} "
        f"| {row['n_HF']} "
        f"| {row['elapsed_s']:.1f} |"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "E4-classical α multi-modal sweep harness for plan 47.6b.3.2a"
            " / 47.6b.3.2c.4 empirical measurement."
        )
    )
    parser.add_argument(
        "--seeds",
        required=True,
        type=int,
        nargs="+",
        help="One or more integer seeds to sweep.",
    )
    parser.add_argument(
        "--budget-evals",
        type=int,
        default=60,
        help=(
            "run_mfbo budget_evals override (default 60 for 47.6b.3.2a;"
            " 30 reproduces the 47.6b.3.1 baseline)."
        ),
    )
    parser.add_argument(
        "--recommendation-strategy",
        choices=["mean", "voronoi", "ucb"],
        default="mean",
        help=(
            "run_mfbo recommendation_strategy override (default 'mean'"
            " reproduces the pre-47.6b.3.2c behaviour; 'voronoi' /"
            " 'ucb' enable the feasibility-aware recommendations from"
            " 47.6b.3.2c.2 / 47.6b.3.2c.3)."
        ),
    )
    parser.add_argument(
        "--voronoi-radius",
        type=float,
        default=0.1,
        help=(
            "L2 mask radius for the 'voronoi' strategy (default 0.1"
            " matches run_mfbo's default; ignored when strategy != 'voronoi')."
        ),
    )
    parser.add_argument(
        "--ucb-beta",
        type=float,
        default=2.0,
        help=(
            "UCB confidence multiplier for the 'ucb' strategy (default 2.0"
            " matches run_mfbo's default; ignored when strategy != 'ucb')."
        ),
    )
    args = parser.parse_args(argv)

    print(
        "| seed | best_x | best_obj | d_BL | d_DE | found |"
        " stop_reason | n_HF | elapsed_s |"
    )
    print(
        "|-----:|:-------|---------:|-----:|-----:|:-----:|"
        ":-----------:|-----:|----------:|"
    )
    for seed in args.seeds:
        row = _run_one_seed(
            seed,
            args.budget_evals,
            args.recommendation_strategy,
            args.voronoi_radius,
            args.ucb_beta,
        )
        print(_format_row(row), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
