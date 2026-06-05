"""Multi-seed basin survey for E4 classical-α (plan 43.9c).

Analog of Brady & Livescu 2019 Table 4, which reports 101 distinct E4 schemes
discovered across random restarts.  For each Sobol-seed we drive
:func:`run_staged_optimize` on the E4 classical-α feasibility region, record
the validator winner, and cluster winners by rounding ``(α₀, α₁)`` to a fixed
number of decimals.  The returned structure reports how many distinct basins
the survey hit and how many seeds landed in each.

The pipeline uses the widened ``DEFAULT_BOUNDS[("E4","classical")]`` envelope
(plan 43.9a) — the analytical-stability feasibility region, which extends
below the cut-cell-only 197/288 floor.  L8 C++ validation (plan 43.10a)
will diagnose cut-cell violations downstream; this survey treats them as
informational via the ``cpp_cutcell_violates_197_288`` flag propagated from
:func:`run_staged_optimize`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from stencil_gen.optimizer import DEFAULT_BOUNDS, run_staged_optimize

logger = logging.getLogger(__name__)


def _basin_key(best_x: np.ndarray, decimals: int) -> tuple[float, ...]:
    return tuple(float(round(v, decimals)) for v in np.asarray(best_x, dtype=float))


def run_survey(
    n_seeds: int = 20,
    bounds: list[tuple[float, float]] | None = None,
    *,
    scheme: str = "E4",
    kernel: str = "classical",
    report_field: str = "layer6.transient_growth_bound",
    inner_gate: int = 3,
    inner_max_layer: int = 3,
    validator_max_layer: int = 6,
    top_k: int = 5,
    method: str = "Nelder-Mead",
    n_restarts: int = 20,
    max_evals: int = 60,
    base_seed: int = 0,
    cluster_decimals: int = 2,
) -> dict[str, Any]:
    """Run :func:`run_staged_optimize` across ``n_seeds`` seeds and cluster winners.

    Parameters mirror :func:`run_staged_optimize` with two additions:

    * ``base_seed`` — seeds are ``range(base_seed, base_seed + n_seeds)``.
    * ``cluster_decimals`` — basins are identified by rounding ``best_x`` to
      this many decimal places (plan 43.9c specifies 2 decimals).

    Returns
    -------
    dict
        ``{"seed_results": [per_seed_dict, ...], "basins": [basin_dict, ...],
        "n_distinct_basins": int, "n_feasible_seeds": int, "compute_time": float,
        "config": {...}}``.  Each ``basin_dict`` holds
        ``{"alpha": [α₀, α₁], "best_objective": float,
        "n_seeds_in_basin": int, "seeds": [...],
        "cpp_cutcell_violates_197_288": bool}``.
    """
    if n_seeds < 1:
        raise ValueError(f"run_survey: n_seeds must be >= 1, got {n_seeds}")
    if bounds is None:
        bounds = DEFAULT_BOUNDS[(scheme, kernel)]
    if cluster_decimals < 0:
        raise ValueError(f"run_survey: cluster_decimals must be >= 0, got {cluster_decimals}")

    t0 = time.perf_counter()
    seed_results: list[dict[str, Any]] = []
    basins: dict[tuple[float, ...], dict[str, Any]] = {}

    for seed in range(base_seed, base_seed + n_seeds):
        logger.info("basin-survey seed=%d / %d", seed - base_seed + 1, n_seeds)
        result = run_staged_optimize(
            scheme=scheme,
            kernel=kernel,
            report_field=report_field,
            bounds=bounds,
            inner_gate=inner_gate,
            inner_max_layer=inner_max_layer,
            validator_max_layer=validator_max_layer,
            top_k=top_k,
            method=method,
            n_restarts=n_restarts,
            seed=seed,
            max_evals=max_evals,
        )
        feasible = bool(np.isfinite(result.best_objective))
        entry: dict[str, Any] = {
            "seed": seed,
            "best_x": np.asarray(result.best_x, dtype=float).tolist(),
            "best_objective": float(result.best_objective),
            "feasible": feasible,
            "stage": result.extras.get("stage"),
            "n_evals": int(result.n_evals),
            "compute_time": float(result.compute_time),
            "cpp_cutcell_violates_197_288": result.extras.get(
                "cpp_cutcell_violates_197_288"
            ),
        }
        seed_results.append(entry)

        if not feasible:
            continue

        key = _basin_key(result.best_x, cluster_decimals)
        basin = basins.get(key)
        if basin is None:
            basins[key] = {
                "alpha": list(entry["best_x"]),
                "best_objective": entry["best_objective"],
                "n_seeds_in_basin": 1,
                "seeds": [seed],
                "cpp_cutcell_violates_197_288": entry["cpp_cutcell_violates_197_288"],
            }
        else:
            basin["n_seeds_in_basin"] += 1
            basin["seeds"].append(seed)
            if entry["best_objective"] < basin["best_objective"]:
                basin["alpha"] = list(entry["best_x"])
                basin["best_objective"] = entry["best_objective"]
                basin["cpp_cutcell_violates_197_288"] = entry[
                    "cpp_cutcell_violates_197_288"
                ]

    basin_list = sorted(basins.values(), key=lambda b: b["best_objective"])
    compute_time = time.perf_counter() - t0

    return {
        "seed_results": seed_results,
        "basins": basin_list,
        "n_distinct_basins": len(basin_list),
        "n_feasible_seeds": sum(1 for e in seed_results if e["feasible"]),
        "compute_time": compute_time,
        "config": {
            "scheme": scheme,
            "kernel": kernel,
            "report_field": report_field,
            "bounds": [list(b) for b in bounds],
            "n_seeds": n_seeds,
            "base_seed": base_seed,
            "cluster_decimals": cluster_decimals,
            "inner_gate": inner_gate,
            "inner_max_layer": inner_max_layer,
            "validator_max_layer": validator_max_layer,
            "top_k": top_k,
            "method": method,
            "n_restarts": n_restarts,
            "max_evals": max_evals,
        },
    }


def format_survey_table(survey: dict[str, Any]) -> str:
    """Render a basin survey as a markdown table (plan 43.9c)."""
    lines = []
    cfg = survey.get("config", {})
    lines.append(
        f"# Basin survey: {cfg.get('scheme', '?')} {cfg.get('kernel', '?')} "
        f"vs {cfg.get('report_field', '?')}"
    )
    lines.append(
        f"# n_seeds={cfg.get('n_seeds')}, feasible={survey.get('n_feasible_seeds')}, "
        f"distinct basins={survey.get('n_distinct_basins')}, "
        f"wall={survey.get('compute_time', 0.0):.1f}s"
    )
    lines.append("| α₀ | α₁ | best_objective | n_seeds | 197/288 viol |")
    lines.append("|----|----|----------------|---------|--------------|")
    for basin in survey.get("basins", []):
        a = basin["alpha"]
        viol = basin.get("cpp_cutcell_violates_197_288")
        viol_str = "-" if viol is None else ("yes" if viol else "no")
        lines.append(
            f"| {a[0]:+.4f} | {a[1]:+.4f} | {basin['best_objective']:+.4e} "
            f"| {basin['n_seeds_in_basin']:>7d} | {viol_str:>12s} |"
        )
    return "\n".join(lines)
