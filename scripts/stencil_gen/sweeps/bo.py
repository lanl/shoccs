"""Multi-fidelity Bayesian optimization CLI for the stability cascade.

Wraps :func:`stencil_gen.bo.run_mfbo` behind a ``sweeps bo`` subcommand.
Drives a cost-aware qMFKG loop over a configurable subset of the cascade's
layers (cheap surrogates + an HF target) and prints a summary table.

See ``plans/47-mfbo.md`` items 47.4a (this CLI), 47.4c (per-run JSON
persistence), 47.5a (``--validate-with-cpp`` wiring), and 47.5b
(``--baseline staged`` head-to-head against ``run_staged_optimize``) for
each capability.  Items 47.4c and 47.5a/b plug into the stub branches at
the bottom of :func:`main` — they share the parse + dispatch surface
defined here.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

from stencil_gen.brady2d_stability import L8_FINAL_LINF_TOL, brady2d_stability_score
from stencil_gen.bo import (
    DEFAULT_COST_TABLE,
    BOResult,
    run_mfbo,
)
from stencil_gen.cpp_bridge import SHOCCS_BINARY
from stencil_gen.optimizer import (
    DEFAULT_BOUNDS,
    OptimizeResult,
    _infer_max_layer,
    _record_cpp_cutcell_diagnostic,
    run_staged_optimize,
)

from .optimize import _result_to_persist_dict


_KERNEL_CHOICES = ("tension", "gaussian", "multiquadric", "classical")
_KERNEL_DIM = {"tension": 1, "gaussian": 1, "multiquadric": 1, "classical": 2}
_BASELINE_CHOICES = ("none", "staged")
_COST_MODEL_CHOICES = ("constant", "empirical")
_CPP_SUPPORTED_KERNELS = ("classical", "tension", "gaussian", "multiquadric")
_CPP_SUPPORTED_SCHEMES = ("E4",)
_CPP_VALIDATION_N_DEFAULT = 31
_CPP_VALIDATION_T_FINAL_DEFAULT = 5.0

# Default canonical report field for each external fidelity index.  Keys
# match :data:`stencil_gen.bo.DEFAULT_COST_TABLE`; in particular external
# index ``5`` is the synthetic slot for L3r (``layer_bl42``), which sits
# between L3 and L6 in cost.  ``brady2d_stability_score`` populates
# ``layer_bl42`` whenever ``max_layer >= 3``, so passing ``m=5`` to
# :func:`make_multi_fidelity_objective` runs more layers than strictly
# required for ``layer_bl42.max_spectral_abscissa``; we accept that small
# inefficiency in exchange for a unique fidelity index that the ICM
# kernel can identify separately from L3.
_DEFAULT_FIDELITY_FIELDS: dict[int, str] = {
    1: "layer1.boundary_gv_err",
    3: "layer3.max_stab_eig",
    5: "layer_bl42.max_spectral_abscissa",
    6: "layer6.transient_growth_bound",
    7: "layer7.max_spectral_abscissa",
}


def _mangle_objective(field: str) -> str:
    """Encode a single dotted-path objective into a filesystem-safe token.

    Mirrors :func:`sweeps.pareto._mangle_objectives` but for the BO
    subcommand's single-objective case — replaces ``.`` with ``_`` so the
    persisted filename ``<scheme>_<kernel>_<mangled>_<seed>.json`` (47.4c)
    stays readable and unambiguous.
    """
    return field.replace(".", "_")


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
    expected = _KERNEL_DIM[kernel]
    if len(bounds) != expected:
        raise ValueError(
            f"kernel={kernel!r} expects {expected} bound pair(s); "
            f"got {len(bounds)}"
        )


def _parse_fidelity_fields(raw: list[str] | None) -> dict[int, str]:
    """Parse ``--fidelity-fields LAYER=FIELD [...]`` into ``{int: str}``.

    Each item must look like ``5=layer_bl42.max_spectral_abscissa``; the
    layer index parses as ``int``, the field as the literal remainder.
    Empty / ``None`` input returns an empty dict (no overrides).
    """
    if not raw:
        return {}
    out: dict[int, str] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(
                f"--fidelity-fields expects LAYER=FIELD pairs; got {item!r}"
            )
        layer_str, _, field = item.partition("=")
        try:
            layer = int(layer_str)
        except ValueError as exc:
            raise ValueError(
                f"--fidelity-fields layer index must be an int; got {layer_str!r}"
            ) from exc
        if not field:
            raise ValueError(
                f"--fidelity-fields field is empty for LAYER={layer}"
            )
        out[layer] = field
    return out


def _build_report_fields_by_layer(
    objective: str,
    cheap_fidelities: list[int],
    overrides: dict[int, str],
) -> dict[int, str]:
    """Assemble the per-layer report-fields mapping for :func:`run_mfbo`.

    Combines:

    1. The HF entry: ``{hf_layer: objective}`` where ``hf_layer`` is
       inferred from *objective* via :func:`_infer_max_layer`.
    2. One entry per *cheap_fidelity*: pulls the canonical field from
       :data:`_DEFAULT_FIDELITY_FIELDS`, else from *overrides*.
    3. *overrides* always win (lets the caller swap, e.g.,
       ``layer3.max_stab_eig`` for ``layer3.something_else``).

    The cheap entries must all use a layer index strictly less than the
    HF layer — otherwise the user is mis-specifying the "cheap" surrogate
    set.  Raises :class:`ValueError` on any of: unknown HF layer prefix,
    cheap layer ≥ HF layer, missing field for a cheap layer (no default
    and no override), HF override conflicting with --objective.
    """
    hf_layer = _infer_max_layer(objective)
    if hf_layer is None:
        raise ValueError(
            f"cannot infer HF layer from --objective={objective!r}; pass an "
            "objective with a recognised layer prefix (layer1.* … layer8.*) "
            "or a known alias (kreiss.*, layer_bl42.*, non_normality.*)"
        )

    fields: dict[int, str] = {}
    for layer in cheap_fidelities:
        if layer >= hf_layer:
            raise ValueError(
                f"--cheap-fidelities entry {layer} must be strictly less than "
                f"the HF layer {hf_layer} (inferred from --objective)"
            )
        if layer in overrides:
            fields[layer] = overrides[layer]
        elif layer in _DEFAULT_FIDELITY_FIELDS:
            fields[layer] = _DEFAULT_FIDELITY_FIELDS[layer]
        else:
            raise ValueError(
                f"no default report field for --cheap-fidelities entry {layer}; "
                "supply one via --fidelity-fields LAYER=FIELD"
            )

    # HF: --objective always wins over any --fidelity-fields override on the
    # HF slot, since the user explicitly named the HF target.
    if hf_layer in overrides and overrides[hf_layer] != objective:
        raise ValueError(
            f"--fidelity-fields override at HF layer {hf_layer} "
            f"({overrides[hf_layer]!r}) conflicts with --objective={objective!r}"
        )
    fields[hf_layer] = objective
    return fields


def _print_summary(result: BOResult, *, baseline: dict | None = None) -> None:
    print(f"\n{'=' * 72}")
    print(f"  [bo] scheme={result.scheme}  kernel={result.kernel}  method={result.method}")
    print(f"  [bo] objective={result.report_fields_by_layer[result.hf_level]} (HF=L{result.hf_level})")
    print(f"  [bo] bounds={list(result.bounds)}")
    print(f"  [bo] fidelity_levels={list(result.fidelity_levels)}")
    print(f"{'=' * 72}")
    best_x = np.asarray(result.best_x, dtype=float).ravel()
    print(f"  best_x         = {np.array2string(best_x, precision=6)}")
    print(f"  best_params    = {result.best_params}")
    print(f"  best_objective = {result.best_objective:.6e}")
    print(f"  converged      = {result.converged}")
    print(f"  stop_reason    = {result.stop_reason}")
    print(f"  total_eval_count = {sum(result.n_evals_per_fidelity.values())}")
    print(f"  total_compute_time = {result.total_compute_time:.3f} s")

    print(f"\n  per-fidelity breakdown:")
    print(f"    {'layer':>6s}  {'n_evals':>8s}  {'wall (s)':>10s}  {'cost (s)':>10s}  field")
    print(f"    {'-' * 6}  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 30}")
    for layer in result.fidelity_levels:
        n = result.n_evals_per_fidelity.get(layer, 0)
        wt = result.wall_time_per_fidelity.get(layer, 0.0)
        cost = result.cost_model.get(layer, float("nan"))
        field = result.report_fields_by_layer.get(layer, "")
        marker = "  *" if layer == result.hf_level else "   "
        print(f"   {marker}{layer:>3d}  {n:>8d}  {wt:>10.3f}  {cost:>10.4f}  {field}")

    extras = result.extras or {}
    if extras:
        print(f"\n  extras:")
        for k, v in extras.items():
            if k == "baseline":  # printed separately below
                continue
            if isinstance(v, np.ndarray):
                print(f"    {k:<24s} = {np.array2string(v, precision=6)}")
            else:
                print(f"    {k:<24s} = {v}")

    if baseline is not None:
        print(f"\n  baseline (staged):")
        print(f"    best_x         = {baseline.get('best_x')}")
        print(f"    best_params    = {baseline.get('best_params')}")
        print(f"    best_objective = {baseline.get('best_objective')}")
        print(f"    n_evals        = {baseline.get('n_evals')}")
        print(f"    compute_time   = {baseline.get('compute_time')} s")
        print(f"    converged      = {baseline.get('converged')}")

        # Side-by-side comparison: (method, best_objective, compute_time, n_evals_at_HF).
        mf_n_hf = result.n_evals_per_fidelity.get(result.hf_level, 0)
        sg_n_hf = baseline.get("n_evals_at_hf", 0)
        sg_obj = baseline.get("best_objective", float("nan"))
        sg_time = baseline.get("compute_time", float("nan"))
        print(f"\n  comparison (side-by-side):")
        print(
            f"    {'method':<16s}  {'best_objective':>16s}  "
            f"{'compute_time (s)':>18s}  {'n_evals_at_HF':>14s}"
        )
        print(
            f"    {'-' * 16}  {'-' * 16}  {'-' * 18}  {'-' * 14}"
        )
        print(
            f"    {result.method:<16s}  {result.best_objective:>16.6e}  "
            f"{result.total_compute_time:>18.3f}  {mf_n_hf:>14d}"
        )
        sg_obj_str = f"{sg_obj:.6e}" if isinstance(sg_obj, (int, float)) else str(sg_obj)
        sg_time_str = f"{sg_time:.3f}" if isinstance(sg_time, (int, float)) else str(sg_time)
        print(
            f"    {'staged':<16s}  {sg_obj_str:>16s}  "
            f"{sg_time_str:>18s}  {int(sg_n_hf):>14d}"
        )


def _run_cpp_validation(
    result: BOResult,
    *,
    N: int = _CPP_VALIDATION_N_DEFAULT,
    t_final: float = _CPP_VALIDATION_T_FINAL_DEFAULT,
) -> dict[str, Any] | None:
    """Re-run ``result.best_x`` at ``max_layer=8`` via the shoccs bridge.

    Returns a JSON-friendly dict with keys ``l8_stable``, ``l8_final_linf``,
    ``wall_time_s`` (and ``cpp_cutcell_violates_197_288`` for E4-classical),
    or ``None`` when validation is globally skipped: empty / non-finite
    winner, unsupported ``(scheme, kernel)``, or the shoccs binary cannot
    be found.  Per-call failures (L8 raises, bridge returns no payload)
    are captured as ``l8_error`` and do **not** raise — the analytical MF-BO
    verdict stands; L8 disagreement is purely diagnostic, matching the
    contract in :func:`sweeps.optimize._run_cpp_validation` and
    :func:`sweeps.pareto._run_front_cpp_validation`.

    The ``l8_`` prefix on the stable/final_linf keys mirrors the per-member
    schema in the Pareto module so the persisted JSON can be ingested by a
    common downstream parser.
    """
    if not result.best_params:
        print(
            "\n[bo] --validate-with-cpp: skipping — MF-BO did not produce a "
            "feasible winner (best_params is empty)."
        )
        return None
    if not np.isfinite(result.best_objective):
        print(
            "\n[bo] --validate-with-cpp: skipping — best_objective is "
            "non-finite; the analytical stack did not produce a feasible winner."
        )
        return None
    if result.kernel not in _CPP_SUPPORTED_KERNELS:
        print(
            f"\n[bo] --validate-with-cpp: skipping — kernel={result.kernel!r} "
            "is not C++-supported."
        )
        return None
    if result.scheme not in _CPP_SUPPORTED_SCHEMES:
        print(
            f"\n[bo] --validate-with-cpp: skipping — scheme={result.scheme!r} "
            "has no L8 bridge wired."
        )
        return None
    if not SHOCCS_BINARY.exists():
        print(
            f"\n[bo] --validate-with-cpp: skipping — shoccs binary not found "
            f"at {SHOCCS_BINARY}. Build it with `cmake --build build`."
        )
        return None

    print(
        f"\n[bo] --validate-with-cpp: running L8 ({result.scheme}/{result.kernel}) "
        f"at N={N}, t_final={t_final}..."
    )

    entry: dict[str, Any] = {}
    _record_cpp_cutcell_diagnostic(entry, result.scheme, result.kernel, result.best_x)

    try:
        report = brady2d_stability_score(
            result.scheme,
            result.kernel,
            result.best_params,
            max_layer=8,
            short_circuit=False,
            layer8_N=N,
            layer8_t_final=t_final,
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        print(f"[bo] L8 raised ({msg}); recording as failure.")
        entry["l8_stable"] = False
        entry["l8_final_linf"] = float("nan")
        entry["wall_time_s"] = 0.0
        entry["l8_error"] = msg
        return entry

    if report.layer8 is None:
        print(
            "[bo] L8 did not populate a report (unexpected with "
            "short_circuit=False); recording as failure."
        )
        entry["l8_stable"] = False
        entry["l8_final_linf"] = float("nan")
        entry["wall_time_s"] = 0.0
        entry["l8_error"] = "layer8 not populated"
        return entry

    l8 = report.layer8
    entry["l8_stable"] = bool(l8["stable"])
    entry["l8_final_linf"] = float(l8["final_linf"])
    entry["wall_time_s"] = float(l8["wall_time_s"])

    passed = entry["l8_stable"] and entry["l8_final_linf"] <= L8_FINAL_LINF_TOL
    if passed:
        print(
            f"[bo] L8 PASS: final_linf={entry['l8_final_linf']:.4e} "
            f"(wall={entry['wall_time_s']:.1f}s)"
        )
    elif entry["l8_stable"]:
        print(
            f"[bo] WARNING: L8 soft-failure: "
            f"final_linf={entry['l8_final_linf']:.4e} > "
            f"L8_FINAL_LINF_TOL={L8_FINAL_LINF_TOL} "
            f"(wall={entry['wall_time_s']:.1f}s). "
            "Analytical best_objective unchanged — L8 disagreement is diagnostic."
        )
    else:
        print(
            f"[bo] WARNING: L8 FAIL "
            f"(stable=False, final_linf={entry['l8_final_linf']:.4e}, "
            f"wall={entry['wall_time_s']:.1f}s). "
            "Analytical best_objective unchanged — L8 disagreement is diagnostic."
        )
    return entry


def _run_staged_baseline(
    result: BOResult,
    *,
    bounds: list[tuple[float, float]],
    seed: int,
    n_restarts: int = 10,
) -> dict[str, Any]:
    """Run ``run_staged_optimize`` against the same HF objective as MF-BO.

    Mirrors ``python -m sweeps optimize --method staged ...``'s CLI-resolved
    defaults so the baseline matches what users actually invoke.  The CLI
    defaults (see :func:`sweeps.optimize._run_method`) are:

    - ``inner_max_layer = 3`` (``args.max_layer or 3``)
    - ``inner_gate = max(inner_max_layer - 1, 0) = 2`` (``args.gate_layer or
      max(inner_max_layer - 1, 0)`` — matches ``make_objective``'s
      auto-inferred gate)

    The only fairness-fix override is ``validator_max_layer = max(hf_level,
    3)`` so the baseline validates at the same depth MF-BO targeted (the
    floor of 3 satisfies the staged constraint ``validator_max_layer >=
    inner_max_layer``).  Both runs share *seed*.

    Note: ``run_staged_optimize``'s function-level defaults are
    ``inner_gate=3, inner_max_layer=3`` (stricter than the CLI's
    ``inner_gate=2``).  The CLI form is the user-facing reference and is
    what 47.7a's head-to-head benchmark must compare against; the function
    defaults only fire when ``run_staged_optimize`` is called directly from
    Python without the CLI shim.

    Returns the JSON-friendly serialisation produced by
    :func:`sweeps.optimize._result_to_persist_dict`, with one extra key
    ``n_evals_at_hf`` derived from
    ``len(extras["validator_ranking"])`` so the side-by-side print can show
    how many HF (validator) evaluations the staged baseline consumed.  The
    persisted snapshot is what gets written under
    ``result.extras["baseline"]`` and ultimately into the JSON file.
    """
    objective = result.report_fields_by_layer[result.hf_level]
    validator_max_layer = max(int(result.hf_level), 3)
    inner_max_layer = min(3, validator_max_layer)
    inner_gate = max(inner_max_layer - 1, 0)

    print(
        f"\n[bo] --baseline staged: running run_staged_optimize "
        f"(scheme={result.scheme}, kernel={result.kernel}, "
        f"objective={objective}, inner_gate={inner_gate}, "
        f"inner_max_layer={inner_max_layer}, "
        f"validator_max_layer={validator_max_layer}, "
        f"n_restarts={n_restarts}, seed={seed})..."
    )
    staged: OptimizeResult = run_staged_optimize(
        scheme=result.scheme,
        kernel=result.kernel,
        report_field=objective,
        bounds=bounds,
        inner_gate=inner_gate,
        inner_max_layer=inner_max_layer,
        validator_max_layer=validator_max_layer,
        n_restarts=n_restarts,
        seed=seed,
    )
    record = _result_to_persist_dict(
        staged,
        scheme=result.scheme,
        kernel=result.kernel,
        objective=objective,
        bounds=bounds,
        gate_layer=inner_gate,
        max_layer=inner_max_layer,
        validator_max_layer=validator_max_layer,
    )
    # n_evals_at_hf for staged = number of validator-stage evaluations,
    # which is the count of (x, f) entries in the validator ranking.
    validator_ranking = staged.extras.get("validator_ranking") if staged.extras else None
    record["n_evals_at_hf"] = (
        len(validator_ranking) if validator_ranking is not None else 0
    )
    return record


def _resolve_cost_table(
    cost_model: str,
    fidelity_layers: list[int],
) -> dict[int, float] | None:
    """Resolve the per-layer cost table to forward to :func:`run_mfbo`.

    Returns ``None`` when *cost_model* is ``"constant"`` and every layer
    in *fidelity_layers* has an entry in :data:`DEFAULT_COST_TABLE` —
    letting :func:`run_mfbo` apply its own default behaviour.
    Returns a fresh dict slice when explicit costs are needed (currently
    just the same default values, but kept as an explicit dict so future
    overrides plug in here).  Raises :class:`NotImplementedError` for the
    ``"empirical"`` choice (deferred to a future item).
    """
    if cost_model == "empirical":
        raise NotImplementedError(
            "--cost-model empirical is reserved for a future item; pass "
            "--cost-model constant (or omit the flag) to use the plan-46 "
            "calibrated table."
        )
    missing = [m for m in fidelity_layers if m not in DEFAULT_COST_TABLE]
    if missing:
        raise ValueError(
            f"DEFAULT_COST_TABLE has no entries for layers {missing}; "
            "pass --fidelity-fields with a custom cost table is not yet "
            "supported on the CLI — file a follow-up if needed."
        )
    return None  # let run_mfbo slice DEFAULT_COST_TABLE itself


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweeps.bo",
        description=(
            "Multi-fidelity Bayesian optimisation (BoTorch qMFKG) over the "
            "Brady-Livescu stability cascade.  Picks (x, m) jointly to "
            "maximise expected information gain at the HF target per second "
            "of wall time.  See plans/47-mfbo.md."
        ),
    )
    parser.add_argument("--scheme", choices=["E2", "E4"], required=True)
    parser.add_argument("--kernel", choices=list(_KERNEL_CHOICES), required=True)
    parser.add_argument(
        "--objective",
        required=True,
        help=(
            'HF target as a dotted-path report field, e.g. '
            '"layer7.max_spectral_abscissa".  The HF layer is inferred from '
            "the prefix."
        ),
    )
    parser.add_argument(
        "--cheap-fidelities",
        type=int,
        nargs="+",
        required=True,
        metavar="N",
        help=(
            "External cascade layer indices to use as cheap surrogates, e.g. "
            "'1 3' or '1 3 5 6'.  Each must be < the HF layer inferred from "
            "--objective.  Default field per layer comes from a built-in "
            "table; override with --fidelity-fields."
        ),
    )
    parser.add_argument(
        "--fidelity-fields",
        nargs="+",
        default=None,
        metavar="LAYER=FIELD",
        help=(
            "Per-layer field overrides, e.g. '3=layer3.something_else'.  "
            "Useful when the canonical default is not what you want."
        ),
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs="+",
        default=None,
        metavar="VAL",
        help=(
            "Flat list of bound pairs (lo hi [lo hi ...]).  Falls back to "
            "DEFAULT_BOUNDS for the (scheme, kernel) pair if absent."
        ),
    )
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument(
        "--budget-evals",
        type=int,
        default=None,
        help="Total number of cascade evaluations (init + acquisition + final HF).",
    )
    budget.add_argument(
        "--budget-seconds",
        type=float,
        default=None,
        help="Wall-time budget in seconds (mutually exclusive with --budget-evals).",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="Initial design size (default: 5*d + 3 per Loeppky et al. 2009).",
    )
    parser.add_argument(
        "--num-fantasies",
        type=int,
        default=64,
        help="Number of fantasies for qMFKG (default: 64).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--cost-model",
        choices=list(_COST_MODEL_CHOICES),
        default="constant",
        help=(
            "'constant' uses the plan-46 calibrated DEFAULT_COST_TABLE.  "
            "'empirical' (per-eval learned cost) is reserved for a future item."
        ),
    )
    parser.add_argument(
        "--baseline",
        choices=list(_BASELINE_CHOICES),
        default="none",
        help=(
            "Run a comparator alongside MF-BO with the same seed.  'staged' "
            "invokes run_staged_optimize against the same HF objective.  "
            "Wired in plan 47.5b."
        ),
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help=(
            "Persist the BOResult as JSON under "
            "sweeps/bo_runs/<scheme>_<kernel>_<mangled>_<seed>.json (plan 47.4c)."
        ),
    )
    parser.add_argument(
        "--validate-with-cpp",
        action="store_true",
        help=(
            "Re-run best_x at max_layer=8 via the C++ bridge after MF-BO "
            "completes (plan 47.5a)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Forward to run_mfbo(verbose=True): one line per evaluation.",
    )

    args = parser.parse_args(argv)

    try:
        bounds = _resolve_bounds(args.scheme, args.kernel, args.bounds)
        _validate_kernel_bounds_dim(args.kernel, bounds)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        overrides = _parse_fidelity_fields(args.fidelity_fields)
        report_fields_by_layer = _build_report_fields_by_layer(
            args.objective,
            list(args.cheap_fidelities),
            overrides,
        )
    except ValueError as exc:
        parser.error(str(exc))

    fidelity_layers = sorted(report_fields_by_layer)
    try:
        cost_table = _resolve_cost_table(args.cost_model, fidelity_layers)
    except (NotImplementedError, ValueError) as exc:
        parser.error(str(exc))

    try:
        result = run_mfbo(
            scheme=args.scheme,
            kernel=args.kernel,
            report_fields_by_layer=report_fields_by_layer,
            bounds=bounds,
            budget_evals=args.budget_evals,
            budget_seconds=args.budget_seconds,
            cost_table=cost_table,
            seed=args.seed,
            n_init=args.n_init,
            num_fantasies=args.num_fantasies,
            verbose=args.verbose,
        )
    except ValueError as exc:
        parser.error(str(exc))

    # --- baseline (47.5b) ---------------------------------------------------
    # A baseline failure must NOT lose the MF-BO investment (plan 47.5b.2):
    # ``run_staged_optimize`` can blow up on ill-conditioned candidates or
    # propagate exceptions from ``brady2d_stability_score``; for ~5-minute
    # benchmark runs that means a successful BOResult is silently discarded
    # before ``--persist`` fires.  Mirror the failure-record convention from
    # :func:`_run_cpp_validation` (``l8_error`` on the L8-raises path) — record
    # ``error`` plus None-valued numeric fields so :func:`_print_summary`'s
    # ``isinstance(sg_obj, (int, float))`` branch falls through to ``str(...)``
    # and the persisted JSON captures the failure for post-hoc inspection.
    baseline_record: dict | None = None
    if args.baseline == "staged":
        try:
            baseline_record = _run_staged_baseline(
                result,
                bounds=bounds,
                seed=args.seed,
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(
                f"\n[bo] --baseline staged: run_staged_optimize raised {msg}; "
                "continuing without baseline."
            )
            baseline_record = {
                "error": msg,
                "method": "staged",
                "compute_time": None,
                "best_objective": None,
                "best_x": None,
                "n_evals": 0,
                "n_evals_at_hf": 0,
            }
        result.extras["baseline"] = baseline_record

    # --- C++ validation (47.5a) --------------------------------------------
    # Validation runs BEFORE --persist so the persisted JSON includes the
    # cpp_validation payload (lesson from plan 45.5a.1: reversing the order
    # would write a stale snapshot that does not reflect the L8 verdict).
    if args.validate_with_cpp:
        validation = _run_cpp_validation(result)
        if validation is not None:
            result.extras["cpp_validation"] = validation

    _print_summary(result, baseline=baseline_record)

    # --- persistence (47.4c) -----------------------------------------------
    if args.persist:
        try:
            from ._bo_io import save_bo_run
        except ImportError:
            print(
                "\n[bo] --persist: deferred to plan 47.4c; sweeps/_bo_io.py "
                "not yet present.  No file written."
            )
        else:
            written = save_bo_run(result)
            print(f"\n[bo] persisted run to {written}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
