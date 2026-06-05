"""Per-run JSON persistence for :class:`stencil_gen.bo.BOResult`.

Mirrors :mod:`sweeps._pareto_io` (plan 45.4a) for multi-fidelity Bayesian
optimization runs.  Each run lives in its own file under
``sweeps/bo_runs/`` so that diffs stay readable, two MF-BO sweeps cannot
race on the same JSON, and :class:`tests.test_phs.TestRegressionBOBenchmark`
can discover stored runs by globbing the directory.

Filename scheme: ``{scheme}_{kernel}_{mangled_objective}_{seed}.json``,
where ``mangled_objective`` is the HF field path with dots replaced by
underscores.  The trailing ``_{seed}`` keeps replicates from clobbering
each other — multi-fidelity BO is stochastic enough that seed sweeps are
the standard validation pattern.

See ``plans/47-mfbo.md`` item 47.4c for the spec.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stencil_gen.bo import BOResult


BO_RUNS_DIR: Path = Path(__file__).parent / "bo_runs"


_INT_KEYED_TOP_LEVEL: tuple[str, ...] = (
    "report_fields_by_layer",
    "cost_model",
    "n_evals_per_fidelity",
    "wall_time_per_fidelity",
)
"""Top-level :class:`BOResult` fields whose ``dict[int, ...]`` keys must be
restored after :func:`json.load` downgrades them to ``str``.

The factory :func:`stencil_gen.bo.make_multi_fidelity_objective` performs
``inferred > layer`` against the layer keys, which raises ``TypeError`` if
the keys are strings.  See plan item 47.4c.1 for the full motivation.
"""


def _restore_int_keys(data: dict) -> dict:
    """Mutate ``data`` in place: restore int keys for the four whitelisted fields.

    JSON serialises every object key as a string; this helper inverts that
    for the four :class:`BOResult` fields whose schema is ``dict[int, ...]``.
    Skips silently if a field is missing (older payloads or partial dicts)
    so we do not break forward-compat.
    """
    for name in _INT_KEYED_TOP_LEVEL:
        field = data.get(name)
        if isinstance(field, dict):
            data[name] = {int(k): v for k, v in field.items()}
    return data


def _mangle_objective(field: str) -> str:
    """Map a single dotted field path to a filename-safe token.

    ``"layer7.max_spectral_abscissa"`` → ``"layer7_max_spectral_abscissa"``.
    BO has a single HF objective (unlike Pareto's tuple of objectives), so
    the mangling collapses to a one-liner without the ``"__"`` join.
    """
    return field.replace(".", "_")


class _BOEncoder(json.JSONEncoder):
    """Encoder that knows how to serialise numpy / dataclasses / Path / complex.

    The ``complex`` handler is intentional: any layer that surfaces a
    :class:`KreissResult` (e.g. via ``layer2`` if it lands in fidelity
    layers — see plan 46.2b) will carry a ``witness_s`` complex through
    ``extras``.  Without this branch, ``json.dump`` would raise
    ``TypeError: Object of type complex is not JSON serializable``.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, complex):
            return [o.real, o.imag]
        return super().default(o)


def _eval_to_ordered(eval_record: Any) -> OrderedDict[str, Any]:
    """Project a :class:`BOEval` into the canonical key order."""
    if is_dataclass(eval_record) and not isinstance(eval_record, type):
        d = asdict(eval_record)
    else:
        d = dict(eval_record)
    return OrderedDict(
        (
            ("x", d.get("x")),
            ("params", d.get("params")),
            ("fidelity", d.get("fidelity")),
            ("value", d.get("value")),
            ("wall_time", d.get("wall_time")),
            ("report", d.get("report", {})),
        )
    )


def _result_to_ordered(result: BOResult) -> OrderedDict[str, Any]:
    """Project a :class:`BOResult` into the canonical key order.

    Order follows the dataclass field declaration in
    :class:`stencil_gen.bo.BOResult` so that diffs against the
    documented schema stay obvious.  ``extras`` is the natural tail.
    """
    eval_history = tuple(_eval_to_ordered(e) for e in result.eval_history)
    hf_eval_history = tuple(_eval_to_ordered(e) for e in result.hf_eval_history)
    return OrderedDict(
        (
            ("best_x", result.best_x),
            ("best_params", result.best_params),
            ("best_objective", float(result.best_objective)),
            ("best_report", result.best_report),
            ("method", result.method),
            ("scheme", result.scheme),
            ("kernel", result.kernel),
            ("bounds", [list(b) for b in result.bounds]),
            ("fidelity_levels", list(result.fidelity_levels)),
            ("hf_level", int(result.hf_level)),
            ("report_fields_by_layer", result.report_fields_by_layer),
            ("cost_model", result.cost_model),
            ("n_evals_per_fidelity", result.n_evals_per_fidelity),
            ("wall_time_per_fidelity", result.wall_time_per_fidelity),
            ("total_compute_time", float(result.total_compute_time)),
            ("eval_history", eval_history),
            ("hf_eval_history", hf_eval_history),
            ("gp_hyperparameters", result.gp_hyperparameters),
            ("seed", int(result.seed)),
            ("converged", bool(result.converged)),
            ("stop_reason", result.stop_reason),
            ("extras", result.extras or {}),
        )
    )


def save_bo_run(
    result: BOResult,
    directory: Path = BO_RUNS_DIR,
) -> Path:
    """Serialize ``result`` to ``{directory}/{scheme}_{kernel}_{mangled}_{seed}.json``.

    Creates ``directory`` (and any missing parents) on demand.  The HF
    field path is read from ``result.report_fields_by_layer[result.hf_level]``
    and mangled by replacing ``.`` with ``_``.  The ``_{seed}`` suffix is
    always present so replicates of the same ``(scheme, kernel, objective)``
    do not clobber each other.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    hf_field = result.report_fields_by_layer[result.hf_level]
    mangled = _mangle_objective(hf_field)
    filename = f"{result.scheme}_{result.kernel}_{mangled}_{result.seed}.json"
    path = directory / filename
    payload = _result_to_ordered(result)
    with open(path, "w") as f:
        json.dump(payload, f, cls=_BOEncoder, sort_keys=False, indent=2)
        f.write("\n")
    return path


def load_bo_run(path: Path) -> dict:
    """Read a previously-saved BO run JSON file and return the raw dict.

    Returns the JSON as a plain ``dict`` (not a :class:`BOResult`).  The
    regression test in :class:`tests.test_phs.TestRegressionBOBenchmark`
    rebuilds a multi-fidelity objective from ``report_fields_by_layer`` and
    re-evaluates ``best_x`` at HF; it does not need dataclass round-tripping.

    Int keys are restored on the four whitelisted top-level fields
    (``report_fields_by_layer``, ``cost_model``, ``n_evals_per_fidelity``,
    ``wall_time_per_fidelity``) — see :data:`_INT_KEYED_TOP_LEVEL` and plan
    item 47.4c.1.  Without this, the loaded dict cannot be piped into
    :func:`stencil_gen.bo.make_multi_fidelity_objective` (the factory's
    field-vs-layer validation raises ``TypeError`` on string keys).
    """
    with open(Path(path)) as f:
        data = json.load(f)
    return _restore_int_keys(data)


def iter_bo_runs(directory: Path = BO_RUNS_DIR) -> Iterator[Path]:
    """Yield every ``*.json`` file in ``directory`` (non-recursive, sorted)."""
    directory = Path(directory)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        if path.is_file():
            yield path
