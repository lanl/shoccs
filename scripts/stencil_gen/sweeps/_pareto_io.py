"""Per-run JSON persistence for :class:`stencil_gen.pareto.ParetoResult`.

Deliberately kept separate from :mod:`sweeps._common` (whose single concern is
``known_values.json``).  NSGA-II runs produce larger, more structured records
than the scalar optima table and each run lives in its own file under
``sweeps/pareto_fronts/``.  That split keeps git diffs clean, removes a
concurrency hazard (two sweeps racing on the same JSON), and lets
:class:`tests.test_phs.TestRegressionBrady2DPareto` discover stored fronts by
globbing a directory instead of indexing into a growing dict.

See ``plans/45-pareto-optimization.md`` item 45.4a for the spec.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from stencil_gen.pareto import ParetoResult


PARETO_FRONTS_DIR: Path = Path(__file__).parent / "pareto_fronts"


def _mangle_objectives(fields: tuple[str, ...] | list[str]) -> str:
    """Duplicate of ``sweeps.pareto._mangle_objectives`` without the import cycle.

    Kept tiny so this module stays independent of the CLI layer: importing
    :mod:`sweeps.pareto` here would pull in pymoo on every persistence access.
    """
    return "__".join(f.replace(".", "_") for f in fields)


class _ParetoEncoder(json.JSONEncoder):
    """Encoder that knows how to serialize numpy scalars/arrays and dataclasses.

    ``json`` itself already serialises tuples as lists, so no special-case is
    needed for tuples — the ordering guarantee in :func:`save_pareto_front`
    comes from the explicit ``OrderedDict`` layout, not from the encoder.
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


def _point_to_ordered(point: Any) -> OrderedDict[str, Any]:
    """Serialize a :class:`ParetoPoint` (or compatible mapping) with stable key order."""
    if is_dataclass(point) and not isinstance(point, type):
        d = asdict(point)
    else:
        d = dict(point)
    return OrderedDict(
        (
            ("x", d.get("x")),
            ("params", d.get("params")),
            ("objectives", d.get("objectives")),
            ("report", d.get("report", {})),
        )
    )


def _result_to_ordered(result: ParetoResult) -> OrderedDict[str, Any]:
    """Project a :class:`ParetoResult` into the canonical key order.

    The order matches the documented schema in plan 45.4a:
    ``scheme, kernel, method, objective_fields, bounds, pop_size, n_gen,
    n_evals, seed, compute_time, ref_point, hv_trace, front, extras``.
    """
    front = tuple(_point_to_ordered(p) for p in result.front)
    return OrderedDict(
        (
            ("scheme", result.scheme),
            ("kernel", result.kernel),
            ("method", result.method),
            ("objective_fields", list(result.objective_fields)),
            ("bounds", [list(b) for b in result.bounds]),
            ("pop_size", int(result.pop_size)),
            ("n_gen", int(result.n_gen)),
            ("n_evals", int(result.n_evals)),
            ("seed", int(result.seed)),
            ("compute_time", float(result.compute_time)),
            ("ref_point", list(result.ref_point)),
            ("hv_trace", list(result.hv_trace)),
            ("front", front),
            ("extras", result.extras or {}),
        )
    )


def save_pareto_front(
    result: ParetoResult,
    directory: Path = PARETO_FRONTS_DIR,
) -> Path:
    """Serialize ``result`` to ``{directory}/{scheme}_{kernel}_{mangled}.json``.

    Creates ``directory`` (and any missing parents) on demand.  The returned
    path is the absolute location written.  Key ordering is fixed via an
    explicit :class:`OrderedDict` layout so diffs between runs stay readable.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mangled = _mangle_objectives(result.objective_fields)
    filename = f"{result.scheme}_{result.kernel}_{mangled}.json"
    path = directory / filename
    payload = _result_to_ordered(result)
    with open(path, "w") as f:
        json.dump(payload, f, cls=_ParetoEncoder, sort_keys=False, indent=2)
        f.write("\n")
    return path


def load_pareto_front(path: Path) -> dict:
    """Read a previously-saved Pareto front JSON file and return the raw dict.

    Returns the JSON as a plain ``dict`` (not a :class:`ParetoResult`).  The
    regression test in :class:`tests.test_phs.TestRegressionBrady2DPareto`
    rebuilds a vector-valued objective from ``objective_fields`` and
    re-evaluates each stored ``x``; it does not need dataclass round-tripping.
    """
    with open(Path(path)) as f:
        return json.load(f)


def iter_pareto_fronts(directory: Path = PARETO_FRONTS_DIR) -> Iterator[Path]:
    """Yield every ``*.json`` file in ``directory`` (non-recursive, sorted)."""
    directory = Path(directory)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        if path.is_file():
            yield path
