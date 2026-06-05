"""Multi-objective Pareto optimization wrapper around the Brady-Livescu 2D
stability pipeline (plan 45).

Where :mod:`stencil_gen.optimizer` drives *scalar* objectives (a single dotted
path into :class:`StabilityReport`), this module targets the multi-objective
case: minimise a vector ``F(x) = [f_1(x), ..., f_m(x)]`` so the population
converges toward the *Pareto front* — the set of non-dominated parameter
vectors where no axis can be improved without worsening another.

Pareto-dominance: ``x`` dominates ``y`` iff ``F_i(x) <= F_i(y)`` for all ``i``
and strictly ``<`` for at least one.  The front is the subset of evaluated
points not dominated by any other.

References
----------
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). "A fast and elitist
multiobjective genetic algorithm: NSGA-II." *IEEE Transactions on Evolutionary
Computation*, 6(2), 182-197.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from stencil_gen.brady2d_stability import brady2d_stability_score
from stencil_gen.optimizer import (
    _infer_max_layer,
    _report_to_dict,
    extract_field,
    params_from_vector,
)

# Finite sentinel used when the multi-objective evaluation is infeasible
# (gate trip, shape mismatch, or ``brady2d_stability_score`` exception).
# pymoo's hypervolume indicator and ``ftol`` termination both reject ``+inf``,
# so we substitute a large finite number.  Downstream consumers (NSGA-II
# driver, persistence layer) filter sentinel rows out of the reported front.
_PARETO_SENTINEL: float = 1e12


@dataclass(frozen=True)
class ParetoPoint:
    """A single non-dominated member of a Pareto front.

    Attributes
    ----------
    x : np.ndarray
        Flat parameter vector of shape ``(n_var,)``, dtype ``float64``.  The
        optimiser's native representation — convert to a kernel-specific
        ``params`` dict via :func:`stencil_gen.optimizer.params_from_vector`.
    params : dict
        Kernel-specific parameter dict at ``x`` (the thing you would pass to
        :func:`brady2d_stability_score`).  Redundant with ``x`` but included
        so downstream code can consume either representation without carrying
        the kernel through.
    objectives : np.ndarray
        Vector of objective values, shape ``(n_obj,)``, dtype ``float64``.
        Aligned with :attr:`ParetoResult.objective_fields`.
    report : dict
        Serialised :class:`StabilityReport` at ``x`` (produced by
        ``_report_to_dict`` from :mod:`stencil_gen.optimizer`).  Empty dict if
        the evaluation produced no feasible report.
    """

    x: np.ndarray
    params: dict
    objectives: np.ndarray
    report: dict


@dataclass(frozen=True)
class ParetoResult:
    """Frozen record of a single multi-objective optimiser run.

    Attributes
    ----------
    front : tuple[ParetoPoint, ...]
        Non-dominated members at the end of the run.  Empty tuple if the run
        produced no feasible point.
    objective_fields : tuple[str, ...]
        Dotted-path identifiers matching :attr:`ParetoPoint.objectives`, e.g.
        ``("layer1.boundary_gv_err", "layer_bl42.max_spectral_abscissa")``.
    scheme : str
        Scheme identifier forwarded to :func:`brady2d_stability_score`.
    kernel : str
        Kernel identifier forwarded to :func:`brady2d_stability_score`.
    bounds : tuple[tuple[float, float], ...]
        Parameter bounds used for the run, one ``(lo, hi)`` pair per variable.
    method : str
        Name of the driver (``"NSGA-II"`` for plan 45; ``"NSGA-III"`` reserved
        for a future extension).
    pop_size : int
        Population size used by the evolutionary algorithm.
    n_gen : int
        Number of generations executed.
    n_evals : int
        Total number of objective evaluations across the run.
    seed : int
        RNG seed supplied to the algorithm.
    compute_time : float
        Wall-clock seconds for the run.
    hv_trace : tuple[float, ...]
        Hypervolume of the current non-dominated set at the end of each
        generation.  Length equals ``n_gen``.
    ref_point : tuple[float, ...]
        Reference point used for the hypervolume indicator, aligned with
        :attr:`objective_fields`.
    extras : dict
        Free-form additional fields (e.g. ``n_sentinel_filtered``,
        ``cpp_validation``, driver-specific diagnostics).
    """

    front: tuple[ParetoPoint, ...]
    objective_fields: tuple[str, ...]
    scheme: str
    kernel: str
    bounds: tuple[tuple[float, float], ...]
    method: str
    pop_size: int
    n_gen: int
    n_evals: int
    seed: int
    compute_time: float
    hv_trace: tuple[float, ...]
    ref_point: tuple[float, ...]
    extras: dict


def make_multi_objective(
    scheme: str,
    kernel: str,
    report_fields: Sequence[str],
    *,
    gate_layer: int | None = None,
    max_layer: int | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a feasibility-gated vector-valued objective ``f(x) -> np.ndarray``.

    Parallels :func:`stencil_gen.optimizer.make_objective` but returns a
    length-``len(report_fields)`` vector instead of a scalar, for use with
    multi-objective evolutionary algorithms (NSGA-II; see :func:`run_nsga2`).

    The returned closure converts a flat vector ``x`` into a kernel-specific
    ``params`` dict, runs :func:`brady2d_stability_score` in short-circuit
    mode up to ``max_layer``, and returns:

    - ``np.full(n_obj, _PARETO_SENTINEL)`` if any layer at or before
      ``gate_layer`` failed (feasibility cliff), if ``x`` has the wrong
      shape for the kernel, or if :func:`brady2d_stability_score` raised.
    - a vector of :func:`extract_field` values (one per ``report_fields``)
      otherwise.  Individual missing fields still produce ``+inf`` from
      :func:`extract_field`; pymoo tolerates per-element ``+inf`` on a
      partially-successful evaluation, but the sentinel path keeps
      hypervolume well-defined when the whole evaluation is infeasible.

    Parameters
    ----------
    scheme, kernel
        Forwarded to :func:`brady2d_stability_score`.
    report_fields
        Sequence of dotted-path identifiers (length ≥ 2).  Each must resolve
        via :func:`_infer_max_layer` unless ``max_layer`` is supplied
        explicitly.
    gate_layer
        Highest layer whose failure forces the sentinel vector.  Defaults to
        ``max_layer - 1`` (floored at 0) — consistent with
        :func:`make_objective`.
    max_layer
        Highest layer actually executed.  Defaults to
        ``max(_infer_max_layer(f) for f in report_fields)`` so the pipeline
        runs deep enough to populate every requested field.  Raises
        ``ValueError`` if any field's layer cannot be inferred and no
        explicit ``max_layer`` is given, or if ``max_layer < gate_layer``.
    """
    fields = tuple(report_fields)
    if len(fields) < 2:
        raise ValueError(
            f"make_multi_objective requires >= 2 report_fields, got {len(fields)}"
        )

    if max_layer is None:
        inferred_layers = []
        for f in fields:
            layer = _infer_max_layer(f)
            if layer is None:
                raise ValueError(
                    f"cannot infer max_layer from report_field={f!r}; "
                    "pass max_layer explicitly"
                )
            inferred_layers.append(layer)
        max_layer = max(inferred_layers)
    if gate_layer is None:
        gate_layer = max(max_layer - 1, 0)
    if max_layer < gate_layer:
        raise ValueError(
            f"max_layer={max_layer} is less than gate_layer={gate_layer}; "
            "raise max_layer or lower gate_layer"
        )

    n_obj = len(fields)
    sentinel_vec = np.full(n_obj, _PARETO_SENTINEL, dtype=float)

    def objective(x: np.ndarray) -> np.ndarray:
        try:
            params = params_from_vector(kernel, x)
            report = brady2d_stability_score(
                scheme,
                kernel,
                params,
                max_layer=max_layer,
                short_circuit=True,
            )
        except Exception:
            return sentinel_vec.copy()
        if report.failed_layer is not None and report.failed_layer <= gate_layer:
            return sentinel_vec.copy()
        return np.array(
            [extract_field(report, f) for f in fields], dtype=float
        )

    return objective


# --- NSGA-II driver ----------------------------------------------------------


def _auto_ref_point(
    f: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple[float, float]],
    n_obj: int,
    seed: int,
    n_probes: int = 20,
) -> np.ndarray:
    """Pick a pymoo-friendly reference point for the hypervolume indicator.

    Runs ``n_probes`` uniform-random samples inside ``bounds``, filters out
    sentinel rows, and returns ``1.1 * max`` of the finite objective columns
    clipped to ``[1.0, _PARETO_SENTINEL)``.  Falls back to a vector of ones if
    no feasible probe is found (the HV will be zero in that case — still
    well-defined, and the fallback makes the downstream code total).
    """
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    samples = rng.uniform(lb, ub, size=(n_probes, lb.size))
    rows: list[np.ndarray] = []
    for x in samples:
        y = np.asarray(f(x), dtype=float)
        if y.shape != (n_obj,):
            continue
        if np.all(np.isfinite(y)) and np.all(y < _PARETO_SENTINEL):
            rows.append(y)
    if not rows:
        return np.ones(n_obj, dtype=float)
    arr = np.vstack(rows)
    ref = 1.1 * np.max(arr, axis=0)
    ref = np.clip(ref, 1.0, _PARETO_SENTINEL - 1.0)
    return ref


class _HVCallback:
    """Per-generation hypervolume recorder for a pymoo ``Algorithm`` run.

    Subclasses :class:`pymoo.core.callback.Callback` at import time so the
    pymoo dependency is contained in :func:`run_nsga2` (and monkey-patchable
    in tests that do not want to import pymoo).  On each ``notify`` the
    callback pulls the current non-dominated set ``algorithm.opt`` (NOT the
    full population history — see pymoo docs on ``Algorithm.opt``), filters
    sentinel / non-finite rows, and records the hypervolume under the
    configured reference point.  Empty filtered set produces ``0.0`` (by
    convention — the indicator is undefined there).
    """

    def __init__(self, hv_indicator, ref_point: np.ndarray):
        from pymoo.core.callback import Callback

        class _Inner(Callback):
            def __init__(self, hv_indicator, ref_point):
                super().__init__()
                self.data["hv"] = []
                self.data["n_nds"] = []
                self._hv = hv_indicator
                self._ref = ref_point

            def notify(self, algorithm):
                F = algorithm.opt.get("F") if algorithm.opt is not None else None
                if F is None or len(F) == 0:
                    self.data["hv"].append(0.0)
                    self.data["n_nds"].append(0)
                    return
                F = np.asarray(F, dtype=float)
                finite = np.all(np.isfinite(F), axis=1)
                below = np.all(F < self._ref, axis=1)
                sel = finite & below
                F_filt = F[sel]
                self.data["n_nds"].append(int(F_filt.shape[0]))
                if F_filt.shape[0] == 0:
                    self.data["hv"].append(0.0)
                else:
                    self.data["hv"].append(float(self._hv(F_filt)))

        self._cb = _Inner(hv_indicator, ref_point)

    @property
    def data(self) -> dict:
        return self._cb.data

    @property
    def pymoo_callback(self):
        return self._cb


def run_nsga2(
    scheme: str,
    kernel: str,
    report_fields: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    *,
    pop_size: int = 40,
    n_gen: int = 50,
    seed: int = 1,
    ref_point: Sequence[float] | None = None,
    gate_layer: int | None = None,
    max_layer: int | None = None,
    verbose: bool = False,
    objective: Callable[[np.ndarray], np.ndarray] | None = None,
) -> ParetoResult:
    """Run pymoo's NSGA-II on the multi-objective Brady-Livescu problem.

    Parameters
    ----------
    scheme, kernel
        Forwarded to :func:`brady2d_stability_score`.
    report_fields
        Sequence of dotted-path identifiers (length ≥ 2) identifying the
        objectives to minimise.  Each is resolved via :func:`extract_field`.
    bounds
        ``[(lo, hi), ...]`` per parameter; the optimiser samples ``x`` uniformly
        in this box.
    pop_size, n_gen, seed
        Standard NSGA-II hyperparameters.  Defaults match plan 45.
    ref_point
        Reference point for the hypervolume indicator, shape ``(n_obj,)``.  If
        ``None``, auto-selected as ``1.1 * max`` of ``n_probes=20`` uniform
        feasible samples (per plan 45.2a).
    gate_layer, max_layer
        Forwarded to :func:`make_multi_objective`; both auto-inferred if
        omitted.
    verbose
        Forwarded to pymoo's :func:`minimize`.
    objective
        Override hook for tests: if supplied, this vector-valued callable is
        used instead of building one via :func:`make_multi_objective`.  The
        signature must be ``(np.ndarray of shape (n_var,)) -> np.ndarray of
        shape (n_obj,)``.

    Returns
    -------
    ParetoResult
        Frozen record.  ``front`` contains only finite (non-sentinel) members;
        ``extras["n_sentinel_filtered"]`` counts how many ``res.F`` rows were
        discarded.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.indicators.hv import HV
    from pymoo.optimize import minimize

    fields = tuple(report_fields)
    if len(fields) < 2:
        raise ValueError(
            f"run_nsga2 requires >= 2 report_fields, got {len(fields)}"
        )
    bounds_t = tuple((float(lo), float(hi)) for lo, hi in bounds)
    n_var = len(bounds_t)
    n_obj = len(fields)
    if n_var == 0:
        raise ValueError("run_nsga2 requires at least one parameter bound")

    rebuild_reports = objective is None
    if objective is None:
        objective = make_multi_objective(
            scheme,
            kernel,
            fields,
            gate_layer=gate_layer,
            max_layer=max_layer,
        )

    if ref_point is None:
        ref_arr = _auto_ref_point(objective, bounds_t, n_obj, seed)
    else:
        ref_arr = np.asarray(ref_point, dtype=float).ravel()
        if ref_arr.shape != (n_obj,):
            raise ValueError(
                f"ref_point shape {ref_arr.shape} does not match n_obj={n_obj}"
            )

    xl = np.array([b[0] for b in bounds_t], dtype=float)
    xu = np.array([b[1] for b in bounds_t], dtype=float)

    class _StabilityProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=0, xl=xl, xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = np.asarray(objective(np.asarray(x, dtype=float)), dtype=float)

    hv_indicator = HV(ref_point=ref_arr)
    cb = _HVCallback(hv_indicator, ref_arr)
    problem = _StabilityProblem()
    algorithm = NSGA2(pop_size=pop_size)

    t0 = time.perf_counter()
    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=seed,
        verbose=verbose,
        callback=cb.pymoo_callback,
    )
    compute_time = time.perf_counter() - t0

    X = np.atleast_2d(np.asarray(res.X, dtype=float)) if res.X is not None else np.zeros((0, n_var))
    F = np.atleast_2d(np.asarray(res.F, dtype=float)) if res.F is not None else np.zeros((0, n_obj))
    if X.size and X.shape == (n_var,):
        X = X.reshape(1, n_var)
    if F.size and F.shape == (n_obj,):
        F = F.reshape(1, n_obj)

    finite_mask = np.all(np.isfinite(F), axis=1) if F.size else np.zeros(0, dtype=bool)
    below_sentinel_mask = np.all(F < _PARETO_SENTINEL, axis=1) if F.size else np.zeros(0, dtype=bool)
    keep_mask = finite_mask & below_sentinel_mask
    n_sentinel_filtered = int(F.shape[0] - int(keep_mask.sum())) if F.size else 0
    X_keep = X[keep_mask]
    F_keep = F[keep_mask]

    rebuild_max_layer = (
        max_layer
        if max_layer is not None
        else max((_infer_max_layer(f) or 0) for f in fields)
    )
    points: list[ParetoPoint] = []
    for xi, fi in zip(X_keep, F_keep):
        try:
            params = params_from_vector(kernel, xi)
        except Exception:
            params = {}
        report_dict: dict = {}
        if rebuild_reports and params:
            try:
                rep = brady2d_stability_score(
                    scheme,
                    kernel,
                    params,
                    max_layer=rebuild_max_layer,
                    short_circuit=True,
                )
                report_dict = _report_to_dict(rep)
            except Exception:
                report_dict = {}
        points.append(
            ParetoPoint(
                x=np.asarray(xi, dtype=float).copy(),
                params=params,
                objectives=np.asarray(fi, dtype=float).copy(),
                report=report_dict,
            )
        )

    hv_trace = tuple(float(v) for v in cb.data.get("hv", []))
    n_evals = int(getattr(res.algorithm.evaluator, "n_eval", 0))

    return ParetoResult(
        front=tuple(points),
        objective_fields=fields,
        scheme=scheme,
        kernel=kernel,
        bounds=bounds_t,
        method="NSGA-II",
        pop_size=pop_size,
        n_gen=n_gen,
        n_evals=n_evals,
        seed=seed,
        compute_time=compute_time,
        hv_trace=hv_trace,
        ref_point=tuple(float(v) for v in ref_arr),
        extras={
            "n_sentinel_filtered": n_sentinel_filtered,
            "hv_n_nds": tuple(int(v) for v in cb.data.get("n_nds", [])),
        },
    )
