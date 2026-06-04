"""Optimization wrapper around the Brady-Livescu 2D stability pipeline.

Implements the layered cascade approach of Phase 43: a cheap inner objective
built from short-circuited :func:`brady2d_stability_score` calls drives an
off-the-shelf scipy optimizer, top-k survivors are re-ranked at a higher
(more expensive) ``max_layer``, and the winner is optionally pushed through
the L8 C++ bridge for simulation-level validation.

See ``plans/43-stability-optimization-framework.md`` for the plan, scope, and
algorithm choices (Nelder-Mead, COBYQA, SHGO, differential_evolution,
Sobol-seeded multi-start).

The public API surface is:

- :class:`OptimizeResult` — frozen record returned by every ``run_*`` helper.
- :data:`DEFAULT_BOUNDS` — ``(scheme, kernel) -> list[(lo, hi)]`` fallback.
- :func:`params_from_vector` / :func:`vector_from_params` — kernel-aware
  mapping between the optimizer's flat ``x`` and the nested ``params`` dict
  that ``brady2d_stability_score`` expects.
- :func:`extract_field` — dotted-path lookup into :class:`StabilityReport`.
- :func:`make_objective` — builds a feasibility-gated ``f(x) -> float``.
- :func:`run_scipy_local`, :func:`run_scipy_shgo`, :func:`run_scipy_de`,
  :func:`multi_start_optimize`, :func:`run_staged_optimize` — the driver
  helpers.
"""

from __future__ import annotations

import dataclasses
import operator
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np
import scipy
from scipy.optimize import differential_evolution, minimize, shgo
from scipy.stats import qmc

from stencil_gen.brady2d_stability import brady2d_stability_score


# --- bounds ------------------------------------------------------------------

# Per-(scheme, kernel) default bounds.  Matches the parameter-spaces table in
# plans/43-stability-optimization-framework.md.  Gaussian/multiquadric ranges
# are given in log-uniform space only in the UI sense; the optimizer operates
# on the raw ε value — log-sampling for multi-start is applied by Sobol scaling
# inside :func:`multi_start_optimize` when bounds span more than a decade.
#
# Scope note (plan 43.1d, option b): ``"tension-penalty"`` and
# ``"mixed-epsilon"`` are intentionally omitted.  ``brady2d_stability_score``
# and its layer helpers currently dispatch only
# ``kernel ∈ {"classical", "tension", "gaussian", "multiquadric"}``; the two
# excluded families live in dedicated sweeps (``sweeps/tension_penalty_sweep``
# and ``sweeps/mixed_epsilon_sweep``) that bypass the layered pipeline.
# Extending the layered pipeline to those kernels is deferred — see the
# "What this plan does NOT do" section of the plan file.
#
# Classical-α note (plan 43.9a): the C++ ``E4_1`` stencil in
# ``src/stencils/E4_1.cpp`` imposes ``alpha[1] >= 197/288 ≈ 0.684`` to avoid
# an interior singularity in the cut-cell psi denominator for ``psi ∈ (0, 1)``.
# That constraint is cut-cell-specific.  The analytical layers L1–L7 (and the
# Python ``_build_classical_diff_matrix``) operate on uniform grids with no
# psi involvement, and the Brady-Livescu published feasible point
# ``α ≈ [-0.7733, 0.1624]`` lives *below* 197/288 — a grid-probe of
# ``alpha[1] ∈ [0.0, 2.0]`` at ``alpha[0] = -0.77`` finds L3-feasibility only
# in ``alpha[1] ∈ [~0.08, ~0.17]`` and total infeasibility once
# ``alpha[1] ≥ 0.2``.  The analytical and cut-cell feasible regions therefore
# do not overlap for E4 classical-α, so the optimizer uses the *analytical*
# feasible envelope.  L8 C++ validation (plan 43.10) will reject any winner
# that violates ``alpha[1] ≥ 197/288``; that rejection is the diagnostic
# signal, not an optimizer bound.
DEFAULT_BOUNDS: dict[tuple[str, str], list[tuple[float, float]]] = {
    ("E2", "tension"): [(0.5, 20.0)],
    ("E4", "tension"): [(0.5, 20.0)],
    ("E2", "gaussian"): [(0.1, 5.0)],
    ("E4", "gaussian"): [(0.1, 5.0)],
    ("E2", "multiquadric"): [(0.1, 5.0)],
    ("E4", "multiquadric"): [(0.1, 5.0)],
    ("E4", "classical"): [(-2.0, 2.0), (0.05, 2.0)],
}


# --- result record -----------------------------------------------------------

@dataclass(frozen=True)
class OptimizeResult:
    """Frozen record of a single optimizer run.

    Attributes
    ----------
    best_params : dict
        Kernel-specific params dict at the optimum (the thing you would pass
        to :func:`brady2d_stability_score`).
    best_x : np.ndarray
        Flat parameter vector at the optimum.
    best_objective : float
        Value of the objective at ``best_x``.  ``+inf`` if no feasible point
        was found.
    best_report : dict
        Serialized :class:`StabilityReport` at the optimum (empty dict if no
        feasible point was found).
    method : str
        Name of the driver ("Nelder-Mead", "COBYQA", "SHGO", "DE",
        "multi-start", "staged").
    converged : bool
        Whether the underlying scipy call reported convergence AND the best
        objective is finite.
    n_evals : int
        Total objective evaluations performed.
    compute_time : float
        Wall-clock seconds.
    history : list
        ``[(x, f), ...]`` sampled during the run.  May be empty for drivers
        that do not expose per-step callbacks.
    extras : dict
        Free-form additional fields (e.g. ``n_local_minima`` for SHGO,
        ``stage`` for staged).
    """

    best_params: dict
    best_x: np.ndarray
    best_objective: float
    best_report: dict
    method: str
    converged: bool
    n_evals: int
    compute_time: float
    history: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)


# --- primitives: params <-> vector -------------------------------------------

_SCALAR_EPSILON_KERNELS = ("gaussian", "multiquadric")


def params_from_vector(kernel: str, x: np.ndarray) -> dict:
    """Convert a flat vector to the kernel-specific ``params`` dict that
    :func:`brady2d_stability_score` consumes.

    Kernel mapping (see plan 43, section "Parameter spaces in scope"):

    - ``"tension"``                     : ``x=[σ]``            → ``{"sigma": σ}``
    - ``"gaussian"`` / ``"multiquadric"``: ``x=[ε]``            → ``{"epsilon": ε}``
    - ``"classical"``                   : ``x=[α₀, α₁]``       → ``{"alpha": [α₀, α₁]}``

    The ``"tension-penalty"`` and ``"mixed-epsilon"`` families are out of
    scope for this optimizer (plan 43.1d, option b) — ``brady2d_stability_score``
    does not route those kernels.  Use the standalone
    ``sweeps/tension_penalty_sweep`` / ``sweeps/mixed_epsilon_sweep`` entry
    points for those parameter spaces.
    """
    x = np.asarray(x, dtype=float).ravel()
    if kernel == "tension":
        if x.size != 1:
            raise ValueError(f"kernel='tension' expects 1D vector, got shape {x.shape}")
        return {"sigma": float(x[0])}
    if kernel in _SCALAR_EPSILON_KERNELS:
        if x.size != 1:
            raise ValueError(f"kernel={kernel!r} expects 1D vector, got shape {x.shape}")
        return {"epsilon": float(x[0])}
    if kernel == "classical":
        if x.size != 2:
            raise ValueError(f"kernel='classical' expects 2D vector, got shape {x.shape}")
        return {"alpha": [float(x[0]), float(x[1])]}
    raise ValueError(f"unknown kernel: {kernel!r}")


def vector_from_params(kernel: str, params: dict) -> np.ndarray:
    """Inverse of :func:`params_from_vector`.

    Returns a flat ``np.ndarray`` of dtype ``float`` whose layout matches the
    convention in :func:`params_from_vector`.
    """
    if kernel == "tension":
        return np.array([float(params["sigma"])], dtype=float)
    if kernel in _SCALAR_EPSILON_KERNELS:
        return np.array([float(params["epsilon"])], dtype=float)
    if kernel == "classical":
        alpha = params["alpha"]
        if len(alpha) != 2:
            raise ValueError(
                f"kernel='classical' expects alpha of length 2, got {len(alpha)}"
            )
        return np.array([float(alpha[0]), float(alpha[1])], dtype=float)
    raise ValueError(f"unknown kernel: {kernel!r}")


# --- primitives: report field extraction -------------------------------------

def extract_field(report, dotted_path: str) -> float:
    """Dotted-path lookup into a :class:`StabilityReport`.

    The first segment resolves as an attribute on ``report`` (via
    :func:`operator.attrgetter`); remaining segments walk the nested payload
    — ``dict[key]`` when the current node is a mapping, ``getattr`` otherwise
    so dataclass-valued fields like ``kreiss`` work the same as the dict-
    valued layer payloads.  Returns ``float('inf')`` if any segment is
    missing, the layer was not run (``None``), or the final value cannot be
    coerced to ``float``.

    Examples
    --------
    >>> extract_field(report, "layer1.boundary_gv_err")
    >>> extract_field(report, "layer6.transient_growth_bound")
    >>> extract_field(report, "kreiss.witness_sigma_min")
    """
    segments = dotted_path.split(".")
    if not segments or not segments[0]:
        return float("inf")
    first, *rest = segments
    try:
        node = operator.attrgetter(first)(report)
    except AttributeError:
        return float("inf")
    if node is None:
        return float("inf")
    for seg in rest:
        if isinstance(node, dict):
            if seg not in node:
                return float("inf")
            node = node[seg]
        else:
            try:
                node = getattr(node, seg)
            except AttributeError:
                return float("inf")
        if node is None:
            return float("inf")
    try:
        return float(node)
    except (TypeError, ValueError):
        return float("inf")


# --- objective factory -------------------------------------------------------

_LAYER_PREFIX_RE = re.compile(r"^layer(\d+)\.")

# Dotted-path prefixes that alias to a populating layer.  ``kreiss`` is
# assigned in layer 2; extend this mapping if new aliased fields are added
# to :class:`StabilityReport`.
_FIELD_LAYER_ALIAS = {
    "kreiss": 2,
    "layer_bl42": 3,
    "non_normality": 6,
}


def _infer_max_layer(report_field: str) -> int | None:
    """Return the layer that populates ``report_field``, or ``None`` if the
    prefix is unrecognised.  Layer-prefixed fields (``layer1.*`` …
    ``layer8.*``) are parsed directly; aliased fields such as ``kreiss.*`` are
    mapped via :data:`_FIELD_LAYER_ALIAS`.
    """
    m = _LAYER_PREFIX_RE.match(report_field)
    if m:
        return int(m.group(1))
    head, _, _ = report_field.partition(".")
    return _FIELD_LAYER_ALIAS.get(head)


def make_objective(
    scheme: str,
    kernel: str,
    report_field: str,
    *,
    gate_layer: int | None = None,
    max_layer: int | None = None,
) -> Callable[[np.ndarray], float]:
    """Build a feasibility-gated objective ``f(x) -> float``.

    The returned closure converts a flat vector ``x`` to a kernel-specific
    ``params`` dict, runs :func:`brady2d_stability_score` in short-circuit
    mode up to ``max_layer``, and returns:

    - ``+inf`` if any layer at or before ``gate_layer`` failed (the
      feasibility cliff).
    - ``+inf`` if :func:`brady2d_stability_score` raised (extreme parameters
      can produce singular/ill-conditioned RBF systems).
    - :func:`extract_field` of ``report_field`` otherwise (which itself
      returns ``+inf`` when the requested dotted path is absent).

    Parameters
    ----------
    scheme, kernel, report_field
        Forwarded to :func:`brady2d_stability_score` and
        :func:`extract_field`.
    gate_layer
        Highest layer whose failure forces ``+inf`` (the feasibility gate).
        Defaults to ``max_layer - 1`` (floored at 0), so an objective living
        in layer N gates on all strictly-earlier layers — the natural
        feasibility gate for a cascade where layer N depends on layers
        ``< N`` passing.  A value of 0 means no gate (the objective layer
        is the only one run).
    max_layer
        Highest layer actually executed.  Defaults to the layer implied by
        ``report_field`` (``layer6.*`` → 6, ``kreiss.*`` → 2, …).  Raises
        ``ValueError`` if the resolved value is less than ``gate_layer`` —
        the optimiser cannot gate on layers it never runs.
    """
    if max_layer is None:
        inferred = _infer_max_layer(report_field)
        if inferred is None:
            raise ValueError(
                f"cannot infer max_layer from report_field={report_field!r}; "
                "pass max_layer explicitly"
            )
        max_layer = inferred
    if gate_layer is None:
        gate_layer = max(max_layer - 1, 0)
    if max_layer < gate_layer:
        raise ValueError(
            f"max_layer={max_layer} is less than gate_layer={gate_layer}; "
            "raise max_layer or lower gate_layer"
        )

    def objective(x: np.ndarray) -> float:
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
            return float("inf")
        if report.failed_layer is not None and report.failed_layer <= gate_layer:
            return float("inf")
        return extract_field(report, report_field)

    return objective


# --- drivers -----------------------------------------------------------------

def _probe_cobyqa_available() -> bool:
    """Probe whether ``scipy.optimize.minimize(method="COBYQA")`` is usable.

    COBYQA was added in scipy 1.14; older installations raise ``ValueError``
    when the method is requested.  The probe itself is a sub-millisecond call
    on a 1-variable identity objective.
    """
    try:
        minimize(
            lambda x: float(x[0] ** 2),
            x0=np.array([1.0]),
            method="COBYQA",
            options={"maxfev": 2},
        )
    except Exception:
        return False
    return True


_COBYQA_AVAILABLE = _probe_cobyqa_available()

_LOCAL_METHODS = ("Nelder-Mead", "COBYQA")


def run_scipy_local(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    *,
    method: str = "Nelder-Mead",
    max_evals: int = 200,
    tol: float = 1e-6,
) -> OptimizeResult:
    """Local optimization via ``scipy.optimize.minimize``.

    Wraps the user-supplied objective ``f`` in a recorder that appends
    ``(x.copy(), fval)`` to ``history`` on every evaluation — scipy's
    per-iteration ``callback`` only samples once per simplex step, which
    would miss most feasibility-cliff evaluations.

    ``method="Nelder-Mead"`` and ``method="COBYQA"`` are both supported.
    COBYQA (derivative-free trust region, scipy ≥ 1.14) handles 1-6D problems
    with feasibility cliffs better than Nelder-Mead in practice; see plan
    43.3b.  If COBYQA is requested on a scipy build that lacks it, a clear
    ``RuntimeError`` is raised rather than the opaque internal ``ValueError``.
    """
    if method not in _LOCAL_METHODS:
        raise ValueError(
            f"run_scipy_local: method must be one of {_LOCAL_METHODS}, got {method!r}"
        )
    if method == "COBYQA" and not _COBYQA_AVAILABLE:
        raise RuntimeError(
            f"COBYQA requires scipy >= 1.14; got {scipy.__version__}"
        )
    x0 = np.asarray(x0, dtype=float).ravel()
    if len(bounds) != x0.size:
        raise ValueError(
            f"run_scipy_local: bounds length {len(bounds)} does not match x0 size {x0.size}"
        )

    history: list[tuple[np.ndarray, float]] = []

    def _recorder(x: np.ndarray) -> float:
        fval = float(f(np.asarray(x, dtype=float)))
        history.append((np.asarray(x, dtype=float).copy(), fval))
        return fval

    if method == "Nelder-Mead":
        options = {
            "xatol": tol,
            "fatol": tol,
            "maxfev": max_evals,
            "adaptive": True,
        }
    else:  # COBYQA
        options = {
            "maxfev": max_evals,
            "feasibility_tol": tol,
        }

    t0 = time.perf_counter()
    result = minimize(
        _recorder,
        x0=x0,
        method=method,
        bounds=bounds,
        options=options,
    )
    compute_time = time.perf_counter() - t0

    best_x = np.asarray(result.x, dtype=float).ravel()
    best_objective = float(result.fun)
    converged = bool(result.success) and np.isfinite(best_objective)

    # ``run_scipy_local`` is kernel-agnostic — it receives a black-box ``f``
    # and cannot map ``best_x`` back to a kernel-specific params dict.
    # Higher-level drivers (``multi_start_optimize``, ``run_staged_optimize``)
    # own the kernel and use ``dataclasses.replace`` to fill ``best_params``.
    return OptimizeResult(
        best_params={},
        best_x=best_x,
        best_objective=best_objective,
        best_report={},
        method=method,
        converged=converged,
        n_evals=int(getattr(result, "nfev", len(history))),
        compute_time=compute_time,
        history=history,
        extras={"scipy_message": str(getattr(result, "message", ""))},
    )


def multi_start_optimize(
    f: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    n_restarts: int = 10,
    *,
    method: str = "Nelder-Mead",
    seed: int = 0,
    max_evals: int = 200,
    tol: float = 1e-6,
) -> OptimizeResult:
    """Sobol-seeded multi-start wrapper around :func:`run_scipy_local`.

    Generates ``n_restarts`` starting points via
    :class:`scipy.stats.qmc.Sobol` scaled to ``bounds`` and runs
    :func:`run_scipy_local` from each.  Returns the :class:`OptimizeResult`
    whose ``best_objective`` is the smallest finite value across restarts,
    with ``history`` concatenated and ``n_evals`` summed.  ``compute_time`` is
    the total wall-clock across restarts.

    If every restart returns ``+inf`` (fully infeasible bounds), the last
    restart's record is returned with ``converged=False`` — preserving the
    attempted ``best_x`` and any diagnostic ``scipy_message`` from the final
    call.
    """
    if n_restarts < 1:
        raise ValueError(f"multi_start_optimize: n_restarts must be >= 1, got {n_restarts}")
    if not bounds:
        raise ValueError("multi_start_optimize: bounds must be non-empty")

    sampler = qmc.Sobol(d=len(bounds), seed=seed)
    unit_sample = sampler.random(n_restarts)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    x0s = qmc.scale(unit_sample, lo, hi)

    aggregated_history: list[tuple[np.ndarray, float]] = []
    total_evals = 0
    total_time = 0.0
    n_feasible = 0
    best: OptimizeResult | None = None
    last: OptimizeResult | None = None

    for i in range(n_restarts):
        r = run_scipy_local(
            f,
            x0=x0s[i],
            bounds=bounds,
            method=method,
            max_evals=max_evals,
            tol=tol,
        )
        aggregated_history.extend(r.history)
        total_evals += r.n_evals
        total_time += r.compute_time
        last = r
        if np.isfinite(r.best_objective):
            n_feasible += 1
            if best is None or r.best_objective < best.best_objective:
                best = r

    chosen = best if best is not None else last
    assert chosen is not None  # n_restarts >= 1 guarantees at least one run
    converged = best is not None and chosen.converged

    return OptimizeResult(
        best_params={},
        best_x=np.asarray(chosen.best_x, dtype=float).copy(),
        best_objective=float(chosen.best_objective),
        best_report={},
        method="multi-start",
        converged=converged,
        n_evals=total_evals,
        compute_time=total_time,
        history=aggregated_history,
        extras={
            "inner_method": method,
            "n_restarts": n_restarts,
            "seed": seed,
            "n_feasible_restarts": n_feasible,
        },
    )


def run_scipy_shgo(
    f: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    *,
    n: int = 100,
    iters: int = 3,
) -> OptimizeResult:
    """Global optimization via ``scipy.optimize.shgo``.

    Simplicial homology global optimization is a deterministic global
    optimizer that constructs a simplicial complex over ``bounds`` and
    polishes each local basin with a scipy local minimizer (Nelder-Mead here,
    to match the rest of plan 43's derivative-free stack).

    The returned :class:`OptimizeResult` carries the count of distinct local
    minima SHGO discovered in ``extras["n_local_minima"]`` and the local
    minima table (``[(x, f)]``) in ``extras["local_minima"]``.  If the whole
    domain is infeasible (``+inf`` everywhere) scipy returns ``x=None``; in
    that case we return a non-finite, non-converged record so callers can
    detect the failure without special-casing an ``AttributeError``.
    """
    if not bounds:
        raise ValueError("run_scipy_shgo: bounds must be non-empty")

    history: list[tuple[np.ndarray, float]] = []

    def _recorder(x: np.ndarray) -> float:
        fval = float(f(np.asarray(x, dtype=float)))
        history.append((np.asarray(x, dtype=float).copy(), fval))
        return fval

    t0 = time.perf_counter()
    result = shgo(
        _recorder,
        bounds=bounds,
        n=n,
        iters=iters,
        minimizer_kwargs={"method": "Nelder-Mead"},
    )
    compute_time = time.perf_counter() - t0

    xl = getattr(result, "xl", None)
    funl = getattr(result, "funl", None)
    if xl is None or len(xl) == 0:
        local_minima: list[tuple[np.ndarray, float]] = []
    else:
        local_minima = [
            (np.asarray(x, dtype=float).copy(), float(fv))
            for x, fv in zip(xl, funl)
        ]
    n_local_minima = len(local_minima)

    if result.x is None or not np.isfinite(float(result.fun) if result.fun is not None else float("inf")):
        # Fully infeasible domain: preserve the attempted bound midpoint as
        # best_x so callers see something sensible, but mark non-converged.
        fallback_x = np.array([0.5 * (lo + hi) for (lo, hi) in bounds], dtype=float)
        return OptimizeResult(
            best_params={},
            best_x=fallback_x,
            best_objective=float("inf"),
            best_report={},
            method="SHGO",
            converged=False,
            n_evals=int(getattr(result, "nfev", len(history))),
            compute_time=compute_time,
            history=history,
            extras={
                "n_local_minima": n_local_minima,
                "local_minima": local_minima,
                "scipy_message": str(getattr(result, "message", "")),
            },
        )

    best_x = np.asarray(result.x, dtype=float).ravel()
    best_objective = float(result.fun)
    converged = bool(result.success) and np.isfinite(best_objective)

    return OptimizeResult(
        best_params={},
        best_x=best_x,
        best_objective=best_objective,
        best_report={},
        method="SHGO",
        converged=converged,
        n_evals=int(getattr(result, "nfev", len(history))),
        compute_time=compute_time,
        history=history,
        extras={
            "n_local_minima": n_local_minima,
            "local_minima": local_minima,
            "scipy_message": str(getattr(result, "message", "")),
        },
    )


def run_scipy_de(
    f: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    *,
    popsize: int = 15,
    maxiter: int = 100,
    seed: int = 0,
    strategy: str = "best1bin",
) -> OptimizeResult:
    """Global optimization via ``scipy.optimize.differential_evolution``.

    Population-based global search with Sobol initialization and a final
    L-BFGS-B polish (``polish=True``; scipy's documented default for a bounded,
    unconstrained problem — it falls back to ``trust-constr`` when constraints
    are supplied, which we do not do today).  Suited to 4-6D landscapes where
    SHGO's simplicial decomposition becomes expensive.

    The objective is wrapped in a recorder so every evaluation — including
    feasibility-cliff rejections — shows up in ``history``.  ``n_evals`` is
    taken from ``result.nfev`` (scipy's internal counter), which includes the
    polish pass; fully-infeasible domains (``+inf`` everywhere) are surfaced
    as ``best_objective=+inf`` with ``converged=False``.
    """
    if not bounds:
        raise ValueError("run_scipy_de: bounds must be non-empty")
    if popsize < 1:
        raise ValueError(f"run_scipy_de: popsize must be >= 1, got {popsize}")
    if maxiter < 1:
        raise ValueError(f"run_scipy_de: maxiter must be >= 1, got {maxiter}")

    history: list[tuple[np.ndarray, float]] = []

    def _recorder(x: np.ndarray) -> float:
        fval = float(f(np.asarray(x, dtype=float)))
        history.append((np.asarray(x, dtype=float).copy(), fval))
        return fval

    t0 = time.perf_counter()
    result = differential_evolution(
        _recorder,
        bounds=bounds,
        popsize=popsize,
        maxiter=maxiter,
        seed=seed,
        strategy=strategy,
        tol=1e-7,
        init="sobol",
        polish=True,
    )
    compute_time = time.perf_counter() - t0

    best_x = np.asarray(result.x, dtype=float).ravel()
    best_objective = float(result.fun)
    converged = bool(result.success) and np.isfinite(best_objective)

    return OptimizeResult(
        best_params={},
        best_x=best_x,
        best_objective=best_objective,
        best_report={},
        method="DE",
        converged=converged,
        n_evals=int(getattr(result, "nfev", len(history))),
        compute_time=compute_time,
        history=history,
        extras={
            "popsize": popsize,
            "maxiter": maxiter,
            "seed": seed,
            "strategy": strategy,
            "scipy_message": str(getattr(result, "message", "")),
        },
    )


def _report_to_dict(report) -> dict[str, Any]:
    """Serialise a :class:`StabilityReport` to a JSON-friendly dict.

    Mirrors ``sweeps/brady2d_sweep._report_to_dict`` so staged/optimize output
    can live in ``known_values.json`` beside the sweep records (plan 43.8a).
    Duplicated deliberately: ``stencil_gen`` should not take a dependency on
    ``sweeps``.
    """
    out: dict[str, Any] = {
        "overall_verdict": getattr(report, "overall_verdict", "unknown"),
        "failed_layer": getattr(report, "failed_layer", None),
        "failed_reason": getattr(report, "failed_reason", ""),
        "compute_time": float(getattr(report, "compute_time", 0.0)),
    }
    if getattr(report, "layer1", None) is not None:
        out["layer1"] = {
            k: float(v) for k, v in report.layer1.items() if isinstance(v, (int, float))
        }
    if getattr(report, "layer2", None) is not None:
        out["layer2"] = {"is_stable": bool(report.layer2.is_stable)}
    if getattr(report, "layer3", None) is not None:
        out["layer3"] = {"max_stab_eig": float(report.layer3["max_stab_eig"])}
    if getattr(report, "layer_bl42", None) is not None:
        out["layer_bl42"] = {
            k: float(v)
            for k, v in report.layer_bl42.items()
            if isinstance(v, (int, float))
        }
    if getattr(report, "layer4", None) is not None:
        out["layer4"] = {"max_local_gv_error": float(report.layer4["max_local_gv_error"])}
    if getattr(report, "layer5", None) is not None:
        out["layer5"] = {"max_aligned_error": float(report.layer5["max_aligned_error"])}
    if getattr(report, "layer6", None) is not None:
        out["layer6"] = {
            "spectral_abscissa": float(report.layer6.get("spectral_abscissa", float("nan"))),
            "transient_growth_bound": float(
                report.layer6.get("transient_growth_bound", float("nan"))
            ),
        }
    if getattr(report, "layer7", None) is not None:
        out["layer7"] = {
            "max_spectral_abscissa": float(report.layer7["max_spectral_abscissa"]),
        }
    if getattr(report, "layer8", None) is not None:
        out["layer8"] = {
            "final_linf": float(report.layer8["final_linf"]),
            "stable": bool(report.layer8["stable"]),
            "wall_time_s": float(report.layer8.get("wall_time_s", float("nan"))),
        }
    if getattr(report, "non_normality", None) is not None:
        out["non_normality"] = dataclasses.asdict(report.non_normality)
    if getattr(report, "kreiss", None) is not None:
        kr_dict = dataclasses.asdict(report.kreiss)
        kr_dict.pop("sigma_min_field", None)
        kr_dict.pop("s_grid", None)
        out["kreiss"] = kr_dict
    return out


def _top_k_candidates(
    history: list[tuple[np.ndarray, float]],
    top_k: int,
    *,
    decimals: int = 6,
) -> list[tuple[np.ndarray, float]]:
    """Deduplicate ``history`` by rounding ``x`` to ``decimals`` places and
    return up to ``top_k`` finite entries ordered by ascending objective.

    The rounded tuple is the dedup key; the representative stored per key is
    the entry with the smallest objective (which matches our minimization
    semantics and avoids re-validating two syntactically-identical points).
    """
    best_per_key: dict[tuple[float, ...], tuple[np.ndarray, float]] = {}
    for x, fv in history:
        if not np.isfinite(fv):
            continue
        x_arr = np.asarray(x, dtype=float).ravel()
        key = tuple(np.round(x_arr, decimals).tolist())
        prev = best_per_key.get(key)
        if prev is None or fv < prev[1]:
            best_per_key[key] = (x_arr.copy(), float(fv))
    ordered = sorted(best_per_key.values(), key=lambda xy: xy[1])
    return ordered[:top_k]


_E4_CLASSICAL_ALPHA1_CPP_FLOOR = 197.0 / 288.0


def _record_cpp_cutcell_diagnostic(
    extras: dict, scheme: str, kernel: str, best_x: np.ndarray | None
) -> None:
    """Populate the L8 cut-cell ``α₁ ≥ 197/288`` diagnostic flag (plan 43.9b-r1).

    The C++ ``E4_1.cpp`` stencil enforces ``α₁ ≥ 197/288`` to keep the
    cut-cell psi denominator non-zero.  The Python analytical pipeline
    (L1–L7) does not enforce this floor (see plan 43.9a), so an analytical
    winner with ``α₁ < 197/288`` is expected and not an error.  The flag is
    purely informational for downstream plan-43.10 L8 wiring.
    """
    if scheme != "E4" or kernel != "classical":
        return
    if best_x is None:
        return
    x_arr = np.asarray(best_x, dtype=float).ravel()
    if x_arr.size < 2 or not np.isfinite(x_arr[1]):
        return
    extras["cpp_cutcell_violates_197_288"] = bool(
        x_arr[1] < _E4_CLASSICAL_ALPHA1_CPP_FLOOR
    )


def run_staged_optimize(
    scheme: str,
    kernel: str,
    report_field: str,
    bounds: list[tuple[float, float]],
    *,
    inner_gate: int = 3,
    inner_max_layer: int = 3,
    validator_max_layer: int = 6,
    top_k: int = 5,
    method: str = "Nelder-Mead",
    n_restarts: int = 20,
    seed: int = 0,
    max_evals: int = 200,
    tol: float = 1e-6,
) -> OptimizeResult:
    """Cheap-inner + expensive-validator staged pipeline (plan 43.6a).

    Stage 1 — inner: build a feasibility-gated objective at
    ``gate_layer=inner_gate, max_layer=inner_max_layer`` and drive it with
    :func:`multi_start_optimize` (``n_restarts`` Sobol-seeded starts).

    Stage 2 — validator: take the ``top_k`` distinct feasible candidates from
    the inner ``history`` (deduplicated by rounding ``x`` to 6 decimals),
    re-run :func:`brady2d_stability_score` at
    ``max_layer=validator_max_layer``, and re-rank by ``report_field``.

    The returned :class:`OptimizeResult` reports the winner of the validator
    stage.  ``extras["stage"]`` is ``"validated"`` when the validator's top
    pick differs from the inner's top pick (in ``x`` under the same 6-decimal
    dedup), else ``"inner"``.  Diagnostics from the inner stage are forwarded
    under ``extras["inner_*"]`` and the full validator ranking is exposed as
    ``extras["validator_ranking"] = [(x, f_validator), ...]``.

    Raises
    ------
    ValueError
        If ``inner_max_layer < inner_gate`` or
        ``validator_max_layer < inner_max_layer`` — the validator stage must
        be at least as deep as the inner gate.  ``top_k < 1`` is also rejected.
    """
    if inner_max_layer < inner_gate:
        raise ValueError(
            f"run_staged_optimize: inner_max_layer={inner_max_layer} < "
            f"inner_gate={inner_gate}"
        )
    if validator_max_layer < inner_max_layer:
        raise ValueError(
            f"run_staged_optimize: validator_max_layer={validator_max_layer} < "
            f"inner_max_layer={inner_max_layer}"
        )
    if top_k < 1:
        raise ValueError(f"run_staged_optimize: top_k must be >= 1, got {top_k}")

    t0 = time.perf_counter()

    # --- Stage 1: cheap inner multi-start -----------------------------------
    # The inner objective short-circuits at inner_max_layer; make_objective
    # itself will raise ``ValueError`` if the report_field implies a layer
    # deeper than inner_max_layer, so wrap it and fall back to a pure L3
    # gate field when the caller's report_field is validator-only (e.g.
    # layer6.transient_growth_bound).  We use ``layer3.max_stab_eig`` as the
    # inner ranking field in that case — it is the canonical stability-margin
    # metric Brady-Livescu use for the inner short-circuit.
    inner_field = report_field
    inferred = _infer_max_layer(report_field)
    if inferred is not None and inferred > inner_max_layer:
        inner_field = "layer3.max_stab_eig"
    f_inner = make_objective(
        scheme,
        kernel,
        inner_field,
        gate_layer=inner_gate,
        max_layer=inner_max_layer,
    )
    inner_result = multi_start_optimize(
        f_inner,
        bounds=bounds,
        n_restarts=n_restarts,
        method=method,
        seed=seed,
        max_evals=max_evals,
        tol=tol,
    )

    candidates = _top_k_candidates(inner_result.history, top_k=top_k)

    # --- Stage 2: expensive validator re-ranking -----------------------------
    validator_ranking: list[tuple[np.ndarray, float, dict]] = []
    validator_evals = 0
    for x, _f_inner in candidates:
        validator_evals += 1
        try:
            params = params_from_vector(kernel, x)
            report = brady2d_stability_score(
                scheme,
                kernel,
                params,
                max_layer=validator_max_layer,
                short_circuit=True,
            )
        except Exception:
            validator_ranking.append((x.copy(), float("inf"), {}))
            continue
        if (
            report.failed_layer is not None
            and report.failed_layer <= inner_gate
        ):
            fv = float("inf")
        else:
            fv = extract_field(report, report_field)
        validator_ranking.append((x.copy(), float(fv), _report_to_dict(report)))

    validator_ranking.sort(key=lambda xyr: xyr[1])
    compute_time = time.perf_counter() - t0

    # --- Stage outcome -------------------------------------------------------
    if not validator_ranking or not np.isfinite(validator_ranking[0][1]):
        # Every candidate blew up at the deeper layer.  Surface the inner
        # result verbatim so the caller still has a diagnostic anchor.
        fallback_extras = {
            "stage": "inner",
            "validator_ranking": [(x, fv) for (x, fv, _r) in validator_ranking],
            "inner_method": inner_result.extras.get("inner_method", method),
            "inner_n_restarts": inner_result.extras.get("n_restarts", n_restarts),
            "inner_seed": inner_result.extras.get("seed", seed),
            "inner_n_feasible_restarts": inner_result.extras.get(
                "n_feasible_restarts", 0
            ),
            "inner_field": inner_field,
            "inner_best_objective": inner_result.best_objective,
            "inner_best_x": np.asarray(inner_result.best_x, dtype=float).copy(),
            "validator_max_layer": validator_max_layer,
            "inner_max_layer": inner_max_layer,
        }
        _record_cpp_cutcell_diagnostic(
            fallback_extras, scheme, kernel, inner_result.best_x
        )
        fallback = replace(
            inner_result,
            method="staged",
            converged=False,
            best_params=(
                params_from_vector(kernel, inner_result.best_x)
                if np.isfinite(inner_result.best_objective)
                else {}
            ),
            best_report={},
            compute_time=compute_time,
            n_evals=inner_result.n_evals + validator_evals,
            extras=fallback_extras,
        )
        return fallback

    best_x, best_obj, best_report = validator_ranking[0]

    # Did the validator re-order the winner?  Compare against the inner's
    # top-ranked feasible candidate (first entry of ``candidates``) under the
    # same 6-decimal rounding that _top_k_candidates uses.
    inner_top_key = (
        tuple(np.round(candidates[0][0], 6).tolist()) if candidates else None
    )
    validated_top_key = tuple(np.round(best_x, 6).tolist())
    stage = "validated" if inner_top_key != validated_top_key else "inner"

    success_extras = {
        "stage": stage,
        "validator_ranking": [(x, fv) for (x, fv, _r) in validator_ranking],
        "inner_method": inner_result.extras.get("inner_method", method),
        "inner_n_restarts": inner_result.extras.get("n_restarts", n_restarts),
        "inner_seed": inner_result.extras.get("seed", seed),
        "inner_n_feasible_restarts": inner_result.extras.get(
            "n_feasible_restarts", 0
        ),
        "inner_field": inner_field,
        "inner_best_objective": inner_result.best_objective,
        "inner_best_x": np.asarray(inner_result.best_x, dtype=float).copy(),
        "validator_max_layer": validator_max_layer,
        "inner_max_layer": inner_max_layer,
    }
    _record_cpp_cutcell_diagnostic(success_extras, scheme, kernel, best_x)
    return OptimizeResult(
        best_params=params_from_vector(kernel, best_x),
        best_x=np.asarray(best_x, dtype=float).copy(),
        best_objective=float(best_obj),
        best_report=best_report,
        method="staged",
        converged=bool(np.isfinite(best_obj)),
        n_evals=inner_result.n_evals + validator_evals,
        compute_time=compute_time,
        history=list(inner_result.history),
        extras=success_extras,
    )
