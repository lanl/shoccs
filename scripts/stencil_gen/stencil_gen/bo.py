"""Multi-fidelity Bayesian optimization over the cascade.

Implements plan 47: replace the hand-coded ``run_staged_optimize`` cheap-inner
+ expensive-validator heuristic with a principled multi-fidelity Bayesian
optimizer that uses a Gaussian-process surrogate over the cascade's discrete
fidelity levels and a cost-aware acquisition function.

Algorithm
---------

The optimizer chooses ``(x, m)`` jointly to maximize expected information gain
at the high-fidelity target per second of wall time.  The GP surrogate uses an
Intrinsic Coregionalization Model (ICM) kernel to learn correlations between
cascade layers from data — necessary because the cascade's L3 ↔ L3r pair tests
different physics (1D periodic advection vs. reflecting BCs), so a single
Kennedy-O'Hagan autoregressive ladder is inappropriate (see
``docs/handoff/scientific_findings.md`` finding #1).

References
----------

- Wu, J., Toscano-Palmerin, S., Frazier, P. I., & Wilson, A. G. (2020).
  *Practical Multi-fidelity Bayesian Optimization for Hyperparameter Tuning*.
  https://arxiv.org/abs/1903.04703
- BoTorch tutorial: discrete multi-fidelity BO.
  https://botorch.org/docs/tutorials/discrete_multi_fidelity_bo/
- BoTorch tutorial: cost-aware Bayesian optimization.
  https://botorch.org/docs/tutorials/cost_aware_bayesian_optimization/

This is a skeleton module — the dataclasses, factory, GP, cost model, DOE,
acquisition, and ``run_mfbo`` driver are added in subsequent items of plan 47
(47.1 onward).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
import botorch  # noqa: F401  # used in subsequent items
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.transforms import Standardize
from botorch.optim.optimize import optimize_acqf_mixed
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from stencil_gen.brady2d_stability import brady2d_stability_score
from stencil_gen.optimizer import (  # noqa: F401  # reused in subsequent items
    DEFAULT_BOUNDS,
    _FIELD_LAYER_ALIAS,
    _infer_max_layer,
    _report_to_dict,
    extract_field,
    params_from_vector,
    vector_from_params,
)


# Finite sentinel used when a multi-fidelity evaluation is infeasible (gate
# trip, shape mismatch, or ``brady2d_stability_score`` exception).  qMFKG /
# MES break on ``+inf`` (Cholesky failure during fantasy sampling); a large
# finite value keeps the GP fit well-conditioned.  Sentinel rows are filtered
# out of the training tensors before each GP refit.
_BO_SENTINEL: float = 1e12


class _SkipGuard(Exception):
    """Internal control-flow sentinel: skip the variance guard for this iter.

    Raised inside the guard's ``try`` block (e.g. when fewer than 2 HF
    training observations exist, so HF-only spread is undefined) and caught
    locally so the loop continues to acquisition.  Distinct from the bare
    ``except Exception`` that catches *real* posterior() failures.
    """


@dataclass(frozen=True)
class BOEval:
    """A single multi-fidelity evaluation record.

    One row of the BO loop's evaluation history: the design vector ``x``, the
    fidelity ``m`` it was evaluated at, the resulting objective value, the
    measured wall time (for empirical cost calibration), and the serialised
    :class:`StabilityReport`.

    Attributes
    ----------
    x : np.ndarray
        Flat design vector of shape ``(d,)``, dtype ``float64``.  The
        optimiser's native representation — convert to a kernel-specific
        ``params`` dict via :func:`stencil_gen.optimizer.params_from_vector`.
    params : dict
        Kernel-specific parameter dict at ``x``.  Redundant with ``x`` but
        included so downstream code can consume either representation without
        carrying the kernel through.
    fidelity : int
        Cascade layer index this evaluation ran at (e.g. ``1``, ``3``, ``7``).
        This is the *external* layer number used by
        :func:`brady2d_stability_score`, not the internal contiguous fidelity
        index used by the GP/acquisition.
    value : float
        Extracted objective value at this fidelity.  Equals
        :data:`_BO_SENTINEL` if the evaluation was infeasible.
    wall_time : float
        Measured per-eval seconds (``time.perf_counter`` delta around the
        :func:`brady2d_stability_score` call).  Always positive, even on the
        sentinel path.
    report : dict
        Serialised :class:`StabilityReport` (produced by ``_report_to_dict``).
        Contains ``{"error": str(exc)}`` if the evaluation produced no
        feasible report.
    """

    x: np.ndarray
    params: dict
    fidelity: int
    value: float
    wall_time: float
    report: dict


@dataclass(frozen=True)
class BOResult:
    """Frozen record of a single multi-fidelity Bayesian optimisation run.

    Mirrors :class:`stencil_gen.pareto.ParetoResult` in spirit: an immutable
    summary plus enough raw data (full eval history, GP hyperparameters, cost
    table) to reproduce the run's recommendation off-line.

    Attributes
    ----------
    best_x : np.ndarray
        Recommended design vector at the high-fidelity target.  Selected via
        ``argmin_x μ_n(x, m=hf)`` on a Sobol' grid (posterior mean — standard
        for noisy / multi-fidelity GPs), then re-evaluated at HF to populate
        :attr:`best_objective` from real data.
    best_params : dict
        Kernel-specific parameter dict at :attr:`best_x`.
    best_objective : float
        HF objective value at :attr:`best_x` from a final real evaluation
        (NOT the GP posterior mean, which can disagree under model misspec).
    best_report : dict
        Full serialised :class:`StabilityReport` at :attr:`best_x` at HF.
    method : str
        Driver name, e.g. ``"BoTorch-qMFKG"`` (or fallback name like
        ``"BoTorch-qMFMES"`` if KG diagnostics show degeneracy).
    scheme : str
        Scheme identifier forwarded to :func:`brady2d_stability_score`.
    kernel : str
        Kernel identifier forwarded to :func:`brady2d_stability_score`.
    bounds : tuple[tuple[float, float], ...]
        Parameter bounds used for the run, one ``(lo, hi)`` pair per variable.
    fidelity_levels : tuple[int, ...]
        Sorted external layer indices in ascending cost order, e.g.
        ``(1, 3, 7)``.  Sorted so ``[-1]`` is always the HF level.
    hf_level : int
        ``max(fidelity_levels)``.  The optimiser's target.
    report_fields_by_layer : dict[int, str]
        Mapping ``layer index → dotted path``, e.g.
        ``{1: "layer1.boundary_gv_err", 7: "layer7.max_spectral_abscissa"}``.
        The HF layer's field is the optimisation target.
    cost_model : dict[int, float]
        The actual cost table used (with floor applied).  Keyed by external
        layer index, values in seconds.
    n_evals_per_fidelity : dict[int, int]
        Count of evaluations at each fidelity (initial design + acquisition
        steps + final HF re-evaluation at ``best_x``).  Keys match
        :attr:`fidelity_levels`.
    wall_time_per_fidelity : dict[int, float]
        Cumulative measured wall time at each fidelity, in seconds.
    total_compute_time : float
        Total wall-clock seconds for the run (init + GP fits + acquisition
        optimisation + objective evaluations + final HF re-evaluation).
    eval_history : tuple[BOEval, ...]
        Full per-eval log, in chronological order.  Length equals the total
        number of evaluations.
    hf_eval_history : tuple[BOEval, ...]
        Filter of :attr:`eval_history` to ``fidelity == hf_level`` only.
        Used to produce the convergence trace ``best_observed_hf_so_far``.
    gp_hyperparameters : dict
        Final GP state at convergence: lengthscale, outputscale, noise, and
        the ICM ``B = W Wᵀ + diag(κ)`` coregionalization matrix (extracted
        from ``model.covar_module.state_dict()``).  Empty dict if the GP
        never fit (e.g. all initial evals returned sentinel).
    seed : int
        RNG seed supplied to :func:`run_mfbo`.  Setting the same seed
        reproduces :attr:`best_x` to within ``1e-6``.
    converged : bool
        ``True`` if the run terminated by variance / stagnation guard;
        ``False`` if it hit budget.  Always ``False`` on error termination.
    stop_reason : str
        One of ``"budget"``, ``"variance"``, ``"stagnation"``, ``"error"``.
    extras : dict
        Free-form additional fields (e.g. ``n_sentinel_per_fidelity`` —
        per-fidelity sentinel occurrence count, treatment-agnostic, set
        when ``clamp_sentinel_rows=True``; ``n_sentinel_filtered`` —
        global tally, set when ``clamp_sentinel_rows=False``;
        ``baseline`` :class:`OptimizeResult`; ``cpp_validation`` payload).
    """

    best_x: np.ndarray
    best_params: dict
    best_objective: float
    best_report: dict
    method: str
    scheme: str
    kernel: str
    bounds: tuple[tuple[float, float], ...]
    fidelity_levels: tuple[int, ...]
    hf_level: int
    report_fields_by_layer: dict[int, str]
    cost_model: dict[int, float]
    n_evals_per_fidelity: dict[int, int]
    wall_time_per_fidelity: dict[int, float]
    total_compute_time: float
    eval_history: tuple[BOEval, ...]
    hf_eval_history: tuple[BOEval, ...]
    gp_hyperparameters: dict
    seed: int
    converged: bool
    stop_reason: str
    extras: dict


# --- multi-fidelity objective factory ----------------------------------------


def make_multi_fidelity_objective(
    scheme: str,
    kernel: str,
    report_fields_by_layer: dict[int, str],
    *,
    gate_layer: int | None = None,
) -> Callable[[np.ndarray, int], tuple[float, float, dict]]:
    """Build a multi-fidelity objective ``f(x, m) -> (value, wall_time, report)``.

    Mirrors :func:`stencil_gen.optimizer.make_objective` but routes through a
    per-fidelity field selection and returns the wall-time + serialised report
    alongside the scalar value, so the BO loop can record per-eval cost without
    a side channel.

    Parameters
    ----------
    scheme, kernel
        Forwarded to :func:`brady2d_stability_score`.
    report_fields_by_layer
        Mapping ``{layer_index: dotted_field_path}``.  ``max(...)`` is the HF
        target; the HF field is the optimisation objective.  Cheaper layers'
        fields are surrogates that the GP correlates with the HF objective via
        the ICM coregionalization matrix.
    gate_layer
        Highest layer whose failure forces the sentinel value.  Defaults to
        ``max(min(layers) - 1, 0)`` — only layers strictly *cheaper* than the
        cheapest fidelity in ``report_fields_by_layer`` gate; the cheapest
        fidelity itself is always a usable result.  Pass ``0`` to disable
        gating entirely.

    Returns
    -------
    Callable[[np.ndarray, int], tuple[float, float, dict]]
        Closure ``f(x, m)``.  On any of:

        - ``m`` not in ``report_fields_by_layer``,
        - shape mismatch in :func:`params_from_vector`,
        - exception from :func:`brady2d_stability_score`,
        - gate trip (failed layer ≤ ``gate_layer``),

        returns ``(_BO_SENTINEL, measured_wall_time, {"error": str(...)})``.
        On success, returns
        ``(extract_field(report, field_at_m), wall_time, _report_to_dict(report))``.

    Raises
    ------
    ValueError
        At factory time, if any field's :func:`_infer_max_layer` exceeds the
        layer it is keyed under (you cannot extract ``layer7.*`` from an
        ``m=3`` run).  Also raised when ``report_fields_by_layer`` is empty.
    """
    if not report_fields_by_layer:
        raise ValueError("report_fields_by_layer must not be empty")
    for layer, field in report_fields_by_layer.items():
        inferred = _infer_max_layer(field)
        if inferred is not None and inferred > layer:
            raise ValueError(
                f"field {field!r} requires max_layer={inferred} but is keyed "
                f"under layer={layer}; cannot extract a field from a layer "
                "that is not run"
            )
    layers_sorted = sorted(report_fields_by_layer)
    if gate_layer is None:
        gate_layer = max(layers_sorted[0] - 1, 0)

    def objective(x: np.ndarray, m: int) -> tuple[float, float, dict]:
        if m not in report_fields_by_layer:
            return (
                _BO_SENTINEL,
                0.0,
                {"error": f"unknown fidelity m={m}"},
            )
        t0 = time.perf_counter()
        try:
            params = params_from_vector(kernel, x)
            report = brady2d_stability_score(
                scheme,
                kernel,
                params,
                max_layer=m,
                short_circuit=True,
            )
        except Exception as exc:
            return (
                _BO_SENTINEL,
                time.perf_counter() - t0,
                {"error": str(exc)},
            )
        wall_time = time.perf_counter() - t0
        if (
            report.failed_layer is not None
            and report.failed_layer <= gate_layer
        ):
            return (
                _BO_SENTINEL,
                wall_time,
                _report_to_dict(report),
            )
        value = extract_field(report, report_fields_by_layer[m])
        if not np.isfinite(value):
            return (
                _BO_SENTINEL,
                wall_time,
                _report_to_dict(report),
            )
        return (float(value), wall_time, _report_to_dict(report))

    return objective


# --- multi-fidelity GP surrogate ---------------------------------------------


def build_mf_gp(
    train_X: np.ndarray | torch.Tensor,
    train_Y: np.ndarray | torch.Tensor,
    fidelity_dim: int,
    num_fidelities: int,
    *,
    rank: int = 2,
) -> MultiTaskGP:
    """Build and fit an ICM-style multi-fidelity GP surrogate.

    The surrogate is a :class:`MultiTaskGP` — BoTorch's purpose-built ICM
    model — with a Matern-5/2 ARD data kernel and an Intrinsic
    Coregionalization Model (ICM) parameterisation of the layer-pair
    covariance ``B = W Wᵀ + diag(κ)``, with ``W`` of shape
    ``(num_fidelities, rank)``, learned end-to-end via marginal-likelihood
    optimisation.  This lets the data report the actual layer-pair
    correlations rather than baking in a Kennedy-O'Hagan refinement chain
    that the cascade does not satisfy (L3 ↔ L3r test different physics —
    see ``docs/handoff/scientific_findings.md`` finding #1).

    Parameters
    ----------
    train_X
        Training inputs of shape ``(N, d + 1)`` where the column at
        ``fidelity_dim`` holds integer-valued fidelity indices in
        ``{0, ..., num_fidelities - 1}``.  Accepts NumPy or torch; converted
        to ``torch.float64`` internally.
    train_Y
        Training targets of shape ``(N,)`` or ``(N, 1)``.  Sentinel rows must
        be filtered upstream — the GP only fits on finite-value rows.
    fidelity_dim
        Column index of the fidelity feature in ``train_X``.  Conventionally
        the last column (i.e. ``train_X.shape[-1] - 1``).
    num_fidelities
        Number of distinct fidelity levels (informational; the actual task
        count is inferred from values present in the fidelity column).
    rank
        Rank of the ICM coregionalization factor ``W``.  ``rank=2`` is a good
        default for 3–5 fidelities — large enough to capture non-trivial
        layer-pair correlations, small enough to remain identifiable from
        modest training data.

    Returns
    -------
    MultiTaskGP
        Fitted GP.  Hyperparameters can be inspected via:

        - ``model.covar_module.kernels[0].lengthscale`` (Matern ARD).
        - ``model.covar_module.kernels[1].covar_factor`` (W of the ICM
          matrix).
        - ``model.covar_module.kernels[1].var`` (diagonal κ of the ICM
          matrix).
        - ``model.likelihood.noise``.

    Notes
    -----
    The plan body cited :class:`SingleTaskMultiFidelityGP` as the wrapper,
    but that class always composes the user-supplied ``covar_module`` with a
    fixed :class:`LinearTruncatedFidelityKernel` (the AR1 kernel we
    explicitly want to avoid) or :class:`ExponentialDecayKernel`, so a
    custom ICM kernel is not respected.  Hand-composing ``MaternKernel *
    IndexKernel`` on a regular :class:`SingleTaskGP` worked but proved
    fragile: ``fit_gpytorch_mll`` failed on ~70% of small noise-free
    datasets due to NotPSDError during Cholesky factorisation.
    :class:`MultiTaskGP` is BoTorch's purpose-built ICM model — same kernel
    structure (Matern-on-data × IndexKernel-on-task), but with engineered
    parameter initialisation and PSD-stable parameterisation that fits
    reliably.  The MF-aware ``project`` helper that
    :class:`SingleTaskMultiFidelityGP` adds beyond a plain GP is supplied
    directly to ``qMultiFidelityKnowledgeGradient`` in 47.3a.

    Outputs are standardised via :class:`Standardize`; without it the
    marginal-likelihood optimiser fails on raw cascade scales
    (``max_stab_eig`` is ~1e-12 while ``boundary_gv_err`` is ~1e-2 — five
    orders of magnitude apart).
    """
    if num_fidelities < 1:
        raise ValueError(f"num_fidelities must be ≥ 1, got {num_fidelities}")
    if rank < 1:
        raise ValueError(f"rank must be ≥ 1, got {rank}")

    X = torch.as_tensor(train_X, dtype=torch.float64)
    Y = torch.as_tensor(train_Y, dtype=torch.float64)
    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)
    if X.ndim != 2:
        raise ValueError(f"train_X must be 2D, got shape {tuple(X.shape)}")
    if Y.shape[0] != X.shape[0]:
        raise ValueError(
            f"train_X has {X.shape[0]} rows but train_Y has {Y.shape[0]}"
        )
    n_cols = X.shape[-1]
    if not (0 <= fidelity_dim < n_cols):
        raise ValueError(
            f"fidelity_dim={fidelity_dim} out of range for train_X with "
            f"{n_cols} columns"
        )

    n_data_dims = n_cols - 1
    data_kernel = MaternKernel(nu=2.5, ard_num_dims=n_data_dims)

    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-9))

    model = MultiTaskGP(
        train_X=X,
        train_Y=Y,
        task_feature=fidelity_dim,
        covar_module=data_kernel,
        likelihood=likelihood,
        rank=rank,
        all_tasks=list(range(num_fidelities)),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


# --- cost model + cost-aware utility -----------------------------------------


# Default per-layer wall-time costs (seconds) from plan 46 measurements.  Keys
# are *external* cascade layer indices; the contiguous internal fidelity index
# 0..K-1 used by the GP/acquisition is derived by sorting the keys ascending.
# L3r is keyed at external index 5 by plan 47.4a convention (it sits between
# L3=3 and L6=6 in cost) — even though it shares ``max_layer=3`` with L3
# inside :func:`brady2d_stability_score`, the BO module treats it as a
# distinct fidelity so the ICM kernel can learn an L3-vs-L3r task correlation.
# The CLI (47.4a) translates this synthetic ``5`` to the ``layer_bl42`` field
# name when invoking the cascade.
DEFAULT_COST_TABLE: dict[int, float] = {
    1: 0.076,  # L1: GV dispersion (interior + boundary)
    3: 0.038,  # L3: 1D advection eigenvalue
    5: 0.486,  # L3r: BL §4.2 reflecting-hyperbolic spectrum
    6: 0.846,  # L6: non-normality on 1D operator
    7: 1.434,  # L7: full 2D varying-coefficient spectral abscissa
}


# Cost floor as a fraction of the most expensive layer's cost.  Caps the
# acquisition's preference for the cheapest layer when the cost ratio is so
# extreme (here ``c(L7)/c(L3) ≈ 38``) that the cost-aware utility would
# otherwise keep querying the cheapest layer indefinitely, even after the GP
# has learned that layer is uncorrelated with HF.
_DEFAULT_COST_FLOOR_RATIO: float = 0.05


# 47.3i: adaptive HF cost floor — predicate threshold for the
# "HF posterior uncertain" check.  ``var_hf_grid > τ * spread_hf**2`` fires
# whenever the maximum posterior variance across a Sobol' grid at HF is at
# least ``τ`` of the squared HF Y-range.  Distinct from (and orders of
# magnitude looser than) the variance-guard threshold ``1e-6 * spread_hf**2``
# at the same Y scale: the adaptive floor is meant to lift effective HF cost
# during exploration, well before the variance guard would consider the
# incumbent converged.
_ADAPTIVE_HF_FLOOR_TAU: float = 0.01


# 47.3k.1: floor on the default for ``run_mfbo``'s ``min_acquisition_iterations``
# kwarg.  The default resolves to ``max(_MIN_ACQ_ITERATIONS_FLOOR, K)`` where
# ``K`` is the number of fidelity levels.  Raised from ``5`` to ``15`` after
# the AugmentedBranin sweep documented in plan 47.3k showed ``min_acq=20``
# lifts the empirical best from ``3.55`` to ``1.25`` (3-seed, 99 s budget-
# exit) — the GP-uniform-collapse failure mode that the 47.3d combined
# absolute+relative variance-guard criterion only partially defends against
# needs more acquisition headroom on smooth synthetic objectives.  Exposed
# as a module-level constant so 47.3k.4's empirical re-measurement can flip
# the default without touching :func:`run_mfbo`'s body.
_MIN_ACQ_ITERATIONS_FLOOR: int = 15


def apply_cost_floor(
    cost_table: dict[int, float],
    *,
    floor_ratio: float = _DEFAULT_COST_FLOOR_RATIO,
) -> dict[int, float]:
    """Return a copy of *cost_table* with a per-entry cost floor applied.

    For each entry ``c(m)``, the floored cost is
    ``max(c(m), floor_ratio * max_n c(n))`` — any layer whose cost is below
    ``floor_ratio`` of the most expensive layer is lifted to that floor.
    Prevents qMFKG from over-exploiting the cheapest layer; see Wu et al. 2020
    §4.2 for the cost-weighted KG formulation that motivates the floor.

    Parameters
    ----------
    cost_table
        Mapping ``layer index → cost (seconds)``.  Caller's choice of layer
        indices; the function does not interpret them.
    floor_ratio
        Per-entry floor as a fraction of the most expensive layer's cost.
        Pass ``0.0`` to disable (not recommended).

    Returns
    -------
    dict[int, float]
        New dict with the same keys as *cost_table* and floored values.

    Raises
    ------
    ValueError
        If *cost_table* is empty or *floor_ratio* is negative.
    """
    if not cost_table:
        raise ValueError("cost_table must not be empty")
    if floor_ratio < 0:
        raise ValueError(f"floor_ratio must be ≥ 0, got {floor_ratio}")
    hf_cost = max(cost_table.values())
    floor = floor_ratio * hf_cost
    return {layer: max(cost, floor) for layer, cost in cost_table.items()}


def build_cost_model(
    cost_table: dict[int, float],
    fidelity_dim: int,
    *,
    floor_ratio: float = _DEFAULT_COST_FLOOR_RATIO,
) -> InverseCostWeightedUtility:
    """Build the inverse-cost-weighted utility for cost-aware MF acquisition.

    Wraps a step-function deterministic cost model in
    :class:`InverseCostWeightedUtility` so qMFKG (47.3a) weights expected
    information gain by ``1 / cost(m)``.  The deterministic model reads the
    *internal* contiguous fidelity index (integer-rounded) from column
    ``fidelity_dim`` of its input tensor and looks up the corresponding cost
    in a floored copy of *cost_table*.

    Parameters
    ----------
    cost_table
        Mapping ``external layer index → cost (seconds)``.  Sorted ascending
        to derive the internal contiguous index ``0..K-1``: e.g. for keys
        ``{1, 3, 5, 6, 7}``, internal index ``0`` ↔ layer 1, ``4`` ↔ layer 7.
    fidelity_dim
        Column index of the fidelity feature in the acquisition's ``X``
        tensor.  Conventionally the last column (``train_X.shape[-1] - 1``).
    floor_ratio
        Forwarded to :func:`apply_cost_floor`.  Default ``0.05``.

    Returns
    -------
    InverseCostWeightedUtility
        Utility with ``use_mean=True`` (the default; the deterministic cost
        model has no posterior variance, so the choice is moot — but ``True``
        matches the BoTorch discrete-MF tutorial).

    Raises
    ------
    ValueError
        If *cost_table* is empty, *floor_ratio* is negative, or
        *fidelity_dim* is negative.
    """
    if fidelity_dim < 0:
        raise ValueError(f"fidelity_dim must be ≥ 0, got {fidelity_dim}")
    floored = apply_cost_floor(cost_table, floor_ratio=floor_ratio)
    sorted_layers = sorted(floored)
    cost_lookup = torch.tensor(
        [floored[layer] for layer in sorted_layers],
        dtype=torch.float64,
    )
    n_layers = len(sorted_layers)

    def cost_fn(X: torch.Tensor) -> torch.Tensor:
        # ``X`` has shape ``(..., d + 1)``; the fidelity column holds integer
        # internal indices.  Round and clamp before lookup to defend against
        # NaN or out-of-range values from the acquisition optimiser.
        fid = X[..., fidelity_dim].round().long().clamp(0, n_layers - 1)
        lookup = cost_lookup.to(dtype=X.dtype, device=X.device)
        return lookup[fid].unsqueeze(-1)

    cost_model = GenericDeterministicModel(f=cost_fn, num_outputs=1)
    return InverseCostWeightedUtility(cost_model=cost_model, use_mean=True)


# --- initial design (DOE) ----------------------------------------------------


def build_initial_design(
    bounds: Sequence[tuple[float, float]],
    fidelity_levels: Sequence[int],
    *,
    n_init: int | None = None,
    hf_anchors: int = 3,
    mid_anchors: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a stratified Sobol' initial design for the BO loop.

    The DOE has three goals: (i) cover the design space well in the cheap
    fidelity (where evaluations are nearly free), (ii) seed enough HF data
    that the GP's posterior at the target fidelity is informative from
    iteration 0, and (iii) provide *paired* HF/cheap evaluations at the same
    ``x`` so the ICM coregionalization matrix ``B = W Wᵀ + diag(κ)`` is
    identifiable from data.  Without paired evaluations the marginal
    likelihood cannot pin down the off-diagonal task correlations (Wu et al.
    2020 §3.1; this is "Agent 2 pitfall #1" in the plan-47 design notes).

    The stratification: of ``n_init`` total evaluations, ``hf_anchors`` go to
    the HF level (paired with the first ``hf_anchors`` cheap-fidelity points
    at identical ``x``), ``mid_anchors`` go to the median-cost fidelity (at
    additional Sobol' draws), and the remaining ``n_init - hf_anchors -
    mid_anchors`` go to the cheapest fidelity.  With the defaults
    ``hf_anchors=3, mid_anchors=2`` and ``n_init = 5*d + 3`` (Loeppky et al.
    2009), a 2D problem yields 8 cheap + 2 mid + 3 HF = 13 evaluations — a
    reasonable approximation to the 70/20/10 design-intent split.

    Parameters
    ----------
    bounds
        Per-dimension ``(lo, hi)`` pairs; ``len(bounds)`` is the design
        dimension ``d``.
    fidelity_levels
        External cascade layer indices (e.g. ``(1, 3, 7)`` or
        ``(1, 3, 5, 6, 7)``).  Sorted ascending to derive the contiguous
        internal index ``0..K-1`` returned in ``fid_indices``.  The cheapest
        fidelity is index ``0``; the HF fidelity is index ``K - 1``; the mid
        fidelity is the median index ``K // 2`` when ``K >= 3``.  When
        ``K == 2``, ``mid_anchors`` is silently zeroed (no median fidelity to
        anchor on); when ``K == 1`` the entire design lives at that single
        fidelity (``hf_anchors`` and ``mid_anchors`` ignored).
    n_init
        Total number of evaluations.  Defaults to ``5*d + 3``.
    hf_anchors
        Number of HF anchor points; the first ``hf_anchors`` cheap-fidelity
        ``x``-values are replicated at the HF level for paired evaluation.
        Must satisfy ``hf_anchors <= n_init - mid_anchors`` (otherwise there
        are no cheap points to pair with).
    mid_anchors
        Number of mid-fidelity points (additional unique Sobol' draws).
        Silently zeroed when ``K < 3``.
    seed
        Seed for :class:`torch.quasirandom.SobolEngine` (the engine is
        scrambled).  Same seed → identical ``(X, fid_indices)`` output.

    Returns
    -------
    X_init : np.ndarray
        Float64 array of shape ``(n_init, d)``.  The first ``n_cheap`` rows
        are cheap-fidelity Sobol' draws; the next ``mid_anchors`` (when
        ``K >= 3``) are mid-fidelity draws; the final ``hf_anchors`` rows are
        the HF replicas (a verbatim copy of the first ``hf_anchors`` cheap
        rows).
    fid_indices : np.ndarray
        Int64 array of shape ``(n_init,)``.  Holds *internal contiguous*
        fidelity indices ``0..K-1`` aligned with ``sorted(fidelity_levels)``,
        not the external layer numbers.  The BO module is the only place that
        does this internal indexing — the caller is responsible for
        translating back to external layers when invoking the cascade.

    Raises
    ------
    ValueError
        If ``bounds`` or ``fidelity_levels`` is empty; if ``n_init`` is not
        positive; if ``hf_anchors`` or ``mid_anchors`` is negative; or if
        ``hf_anchors`` exceeds the available cheap-fidelity slot count.
    """
    if not bounds:
        raise ValueError("bounds must be non-empty")
    if not fidelity_levels:
        raise ValueError("fidelity_levels must be non-empty")
    if hf_anchors < 0:
        raise ValueError(f"hf_anchors must be ≥ 0, got {hf_anchors}")
    if mid_anchors < 0:
        raise ValueError(f"mid_anchors must be ≥ 0, got {mid_anchors}")

    bounds_arr = np.asarray(bounds, dtype=float)
    if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
        raise ValueError(
            f"bounds must be a sequence of (lo, hi) pairs, got shape "
            f"{bounds_arr.shape}"
        )
    if np.any(bounds_arr[:, 0] >= bounds_arr[:, 1]):
        raise ValueError(f"bounds must satisfy lo < hi for every dim: {bounds}")

    d = bounds_arr.shape[0]
    if n_init is None:
        n_init = 5 * d + 3
    if n_init <= 0:
        raise ValueError(f"n_init must be > 0, got {n_init}")

    sorted_levels = sorted(set(fidelity_levels))
    K = len(sorted_levels)
    cheap_idx = 0
    hf_idx = K - 1
    mid_idx = K // 2  # median index; coincides with cheap_idx when K==1

    # Collapse mid into "no mid" when there is no distinct median fidelity.
    if K < 3:
        mid_anchors = 0
    if K == 1:
        # Single fidelity: ignore HF anchors (no distinct HF level to anchor).
        hf_anchors = 0

    n_cheap = n_init - hf_anchors - mid_anchors
    if n_cheap < hf_anchors:
        raise ValueError(
            f"need at least hf_anchors={hf_anchors} cheap points to pair "
            f"with HF replicas, but n_cheap = n_init - hf_anchors - "
            f"mid_anchors = {n_cheap}"
        )
    if n_cheap < 0:
        raise ValueError(
            f"n_init={n_init} too small for hf_anchors={hf_anchors} + "
            f"mid_anchors={mid_anchors}"
        )

    sobol = torch.quasirandom.SobolEngine(d, scramble=True, seed=seed)
    n_unique = n_cheap + mid_anchors
    raw = sobol.draw(n_unique).numpy().astype(np.float64, copy=False)

    lo = bounds_arr[:, 0]
    span = bounds_arr[:, 1] - bounds_arr[:, 0]
    X_unique = lo + raw * span  # broadcasts over n_unique rows

    X_cheap = X_unique[:n_cheap]
    X_mid = X_unique[n_cheap : n_cheap + mid_anchors]
    X_hf = X_cheap[:hf_anchors].copy()  # paired with first hf_anchors cheap x's

    X_init = np.vstack([X_cheap, X_mid, X_hf]) if (mid_anchors or hf_anchors) else X_cheap
    fid_indices = np.concatenate(
        [
            np.full(n_cheap, cheap_idx, dtype=np.int64),
            np.full(mid_anchors, mid_idx, dtype=np.int64),
            np.full(hf_anchors, hf_idx, dtype=np.int64),
        ]
    )
    return X_init, fid_indices


# --- acquisition + mixed optimiser -------------------------------------------


class _HFBonusAcquisition(qMultiFidelityKnowledgeGradient):
    """qMFKG with a constant additive HF acquisition bonus (47.3k.3).

    Wraps :class:`qMultiFidelityKnowledgeGradient` with an additive bonus on
    candidates whose fidelity column equals the target HF index — composes
    orthogonally with :paramref:`run_mfbo.hf_explore_bias` (a binary on/off
    quota) and :paramref:`run_mfbo.adaptive_hf_explore_bias` (a continuous
    schedule on the same quota).  The bonus is ``+α`` per HF candidate in
    each candidate's q-batch, averaged across q (here ``q == 1`` per
    :func:`build_acquisition`).  Motivation: under high cost ratios the
    cost-aware utility ``EIG / cost`` can deprioritise HF acquisition picks
    even when the basin is genuinely under-resolved; a small constant lift on
    the HF acquisition value tips the cost/benefit toward HF without
    requiring the cost table itself to be modified (which destabilises the
    GP fit when the cheap surrogate is row-starved — see 47.3i empirical
    sweep).

    Explicit ``bonus = 0.0`` is the no-op contract: the wrapper code is
    traversed (mask computation, multiplication by zero, addition to base)
    but the result is bytewise-identical to the un-wrapped qMFKG output.
    Adding floating-point ``+0.0`` is exact for non-NaN finite values, so
    ``base + 0.0 * mask`` returns ``base`` unchanged.
    """

    def __init__(
        self,
        *,
        hf_acquisition_bonus: float,
        fidelity_dim: int,
        target_fidelity_index: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._hf_bonus = float(hf_acquisition_bonus)
        self._hf_bonus_fidelity_dim = int(fidelity_dim)
        self._hf_bonus_target = int(target_fidelity_index)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        base = super().forward(X)
        # X layout per ``qKnowledgeGradient.forward``: ``b x (q + num_fantasies)
        # x d_total``.  ``X[..., :-num_fantasies, :]`` is the q sampling-decision
        # points (the candidates whose fidelity column the cost-aware utility
        # consumes); ``X[..., -num_fantasies:, :]`` is the inner-argmax fantasy
        # points whose fidelity column is incidental KG state.  Gate the bonus
        # on the q candidates only.  Round defends against floating-point drift
        # in the fidelity column produced by ``optimize_acqf_mixed``.
        X_actual = X[..., : -self.num_fantasies, :]
        hf_mask = (
            X_actual[..., self._hf_bonus_fidelity_dim].round()
            == self._hf_bonus_target
        ).to(base.dtype)
        # Mean over the q candidate points so bonus shape matches base shape.
        bonus = self._hf_bonus * hf_mask.mean(dim=-1)
        return base + bonus


def build_acquisition(
    model: MultiTaskGP,
    cost_utility: InverseCostWeightedUtility,
    target_fidelity_index: int,
    *,
    num_fantasies: int = 64,
    candidate_set_size: int = 512,
    hf_acquisition_bonus: float | None = None,
) -> tuple[qMultiFidelityKnowledgeGradient, Callable[..., tuple[np.ndarray, int, float]]]:
    """Build a cost-aware multi-fidelity KG acquisition + mixed optimiser.

    Wraps :class:`qMultiFidelityKnowledgeGradient` (BoTorch 0.17.x;
    ``botorch.acquisition.knowledge_gradient`` — *not* the
    ``multi_fidelity`` submodule cited in some BoTorch docs) with a target-
    fidelity projection and an inverse-cost utility, then returns both the
    raw acquisition object and a closure that runs
    :func:`botorch.optim.optimize.optimize_acqf_mixed` over the design space
    (continuous in ``x``) and the discrete set of candidate fidelities (a
    sequence of internal contiguous indices ``0..K-1``).  ``q=1`` because our
    HF cost is too high to amortise batched candidate generation.

    Parameters
    ----------
    model
        Fitted :class:`MultiTaskGP` (built by :func:`build_mf_gp`).  The GP's
        ``_task_feature`` attribute identifies the fidelity column.
    cost_utility
        :class:`InverseCostWeightedUtility` (from :func:`build_cost_model`).
        Weights expected information gain by ``1 / cost(m)``.
    target_fidelity_index
        Internal contiguous fidelity index of the HF target (``K - 1`` for
        ``sorted(fidelity_levels)``).  KG's inner argmax is over the
        posterior mean projected to this fidelity.
    num_fantasies
        Number of fantasy samples for the KG inner optimisation.  Default 64
        per the BoTorch discrete-MF tutorial; trade-off is acquisition cost
        vs. KG variance.
    candidate_set_size
        Default ``raw_samples`` for :func:`optimize_acqf_mixed` (passable per-
        call via the returned closure's ``raw_samples`` kwarg).
    hf_acquisition_bonus
        Optional constant additive bonus ``+α`` applied to qMFKG's acquisition
        value when the candidate's fidelity column equals
        *target_fidelity_index* (47.3k.3).  When ``None`` (default), the
        plain :class:`qMultiFidelityKnowledgeGradient` is constructed and the
        wrapper code is bypassed.  When a finite ``α >= 0``, the returned
        acquisition is a :class:`_HFBonusAcquisition` instance (a subclass of
        qMFKG that adds ``α * mean_q(hf_mask)`` to ``forward()``'s output).
        Explicit ``α = 0.0`` traverses the wrapper code with bonus term ``+0``
        per HF candidate (no-op contract: ``base + 0.0 == base`` bytewise for
        finite floats).  Validation lives in :func:`run_mfbo`; this function
        passes the value through unchanged.

    Returns
    -------
    acquisition : qMultiFidelityKnowledgeGradient
        The raw acquisition object.  Callers can inspect ``num_fantasies``,
        ``current_value``, etc. for diagnostics.
    optimize : callable
        ``optimize(bounds, fidelity_choices, *, num_restarts=5,
        raw_samples=None, options=None) -> (x_next, fidelity_next,
        acq_value)``.  ``bounds`` is a length-``d`` sequence of ``(lo, hi)``
        pairs covering only the design dimensions (the fidelity column is
        bounded internally to ``[min, max]`` of ``fidelity_choices``).
        ``fidelity_choices`` is a sequence of internal fidelity indices to
        consider as candidates.  Returns ``x_next`` (numpy array, shape
        ``(d,)``), ``fidelity_next`` (internal index, int), and the
        acquisition value at the optimum (float).

    Notes
    -----
    Side effect on *model*: this function mutates ``model._output_tasks`` and
    ``model._num_outputs`` so the GP appears single-output (the target task)
    to qMFKG's ``num_outputs > 1`` check at construction time.  MultiTaskGP's
    posterior already returns just the target task's posterior when ``X``'s
    task column equals ``target_fidelity_index`` (via the ``project``
    closure), so the multi-output check is a false positive — but qMFKG
    raises :class:`UnsupportedError` regardless unless we silence it.  The
    BO loop builds a fresh GP per iteration so this side effect is
    well-contained; callers who reuse the GP for non-acquisition purposes
    after this function should restore the attributes themselves.

    Fallback to MES: if KG diagnostics show degeneracy (Gumbel-sampling
    collapse, all fantasies within ``1e-6``, multi-modal posterior trapping),
    swap to ``qMultiFidelityMaxValueEntropy`` — the import path is
    ``from botorch.acquisition.max_value_entropy_search import
    qMultiFidelityMaxValueEntropy`` (also *not* under the
    ``multi_fidelity`` submodule).  The constructor signature differs (MES
    needs a ``candidate_set`` of points, not ``current_value``) but the
    surrounding ``project`` / ``cost_aware_utility`` plumbing is identical.
    """
    if num_fantasies < 1:
        raise ValueError(f"num_fantasies must be ≥ 1, got {num_fantasies}")
    if candidate_set_size < 1:
        raise ValueError(
            f"candidate_set_size must be ≥ 1, got {candidate_set_size}"
        )

    fidelity_dim = int(model._task_feature)
    d_total = int(model.train_inputs[0].shape[-1])  # design dims + 1 task col
    n_tasks = int(model.num_tasks)
    if not (0 <= target_fidelity_index < n_tasks):
        raise ValueError(
            f"target_fidelity_index={target_fidelity_index} out of range "
            f"for GP with num_tasks={n_tasks}"
        )

    # Specialise the GP to the target task so it appears single-output to
    # qMFKG's ``num_outputs > 1`` check.  See "Notes" above.
    model._output_tasks = [target_fidelity_index]
    model._num_outputs = 1

    def project(X: torch.Tensor) -> torch.Tensor:
        return project_to_target_fidelity(
            X=X,
            target_fidelities={fidelity_dim: float(target_fidelity_index)},
            d=X.shape[-1],
        )

    # KG's ``current_value`` is the best posterior mean at the target
    # fidelity over training inputs — standard for noisy / multi-fidelity
    # GPs (the best *observed* point can be unreliable when the cheap-
    # fidelity surrogate is biased relative to HF).
    model.eval()
    with torch.no_grad():
        current_value = model.posterior(project(model.train_inputs[0])).mean.max()

    if hf_acquisition_bonus is None:
        acquisition = qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=num_fantasies,
            current_value=current_value,
            cost_aware_utility=cost_utility,
            project=project,
        )
    else:
        # 47.3k.3: subclass that adds a constant bonus on HF candidates.
        # Construction kwargs match the plain qMFKG path; the wrapper only
        # adds three auxiliary fields (bonus, fidelity_dim, target).
        acquisition = _HFBonusAcquisition(
            hf_acquisition_bonus=hf_acquisition_bonus,
            fidelity_dim=fidelity_dim,
            target_fidelity_index=target_fidelity_index,
            model=model,
            num_fantasies=num_fantasies,
            current_value=current_value,
            cost_aware_utility=cost_utility,
            project=project,
        )

    def optimize(
        bounds: Sequence[tuple[float, float]],
        fidelity_choices: Sequence[int],
        *,
        num_restarts: int = 5,
        raw_samples: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, int, float]:
        if not bounds:
            raise ValueError("bounds must be non-empty")
        if not fidelity_choices:
            raise ValueError("fidelity_choices must be non-empty")
        if len(bounds) != d_total - 1:
            raise ValueError(
                f"bounds has {len(bounds)} dims but the GP expects "
                f"{d_total - 1} design dims (excluding the fidelity column "
                f"at index {fidelity_dim})"
            )
        for f in fidelity_choices:
            if not (0 <= int(f) < n_tasks):
                raise ValueError(
                    f"fidelity_choices contains {f}, out of range "
                    f"[0, {n_tasks})"
                )

        # Assemble the (2, d_total) bounds tensor.  Walk the d_total columns;
        # the fidelity column gets [min, max] over fidelity_choices and the
        # design columns consume from `bounds` in order.  Robust to any
        # task_feature index, not just the last column.
        lo = torch.zeros(d_total, dtype=torch.float64)
        hi = torch.zeros(d_total, dtype=torch.float64)
        design_iter = iter(bounds)
        f_min = float(min(fidelity_choices))
        f_max = float(max(fidelity_choices))
        for i in range(d_total):
            if i == fidelity_dim:
                lo[i] = f_min
                hi[i] = f_max
            else:
                lo_b, hi_b = next(design_iter)
                lo[i] = float(lo_b)
                hi[i] = float(hi_b)
        bnds_tensor = torch.stack([lo, hi])

        fixed_features_list = [
            {fidelity_dim: float(f)} for f in fidelity_choices
        ]

        candidate, acq_value = optimize_acqf_mixed(
            acq_function=acquisition,
            bounds=bnds_tensor,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples if raw_samples is not None else candidate_set_size,
            fixed_features_list=fixed_features_list,
            options=options or {},
        )
        cand = candidate.detach().cpu().numpy().reshape(d_total)
        fidelity_next = int(round(float(cand[fidelity_dim])))
        x_next = np.delete(cand, fidelity_dim)
        return x_next, fidelity_next, float(acq_value.item())

    return acquisition, optimize


# --- end-to-end BO driver ----------------------------------------------------


def _resolve_min_acq_iters(
    min_acquisition_iterations: int | None, K: int
) -> int:
    """Resolve :func:`run_mfbo`'s ``min_acquisition_iterations`` default.

    Returns ``min_acquisition_iterations`` verbatim when not ``None``;
    otherwise returns ``max(_MIN_ACQ_ITERATIONS_FLOOR, K)`` — the documented
    default that ensures every fidelity gets a chance to be picked under
    cost-aware acquisition (``>= K``) and that the GP's posterior at the
    incumbent has time to become non-uniform after the DOE
    (``>= _MIN_ACQ_ITERATIONS_FLOOR``).

    Extracted from :func:`run_mfbo` in 47.3k.1.1 so the default-resolution
    contract is testable in isolation: the prior conditional-assertion
    integration tests could not distinguish a ``"variance"``-fires-late
    path (correct) from a ``"variance"``-cannot-fire path (mutation
    silently uses the floor instead of the kwarg).  Mirrors the
    :func:`_stagnation_triggered` extraction pattern from 47.3e.

    Validation of the kwarg's type/range lives in :func:`run_mfbo`, not
    here — the helper inherits whatever production code passes through.

    Parameters
    ----------
    min_acquisition_iterations
        The user-supplied kwarg value, or ``None`` to request the default.
    K
        Number of fidelity levels (``len(fidelity_levels)`` in
        :func:`run_mfbo`).  Used as the lower bound on the resolved floor.

    Returns
    -------
    int
        Resolved minimum acquisition-iteration count.
    """
    if min_acquisition_iterations is not None:
        return min_acquisition_iterations
    return max(_MIN_ACQ_ITERATIONS_FLOOR, K)


def _stagnation_triggered(
    hf_evals: Sequence[BOEval], window: int = 10
) -> bool:
    """Return True when the HF running minimum has not improved in the last *window* evals.

    Pure check used by :func:`run_mfbo` to decide whether to short-circuit on
    a stagnant HF trace.  Finds the index of the lowest-valued entry in
    *hf_evals* (treated as chronological) and returns True only when that
    best is older than the trailing *window* — i.e. at index
    ``<= len(hf_evals) - (window + 1)``.

    Returns False when *hf_evals* contains fewer than ``window + 1`` entries:
    the guard cannot fire until at least one improvement-window has elapsed
    after a candidate "best".

    The caller is responsible for pre-filtering to finite (non-sentinel) HF
    rows; this helper applies no filtering of its own.

    Parameters
    ----------
    hf_evals : Sequence[BOEval]
        HF evaluations in chronological (insertion) order.
    window : int, default 10
        Trailing-window size for the no-improvement check.  Must be ``>= 1``.

    Raises
    ------
    ValueError
        If *window* is non-positive.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if len(hf_evals) < window + 1:
        return False
    best_idx = min(range(len(hf_evals)), key=lambda i: hf_evals[i].value)
    return best_idx <= len(hf_evals) - (window + 1)


def _variance_guard_relative_fired(
    var_inc: float, max_var_grid: float, threshold: float
) -> bool:
    """Return True when the variance guard's relative criterion fires.

    The relative criterion is
    ``var_inc < threshold * max(max_var_grid, 1e-30)``.  The
    ``max(..., 1e-30)`` floor protects against ``max_var_grid == 0`` (a
    degenerate posterior on the Sobol' grid) — without the floor, a
    zero-grid-variance would force ``relative_fired = True`` regardless of
    *var_inc*, giving a spurious early exit.

    Extracted from :func:`run_mfbo` in 47.3k.2.1 so the relative-threshold
    contract is testable in isolation.  The prior integration test
    (``test_variance_guard_relative_threshold_kwarg``) admitted a vacuous
    no-effect-observed path that could not distinguish a "kwarg honoured"
    correct case from a "kwarg silently ignored, hardcoded constant"
    mutation.  Mirrors :func:`_stagnation_triggered` and
    :func:`_resolve_min_acq_iters`.

    Parameters
    ----------
    var_inc : float
        Posterior variance at the incumbent (target fidelity).
    max_var_grid : float
        Maximum posterior variance over the variance-guard's Sobol' grid.
    threshold : float
        Relative-threshold multiplier (``run_mfbo``'s
        ``variance_guard_relative_threshold`` kwarg).  Validation lives in
        :func:`run_mfbo`; this helper inherits whatever production code
        passes through.

    Returns
    -------
    bool
        True when the relative criterion fires (variance guard exit allowed).
    """
    return var_inc < threshold * max(max_var_grid, 1e-30)


def _collect_sentinel_x(eval_history: Sequence["BOEval"]) -> np.ndarray | None:
    """Return the design-vector array of sentinel rows, or ``None`` if empty.

    Filters *eval_history* by the inverse of the 47.3b finite-mask
    convention (``np.isfinite(e.value) and e.value < _BO_SENTINEL / 2``);
    rows that fail this predicate are sentinel candidates the
    ``"voronoi"`` recommendation strategy (47.6b.3.2c.2) excludes from the
    Sobol' search grid.  Returns a fresh ``(n_sentinel, d)`` numpy array
    for the helper to consume, or ``None`` when no sentinels exist (the
    helper's masking branch is then a no-op and the recommendation is
    bytewise-identical to the ``"mean"`` strategy).
    """
    sentinel_xs = [
        np.asarray(e.x, dtype=float)
        for e in eval_history
        if not (np.isfinite(e.value) and e.value < _BO_SENTINEL / 2)
    ]
    if not sentinel_xs:
        return None
    return np.stack(sentinel_xs)


def _recommend_incumbent(
    model: MultiTaskGP,
    bounds: Sequence[tuple[float, float]],
    target_fidelity_index: int,
    d: int,
    seed: int,
    *,
    n_grid: int = 1024,
    strategy: str = "mean",
    sentinel_x: np.ndarray | None = None,
    voronoi_radius: float = 0.1,
    ucb_beta: float = 2.0,
    info_out: dict | None = None,
) -> np.ndarray:
    """Return the recommended incumbent on a Sobol' grid of *n_grid* points.

    The default ``strategy="mean"`` picks ``argmin_x μ_n(x, m=target)`` —
    posterior mean (not best observed), standard for noisy or multi-fidelity
    GPs where the cheap-fidelity surrogate may be biased relative to HF.  The
    Sobol' engine is scrambled with the supplied *seed* so the recommendation
    is reproducible across runs.

    The ``"voronoi"`` strategy (47.6b.3.2c.2) takes the same Sobol' grid +
    posterior mean ranking but masks out grid points within *voronoi_radius*
    (L2) of any sentinel ``x`` recorded by the caller in *sentinel_x*.  The
    masked argmin then avoids the residual extrapolation hazard at infeasible
    boundaries that 47.6b.3.1's clamp does not fully eliminate (the clamp lifts
    the GP posterior at sentinel rows, but the GP can still extrapolate over
    nearby Sobol' grid points).  When *sentinel_x* is ``None`` or empty, the
    masking step is a no-op and the result is bytewise-identical to the
    ``"mean"`` strategy.  When all grid points fall inside the union of
    Voronoi cells (degenerate sentinel cluster covers the bounded region),
    the helper falls back to the unmasked argmin and signals the fallback
    via ``info_out["voronoi_fallback"] = True`` if *info_out* was supplied.

    The ``"ucb"`` strategy (47.6b.3.2c.3) replaces the posterior-mean ranking
    with the upper-confidence bound on the minimization target,
    ``score = mean + ucb_beta * sigma``.  This is the pessimistic estimate:
    lower mean wins, lower sigma wins — high-variance regions (where the
    clamped GP is least confident, typically near sentinel boundaries that
    survive the Voronoi mask) are penalised.  The sign convention follows
    Auer et al. 2002 with the formula adjusted for minimization (for
    maximisation the standard UCB acquisition is ``mean + beta * sigma`` where
    higher score wins; here, lower score wins so the same formula expresses
    the upper bound).  When ``ucb_beta == 0``, the score collapses to mean
    and the result is bytewise-identical to the ``"mean"`` strategy.

    Parameters
    ----------
    sentinel_x
        Optional ``(n_sentinel, d)`` array of design vectors that returned
        sentinel values during the BO loop.  Consumed by the ``"voronoi"``
        strategy only.  Caller is expected to apply the existing 47.3b
        finite-mask convention (``np.isfinite(value) and value < _BO_SENTINEL
        / 2``) so the rows correspond to genuinely infeasible candidates.
    voronoi_radius
        L2 mask radius for the ``"voronoi"`` strategy.  Grid points within
        this distance of any sentinel are excluded from the argmin.  Must be
        ``> 0`` and finite when used; the caller (``run_mfbo``) validates this
        on its kwarg surface so callers of this helper directly are responsible
        for passing a sensible value.
    ucb_beta
        Exploration penalty for the ``"ucb"`` strategy.  Must be ``>= 0`` and
        finite when used; the caller (``run_mfbo``) validates this on its
        kwarg surface.  ``0`` collapses UCB to the mean ranking.  Default
        ``2.0`` per Auer et al. 2002.
    info_out
        Optional mutable dict.  When supplied and the ``"voronoi"`` fallback
        fires (all grid points masked → unmasked argmin), the helper sets
        ``info_out["voronoi_fallback"] = True`` so the caller can record the
        flag in :attr:`BOResult.extras`.
    """
    if strategy not in ("mean", "voronoi", "ucb"):
        raise ValueError(
            "_recommend_incumbent strategy must be one of "
            f"{{'mean', 'voronoi', 'ucb'}}, got {strategy!r}"
        )
    sobol = torch.quasirandom.SobolEngine(d, scramble=True, seed=seed)
    raw = sobol.draw(n_grid).double()
    bounds_arr = torch.tensor(
        [[lo, hi] for lo, hi in bounds], dtype=torch.float64
    )
    lo = bounds_arr[:, 0]
    hi = bounds_arr[:, 1]
    X = lo + raw * (hi - lo)
    fid_col = torch.full(
        (n_grid, 1), float(target_fidelity_index), dtype=torch.float64
    )
    X_full = torch.cat([X, fid_col], dim=1)
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(X_full)
        mean = posterior.mean.squeeze(-1)
        if strategy == "ucb":
            # 47.6b.3.2c.3: upper-confidence bound on the minimization
            # target.  ``variance.clamp(min=0.0)`` defends against tiny
            # negative variances from numerical roundoff in BoTorch's
            # posterior chain (the Cholesky-derived variance can dip
            # slightly below zero on near-singular kernels).
            variance = posterior.variance.squeeze(-1).clamp(min=0.0)
            sigma = torch.sqrt(variance)
            score = mean + ucb_beta * sigma
        else:
            score = mean

    # 47.6b.3.2c.2: voronoi mask gates grid points by minimum L2 distance to
    # any recorded sentinel x.  When no sentinels exist (or strategy="mean"),
    # the masking branch is skipped entirely and behaviour is bytewise-
    # identical to the pre-47.6b.3.2c "mean" path.
    if (
        strategy == "voronoi"
        and sentinel_x is not None
        and len(sentinel_x) > 0
    ):
        sentinel_t = torch.as_tensor(
            np.asarray(sentinel_x), dtype=torch.float64
        )
        # cdist returns (n_grid, n_sentinel) pairwise L2 distances.
        dist = torch.cdist(X, sentinel_t)
        min_dist = dist.min(dim=-1).values
        feasible_mask = min_dist >= voronoi_radius
        if int(feasible_mask.sum().item()) > 0:
            inf_t = torch.tensor(float("inf"), dtype=score.dtype)
            masked_score = torch.where(feasible_mask, score, inf_t)
            idx = int(torch.argmin(masked_score).item())
        else:
            # Degenerate: every grid point is within voronoi_radius of some
            # sentinel.  Fall back to the unmasked argmin and signal the
            # fallback so the caller can record it in BOResult.extras.
            idx = int(torch.argmin(score).item())
            if info_out is not None:
                info_out["voronoi_fallback"] = True
    else:
        idx = int(torch.argmin(score).item())
    return X[idx].detach().cpu().numpy()


def _compute_hf_uncertainty_signal(
    model: MultiTaskGP,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    target_fid_idx: int,
    d: int,
    bounds_t: Sequence[tuple[float, float]],
    seed: int,
    *,
    n_grid: int = 256,
) -> tuple[float, float] | None:
    """Return ``(var_hf_grid, spread_hf)`` for the adaptive HF mechanisms.

    Shared signal consumed by both :paramref:`run_mfbo.adaptive_hf_floor`
    (47.3i) and :paramref:`run_mfbo.adaptive_hf_explore_bias` (47.3j).  Both
    mechanisms previously inlined the same ``n_grid``-point Sobol' grid +
    posterior-variance-max + HF-only spread computation; factoring into a
    single helper guarantees the two predicates cannot diverge under future
    edits (47.3j.1 Gap 3).

    Best-effort: returns ``None`` on any internal failure (insufficient HF
    rows, posterior call raises) so callers fall back to the static branch
    without aborting the BO loop.

    Parameters
    ----------
    model : MultiTaskGP
        Fitted GP whose posterior is queried at HF.
    X_train, Y_train
        Current training tensors as numpy arrays; the helper filters to HF
        rows via the fidelity column at index ``d``.
    target_fid_idx : int
        Internal fidelity index (0..K-1) the BO module uses to address HF on
        the GP's task axis.
    d : int
        Design dimension (number of non-fidelity columns in ``X_train``).
    bounds_t : Sequence[tuple[float, float]]
        Per-design-axis ``(lo, hi)`` pairs.  Used to scale the Sobol' grid
        from the unit hypercube into the design domain.
    seed : int
        Sobol' engine seed.  Reproducible across calls with the same seed.
    n_grid : int, default 256
        Sobol' grid size for the variance-max query.  256 is the size used
        by the in-loop adaptive blocks before factoring; the variance-guard
        block (which has additional incumbent-variance machinery) keeps its
        own grid construction.

    Returns
    -------
    tuple[float, float] or None
        ``(var_hf_grid, spread_hf)`` where ``spread_hf`` is floored at
        ``1e-12`` so callers can use it in a denominator without further
        guarding.  ``None`` indicates the signal is unavailable for this
        iteration; callers must treat that as "skip the adaptive branch".
    """
    try:
        Y_hf = Y_train[X_train[:, d] == target_fid_idx]
        if Y_hf.size < 2:
            return None
        spread_hf = max(
            float(np.max(Y_hf)) - float(np.min(Y_hf)), 1e-12
        )
        sobol = torch.quasirandom.SobolEngine(d, scramble=True, seed=seed)
        raw = sobol.draw(n_grid).double()
        bounds_arr = torch.tensor(
            [list(b) for b in bounds_t], dtype=torch.float64
        )
        lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]
        grid_X = lo + raw * (hi - lo)
        grid_full = torch.cat(
            [
                grid_X,
                torch.full(
                    (n_grid, 1),
                    float(target_fid_idx),
                    dtype=torch.float64,
                ),
            ],
            dim=1,
        )
        with torch.no_grad():
            var_hf_grid = float(
                model.posterior(grid_full).variance.max().item()
            )
        return var_hf_grid, spread_hf
    except Exception:
        return None


def run_mfbo(
    scheme: str,
    kernel: str,
    report_fields_by_layer: dict[int, str],
    bounds: Sequence[tuple[float, float]],
    *,
    budget_evals: int | None = None,
    budget_seconds: float | None = None,
    cost_table: dict[int, float] | None = None,
    seed: int = 0,
    n_init: int | None = None,
    hf_anchors: int | None = None,
    min_acquisition_iterations: int | None = None,
    hf_explore_bias: float = 0.0,
    hf_priority_warmup: int = 0,
    adaptive_hf_floor: float | None = None,
    adaptive_hf_explore_bias: float | None = None,
    variance_guard_relative_threshold: float = 1e-5,
    hf_acquisition_bonus: float | None = None,
    clamp_sentinel_rows: bool = True,
    recommendation_strategy: str = "mean",
    voronoi_radius: float = 0.1,
    ucb_beta: float = 2.0,
    num_fantasies: int = 64,
    verbose: bool = False,
    objective: Callable[[np.ndarray, int], tuple[float, float, dict]] | None = None,
) -> BOResult:
    """Drive a multi-fidelity Bayesian optimisation loop end-to-end.

    Wires the dataclasses (47.1a), objective factory (47.1b), GP surrogate
    (47.2a), cost model (47.2b), DOE (47.2c), and acquisition (47.3a) into a
    single driver:

    1. Seed RNGs (``torch``, ``numpy``, Sobol' engines) for reproducibility.
    2. Resolve the per-fidelity cost table (default: filtered slice of
       :data:`DEFAULT_COST_TABLE`).
    3. Build the multi-fidelity objective via
       :func:`make_multi_fidelity_objective` (or use an injected *objective*
       hook for tests).
    4. Generate the stratified initial design via :func:`build_initial_design`
       and evaluate every ``(x_i, m_i)`` pair.
    5. While budget allows: refit the GP on finite (non-sentinel) rows, check
       variance / stagnation guards, build cost-aware qMFKG via
       :func:`build_acquisition`, optimise the acquisition with
       :func:`optimize_acqf_mixed` (mixed continuous/discrete), evaluate at
       the chosen ``(x, m)``.
    6. Recommend the incumbent ``x_inc = argmin μ_n(x, m=hf)`` on a 1024-point
       Sobol' grid (posterior mean, not best observed).
    7. Re-evaluate the cascade at ``x_inc`` at HF to populate the returned
       ``best_objective`` and ``best_report`` from real data, not the GP
       posterior.

    Parameters
    ----------
    scheme, kernel
        Forwarded to :func:`brady2d_stability_score` (and to
        :func:`params_from_vector` for the per-eval ``params`` payload).
    report_fields_by_layer
        Mapping ``layer index → dotted field path``.  ``max(...)`` is the HF
        target; the corresponding field is the optimisation objective.
    bounds
        Per-dimension ``(lo, hi)`` design-space bounds.  Length is the design
        dimension ``d``; the GP appends one fidelity column for a total of
        ``d + 1`` inputs.
    budget_evals
        Total number of cascade evaluations (initial design + acquisition
        steps + final HF re-evaluation at the incumbent).  Mutually exclusive
        with *budget_seconds*; exactly one must be set.  ``sum(n_evals_per_
        fidelity.values()) == budget_evals`` after the run.  Must be ``>= 2``
        to leave room for at least one initial design point and the final
        HF re-evaluation.
    budget_seconds
        Wall-time budget in seconds.  Mutually exclusive with *budget_evals*.
    cost_table
        Per-layer cost dict ``{external layer index: seconds}``.  Defaults to
        :data:`DEFAULT_COST_TABLE` filtered to ``report_fields_by_layer``.
        Floored via :func:`apply_cost_floor` before construction of the
        cost-aware utility.
    seed
        Master RNG seed.  Setting the same seed across two runs produces the
        same :attr:`BOResult.best_x` to within ``1e-6``.
    n_init
        Initial design size; defaults to ``5*d + 3`` (Loeppky et al. 2009).
    hf_anchors
        Number of HF anchor points in the initial design (47.3f).  When
        ``None``, defaults to ``max(3, d + 2)`` so that higher-dimensional
        problems get more HF coverage; the ``d + 2`` term scales the effective
        HF anchor count with problem dimension while keeping ``3`` as a floor
        for the historical d=1/d=2 small-problem behaviour.  Forwarded to
        :func:`build_initial_design` (whose own default remains ``3`` so direct
        callers and existing fixtures keep working).
    min_acquisition_iterations
        Minimum number of acquisition iterations that must run after the
        initial design before the variance early-exit guard can fire (47.3f).
        When ``None``, defaults to ``max(_MIN_ACQ_ITERATIONS_FLOOR, K)`` where
        ``K`` is the number of fidelity levels and
        :data:`_MIN_ACQ_ITERATIONS_FLOOR` is currently ``15`` (raised from
        ``5`` in 47.3k.1).  Defends against the GP collapsing uniformly to
        its noise floor on smooth synthetic objectives — the 47.3d combined
        absolute+relative criterion is necessary but not sufficient when the
        initial design alone seeds enough HF data to satisfy both halves;
        this delay forces the loop to consume some acquisition budget so
        that the incumbent recommendation reflects post-GP-fit evidence
        rather than the DOE alone.
    hf_explore_bias
        Target lower bound on the HF fraction of acquisition picks (47.3g).
        Must lie in ``[0, 1]``.  Default ``0.0`` disables the mechanism and
        preserves the pre-47.3g cost-aware contract (qMFKG picks ``(x, m)``
        unconstrained).  When ``> 0``, before each acquisition step we
        compute the running HF fraction among acquisition picks made so far
        (the initial design is excluded — its stratification is governed by
        :func:`build_initial_design`) and, if that fraction would still be
        below *hf_explore_bias* even after this iteration, we restrict the
        ``fidelity_choices`` argument to ``optimize_acqf_mixed`` to ``[HF]``
        only.  qMFKG then picks the optimal ``x`` for HF specifically rather
        than ``x_next`` chosen for a cheaper layer with the fidelity
        overridden post-hoc.  This addresses the cost-aware utility's
        tendency to under-sample HF when the cost ratio is large
        (``c(HF) / c(cheap) >> 1``) and HF coverage in the GP is still
        sparse — a regime where the basin floor of the objective cannot be
        resolved without forcing more HF picks (Wu et al. 2020 §4 motivates
        a similar quota; the BoTorch tutorial does not, which is why the
        default is off).
    hf_priority_warmup
        Number of opening acquisition iterations forced to HF regardless of
        cost-aware utility or :paramref:`hf_explore_bias` quota (47.3h).
        Must be ``>= 0``; default ``0`` disables the mechanism and preserves
        pre-47.3h behaviour bytewise.  When ``> 0``, the first
        ``hf_priority_warmup`` acquisition steps after the initial design
        restrict ``optimize_acqf_mixed``'s ``fidelity_choices`` to ``[HF]``
        only — qMFKG picks the optimal ``x`` for HF specifically.  Once the
        warmup is exhausted, control returns to the cost-aware utility
        (which may then be further restricted by ``hf_explore_bias`` if
        set).  Motivation: when the HF posterior is malleable (few HF
        anchors in init, high-cost-ratio regime), the cost-aware utility
        defers HF picks indefinitely and the GP cannot localise the basin.
        Forcing the first K-1 (number-of-fidelities minus one) acquisition
        picks to HF guarantees baseline HF coverage near the cheap-fidelity-
        suggested optimum before the cost-aware utility takes over.
        Composes with ``hf_explore_bias``: warmup runs first, quota runs
        after.
    adaptive_hf_floor
        Multiplier ``α`` controlling an adaptive floor on the cost-aware
        utility's effective HF cost (47.3i).  When ``None`` (default), the
        mechanism is disabled and the cost-aware utility sees the floored
        ``cost_table`` unchanged across iterations.  When a float ``α >= 1``,
        before each acquisition step we evaluate two predicates:

        - "HF posterior uncertain": ``var_hf_grid > τ * spread_hf**2`` where
          ``var_hf_grid`` is the maximum posterior variance over a 256-point
          Sobol' grid at HF, ``spread_hf`` is the range of HF-only Y values,
          and ``τ`` is :data:`_ADAPTIVE_HF_FLOOR_TAU` (``0.01``).
        - "Cheap surrogate well-fit": ``n_cheap_finite >= max(2 * d, K)``
          where ``n_cheap_finite`` is the count of finite Y_train rows at
          cheap (non-HF) fidelities.

        When both fire, the effective HF cost is lifted to
        ``min(c(hf), α * min_cheap_cost)`` for that iteration only — i.e. the
        cost-aware utility sees HF as "only ``α×`` more expensive than the
        cheapest layer" rather than the true ratio (often 100×).  This biases
        acquisition toward HF until the posterior tightens or the cheap
        surrogate is no longer well-fit; the floor reverts automatically when
        either predicate fails.  Composes with ``hf_priority_warmup``
        (warmup wins for the first N picks) and ``hf_explore_bias`` (quota
        further restricts choices when the running HF fraction is below
        target).  Motivation: even with the warmup + quota, the cost-aware
        utility's 100× ratio dominates once warmup expires and the basin
        cannot be resolved within tight evaluation budgets (47.3h empirical
        sweep on AugmentedBranin).  Lowering effective HF cost only when the
        GP says HF is still informative *and* cheap data is sufficient
        defends against the GP-starvation regime that pure-quota approaches
        hit at high quotas.
    adaptive_hf_explore_bias
        Coefficient ``β`` controlling an adaptive schedule on top of the
        static :paramref:`hf_explore_bias` quota (47.3j).  When ``None``
        (default), the mechanism is disabled and the quota uses
        :paramref:`hf_explore_bias` verbatim.  When a float in ``[0, 1]``,
        before each acquisition step the effective quota is

            ``effective_bias = max(hf_explore_bias,
                                   β * var_hf_grid /
                                   (var_hf_grid + spread_hf**2))``

        where ``var_hf_grid`` is the maximum posterior variance over a
        256-point Sobol' grid at HF (the same construction used by the
        variance guard and the adaptive floor) and ``spread_hf`` is the
        range of HF-only Y values.  When the HF posterior is uncertain
        (``var_hf_grid >> spread_hf**2``) the second term tends toward
        ``β`` and the quota lifts; when the posterior tightens it tends
        toward zero and the quota reverts to the static value.  Composes
        with :paramref:`hf_priority_warmup` (warmup wins for the first N
        picks) and :paramref:`adaptive_hf_floor` (cost-table floor runs
        on the same uncertainty signal but adjusts cost rather than
        quota).  Continuous schedule rather than the binary on/off of
        the cost-floor swap, so it composes more cleanly with qMFKG's
        cost-utility on small budgets where cost-table swings can
        destabilise the GP fit (47.3i empirical sweep).
    variance_guard_relative_threshold
        Relative-variance threshold used by the variance early-exit guard
        (47.3k.2).  The guard fires only when *both* the absolute criterion
        ``var_inc < 1e-6 * spread_hf**2`` *and* the relative criterion
        ``var_inc < variance_guard_relative_threshold * max_var_grid`` hold,
        where ``var_inc`` is the GP posterior variance at the recommended
        incumbent and ``max_var_grid`` is the maximum posterior variance
        over a 256-point Sobol' grid at HF.  Default ``1e-5`` (tightened
        from the pre-47.3k.2 hardcoded ``1e-3``); empirically the
        ``1e-3`` floor fires aggressively on smooth synthetic objectives
        where the GP collapses uniformly to its noise floor and
        ``var_inc / max_var_grid ≈ 1`` even when the basin is far from
        localised.  Tightening to ``1e-5`` blocks the spurious exits
        without affecting genuinely-converged runs.  Must be strictly
        positive and finite; reject ``<= 0`` and NaN with ``ValueError``.
        Composes orthogonally with :paramref:`min_acquisition_iterations`
        (which delays *when* the guard can fire) and the static absolute
        threshold (which is unchanged at ``1e-6 * spread_hf**2``).
    hf_acquisition_bonus
        Constant additive bonus ``+α`` applied to qMFKG's acquisition value
        on HF candidates (47.3k.3).  When ``None`` (default), the plain
        qMFKG is constructed and the wrapper code is bypassed entirely
        (short-circuit: pre-47.3k.3 behaviour preserved bytewise).  When a
        finite ``α >= 0``, the acquisition is wrapped in
        :class:`_HFBonusAcquisition` (a thin :class:`qMultiFidelity\
KnowledgeGradient` subclass that adds ``α * mean_q(hf_mask)`` to
        ``forward()``'s output, where ``hf_mask`` is the boolean indicator
        ``X[..., fidelity_dim].round() == target_fidelity_index``).
        Composes orthogonally with :paramref:`hf_explore_bias` (binary
        on/off quota), :paramref:`adaptive_hf_explore_bias` (continuous
        schedule), and :paramref:`adaptive_hf_floor` (cost-table swap).
        The bonus is a continuous additive lift on the acquisition value
        directly — qMFKG's ``EIG / cost`` cost-aware utility is unaffected
        in its denominator, so the GP fit's cheap-vs-HF row balance is not
        disturbed (unlike the cost-floor mechanism, which destabilised the
        marginal-likelihood optimiser at high lifts in the 47.3i sweep).
        Motivation: under high cost ratios (e.g. ``c(HF) / c(cheap) = 100``)
        the cost-aware utility deprioritises HF picks even when the basin
        is genuinely under-resolved; a small bonus tips ``EIG / cost + α``
        toward HF when the basin needs more coverage.  Must be ``>= 0`` and
        finite; reject negative and NaN with ``ValueError``.  Explicit
        ``0.0`` traverses the wrapper code with bonus term ``+0`` per HF
        candidate (no-op contract: bytewise-identical to ``None``).
    clamp_sentinel_rows
        Sentinel-row treatment for the GP fit (47.6b.3.1).  When ``True``
        (default), per-fidelity sentinel rows in ``Y_train`` (rows where
        the cascade returned :data:`_BO_SENTINEL` because the candidate
        tripped a feasibility gate or the per-eval objective raised) are
        replaced with ``max(Y_m_finite) + 3 * std(Y_m_finite)`` before the
        GP fit, separately for each fidelity ``m``.  Per-fidelity
        clamping is essential because Y scales differ across fidelities
        (e.g. ``layer1.boundary_gv_err`` ~ 0.02 vs ``layer3.max_stab_eig``
        ~ -1e-4); a global clamp would inflate one fidelity's residuals
        relative to the other and destabilise the ICM kernel's
        identification of off-diagonal correlations.  When ``False``,
        sentinel rows are filtered out (pre-47.6b.3.1 contract preserved
        bytewise).  Motivation: when the cascade objective has many
        infeasible regions (~50% at HF in 47.6b.3's E4-classical α
        sweep), pure filtering means the GP cannot learn that those
        regions are bad; the ``_recommend_incumbent`` posterior-minimum
        scan then extrapolates into the un-trained sentinel regions and
        lands at a phantom-low-mean corner (final HF re-eval there
        returns sentinel, ``best_objective = 1e12``).  Clamping at
        ``max + 3 * std`` is the standard "constrained BO via
        one-sided log-barrier" pattern in the BoTorch literature: the
        GP's posterior at infeasible regions reports a high mean,
        ``_recommend_incumbent``'s minimum-search avoids them naturally,
        and ``optimize_acqf_mixed`` similarly under-prioritises them.
        Fallback: when a fidelity has fewer than 2 finite rows, that
        fidelity's sentinels are filtered (insufficient data to compute
        a meaningful clamp).  Both clamped and fallback-filtered rows
        are tallied in ``BOResult.extras["n_sentinel_per_fidelity"]``
        (treatment-agnostic occurrence count keyed by external layer
        index — 47.6b.3.1.1).
    recommendation_strategy
        How to pick the incumbent ``x_inc`` from the GP posterior at HF.
        Must be one of ``{"mean", "voronoi", "ucb"}``; the comparison is
        case-sensitive.  Default ``"mean"`` is bytewise-identical to the
        pre-47.6b.3.2c behaviour: on a 1024-pt Sobol' grid at HF, pick
        ``argmin_x μ_n(x, m=hf)``.  ``"voronoi"`` (47.6b.3.2c.2) takes the
        same Sobol' grid + posterior mean ranking but masks out grid points
        within :paramref:`run_mfbo.voronoi_radius` of any recorded sentinel
        ``x`` (rows in :attr:`eval_history` that fail the existing 47.3b
        finite-mask convention) — addresses the residual extrapolation
        hazard at infeasible boundaries that 47.6b.3.1's clamp does not
        fully eliminate.  When all grid points fall inside the union of
        Voronoi cells, the helper falls back to the unmasked argmin and
        sets :attr:`BOResult.extras["voronoi_fallback"] = True`.  ``"ucb"``
        (47.6b.3.2c.3) replaces the posterior-mean ranking with the
        upper-confidence bound on the minimization target, ``score = mean +
        ucb_beta * sigma``: lower mean wins, lower sigma wins, so
        high-variance regions (where the clamped GP is least confident,
        typically near sentinel boundaries that survive the Voronoi mask)
        are penalised.  When ``ucb_beta == 0``, the UCB ranking collapses
        to the mean ranking and the result is bytewise-identical to
        ``"mean"``.
    voronoi_radius
        L2 mask radius for the ``"voronoi"`` strategy (47.6b.3.2c.2).  Must
        be a strictly positive finite float.  Validated unconditionally on
        the kwarg surface so the value is always sound regardless of which
        strategy the caller picks; the ``"mean"`` and ``"ucb"`` branches
        ignore it.  Default ``0.1`` is sized for designs scaled to
        ``[-1, 1]^d`` or ``[0, 1]^d``; callers using larger bounds should
        pass a proportionally larger radius.
    ucb_beta
        Exploration penalty for the ``"ucb"`` strategy (47.6b.3.2c.3).
        Must be ``>= 0`` and finite.  Validated unconditionally on the
        kwarg surface; the ``"mean"`` and ``"voronoi"`` branches ignore it.
        Default ``2.0`` per Auer et al. 2002 with the sign convention
        adjusted for minimization.  ``0.0`` collapses UCB to mean-only
        (no-op contract for the default-off equivalence test).
    num_fantasies
        Forwarded to :func:`build_acquisition`.  Default 64.
    verbose
        If ``True``, print one line per evaluation.
    objective
        Override hook for tests: a callable
        ``(x: np.ndarray, m_external: int) -> (value, wall_time, report)``
        that bypasses :func:`make_multi_fidelity_objective`.  When supplied,
        *scheme*/*kernel*/*report_fields_by_layer* are still recorded in the
        returned :class:`BOResult` but the cascade is not invoked.

    Returns
    -------
    BOResult
        Frozen record with the full eval history, per-fidelity counts and
        wall times, final GP hyperparameters, and the recommended incumbent.

    Raises
    ------
    ValueError
        If neither or both of *budget_evals* and *budget_seconds* are set; if
        *report_fields_by_layer* is empty; if *cost_table* is missing entries
        for layers in *report_fields_by_layer*; if *bounds* is empty; if
        *budget_evals* is below 2; or if *budget_evals* leaves no room for the
        full initial design plus the final HF re-evaluation
        (``budget_evals - 1 < resolved n_init``; the ``-1`` reserves the final
        HF slot).  The init-design layout puts HF anchors last, so silent
        truncation under a tight budget would drop exactly the paired
        HF/cheap rows the ICM kernel needs to identify off-diagonal task
        correlations (Wu et al. 2020 §3.1) — we surface the constraint up
        front instead.
    """
    if (budget_evals is None) == (budget_seconds is None):
        raise ValueError(
            "exactly one of budget_evals or budget_seconds must be set"
        )
    if budget_evals is not None and budget_evals < 2:
        raise ValueError(
            f"budget_evals must be >= 2 (>=1 for init + 1 final HF re-eval), "
            f"got {budget_evals}"
        )
    if not report_fields_by_layer:
        raise ValueError("report_fields_by_layer must not be empty")
    if not bounds:
        raise ValueError("bounds must be non-empty")
    if not (0.0 <= hf_explore_bias <= 1.0):
        raise ValueError(
            f"hf_explore_bias must lie in [0, 1], got {hf_explore_bias}"
        )
    if hf_priority_warmup < 0:
        raise ValueError(
            f"hf_priority_warmup must be >= 0, got {hf_priority_warmup}"
        )
    if adaptive_hf_floor is not None:
        # NaN check via self-comparison (adaptive_hf_floor != adaptive_hf_floor
        # is True for NaN); a strict ``< 1.0`` then catches negative and
        # subunit values.  ``α = 1`` makes effective HF cost = cheap cost
        # (pure exploration); larger ``α`` keeps a residual cost penalty.
        if adaptive_hf_floor != adaptive_hf_floor or adaptive_hf_floor < 1.0:
            raise ValueError(
                "adaptive_hf_floor must be None or a float >= 1.0, "
                f"got {adaptive_hf_floor}"
            )
    if adaptive_hf_explore_bias is not None:
        # NaN check via self-comparison; β must lie in [0, 1] to be
        # interpretable as a quota target.  β = 0 collapses the formula
        # back to the static ``hf_explore_bias`` (mechanism a no-op);
        # β = 1 makes the upper bound on the adaptive lift equal to a
        # full HF-only quota when the GP is maximally uncertain.
        if (
            adaptive_hf_explore_bias != adaptive_hf_explore_bias
            or not (0.0 <= adaptive_hf_explore_bias <= 1.0)
        ):
            raise ValueError(
                "adaptive_hf_explore_bias must be None or a float in [0, 1], "
                f"got {adaptive_hf_explore_bias}"
            )
    # 47.3k.2: NaN check via self-comparison + strict positivity + finite.
    # The threshold scales the relative-variance criterion; ``<= 0`` would
    # disable the guard entirely (always-fire), and NaN/inf would make the
    # comparison ill-defined.
    if (
        variance_guard_relative_threshold
        != variance_guard_relative_threshold
        or variance_guard_relative_threshold <= 0.0
        or not np.isfinite(variance_guard_relative_threshold)
    ):
        raise ValueError(
            "variance_guard_relative_threshold must be a strictly positive "
            f"finite float, got {variance_guard_relative_threshold}"
        )
    # 47.3k.3: NaN check via self-comparison + non-negative + finite.  ``0.0``
    # is accepted (no-op contract: traverses the wrapper but adds ``+0`` per
    # HF candidate, bytewise-identical to ``None`` short-circuit).  Negative
    # bonuses would push the acquisition AWAY from HF — the opposite of the
    # mechanism's intent — so we reject them explicitly rather than silently
    # accepting an inverted contract.
    if hf_acquisition_bonus is not None:
        if (
            hf_acquisition_bonus != hf_acquisition_bonus
            or hf_acquisition_bonus < 0.0
            or not np.isfinite(hf_acquisition_bonus)
        ):
            raise ValueError(
                "hf_acquisition_bonus must be None or a non-negative finite "
                f"float, got {hf_acquisition_bonus}"
            )
    # 47.6b.3.2c.1: case-sensitive set membership.  ``"mean"`` is the
    # default and is bytewise-equivalent to the pre-47.6b.3.2c code path;
    # ``"voronoi"`` (47.6b.3.2c.2) masks the recommendation grid by L2
    # distance to recorded sentinel ``x``; ``"ucb"`` is reserved for plan
    # item 47.6b.3.2c.3 and is still surfaced eagerly because the in-loop
    # variance guard wraps :func:`_recommend_incumbent` in ``except
    # Exception:`` and would otherwise silently swallow the stub's
    # :class:`NotImplementedError`.  Case-insensitive matches (``"MEAN"``)
    # are rejected so a mutation that lower-cases the kwarg is caught
    # loudly.
    if recommendation_strategy not in {"mean", "voronoi", "ucb"}:
        raise ValueError(
            "recommendation_strategy must be one of "
            f"{{'mean', 'voronoi', 'ucb'}}, got {recommendation_strategy!r}"
        )
    # 47.6b.3.2c.2: ``voronoi_radius`` validation.  Strict ``> 0`` (a zero
    # radius would mask nothing — defeats the mechanism).  NaN check via
    # self-comparison; finiteness rejects ``inf``/``-inf``.  Validated
    # unconditionally so the kwarg surface is sound regardless of strategy
    # (catches mutations that rely on validation-when-used).
    if (
        voronoi_radius != voronoi_radius
        or voronoi_radius <= 0.0
        or not np.isfinite(voronoi_radius)
    ):
        raise ValueError(
            "voronoi_radius must be a strictly positive finite float, "
            f"got {voronoi_radius}"
        )
    # 47.6b.3.2c.3: ``ucb_beta`` validation.  ``>= 0`` and finite; ``0``
    # collapses the UCB ranking to mean-only (no-op contract for the
    # default-off equivalence test).  NaN check via self-comparison;
    # finiteness rejects ``inf``/``-inf``.  Validated unconditionally so
    # the kwarg surface is sound regardless of strategy (catches mutations
    # that defer the check to the strategy branch).
    if (
        ucb_beta != ucb_beta
        or ucb_beta < 0.0
        or not np.isfinite(ucb_beta)
    ):
        raise ValueError(
            f"ucb_beta must be a non-negative finite float, got {ucb_beta}"
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    fidelity_levels = tuple(sorted(report_fields_by_layer))
    K = len(fidelity_levels)
    hf_level = fidelity_levels[-1]
    target_fid_idx = K - 1
    int_to_ext = dict(enumerate(fidelity_levels))
    ext_to_int = {f: i for i, f in int_to_ext.items()}

    if cost_table is None:
        missing = [f for f in fidelity_levels if f not in DEFAULT_COST_TABLE]
        if missing:
            raise ValueError(
                f"DEFAULT_COST_TABLE has no entries for layers {missing}; "
                "pass an explicit cost_table"
            )
        cost_table_resolved = {f: DEFAULT_COST_TABLE[f] for f in fidelity_levels}
    else:
        missing = [f for f in fidelity_levels if f not in cost_table]
        if missing:
            raise ValueError(
                f"cost_table missing entries for layers {missing}"
            )
        cost_table_resolved = {f: cost_table[f] for f in fidelity_levels}
    floored_costs = apply_cost_floor(cost_table_resolved)

    if objective is None:
        objective = make_multi_fidelity_objective(
            scheme, kernel, report_fields_by_layer
        )

    bounds_t = tuple((float(lo), float(hi)) for lo, hi in bounds)
    d = len(bounds_t)

    # 47.3f: dimension-scaled HF anchor default.  The historical default of
    # ``3`` under-resolves higher-dimensional problems (e.g. 2D Branin needed
    # 4–7 HF anchors empirically to reach the basin floor); ``max(3, d + 2)``
    # scales with ``d`` while preserving the ``3`` floor for existing 1D/2D
    # callers.  Resolved at the run_mfbo layer (not in build_initial_design)
    # so direct callers of build_initial_design and existing TestDOE fixtures
    # keep working unchanged.
    hf_anchors_resolved = (
        hf_anchors if hf_anchors is not None else max(3, d + 2)
    )

    # 47.3f / 47.3k.1: minimum acquisition iterations before the variance
    # guard can fire.  Default ``max(_MIN_ACQ_ITERATIONS_FLOOR, K)``: at
    # least ``K`` so every fidelity gets a chance to be picked under cost-
    # aware acquisition before we declare convergence, and at least
    # :data:`_MIN_ACQ_ITERATIONS_FLOOR` so the GP's posterior at the
    # incumbent has time to become non-uniform after the DOE.  The
    # :data:`_MIN_ACQ_ITERATIONS_FLOOR` was raised from ``5`` to ``15`` in
    # 47.3k.1 after the AugmentedBranin sweep showed ``min_acq=20`` lifts
    # the empirical floor from ``3.55`` to ``1.25`` (3-seed best, 99 s
    # budget-exit) — the GP-uniform-collapse failure mode the 47.3d
    # combined absolute+relative criterion only partially defends against
    # needs more acquisition headroom on smooth synthetic objectives.
    min_acq_iters = _resolve_min_acq_iters(
        min_acquisition_iterations, K
    )

    # Resolve n_init to the same default that build_initial_design uses
    # (Loeppky et al. 2009: 5*d + 3) so we can validate the budget against
    # the actual initial-design size.  The init layout is [cheap | mid | hf];
    # the BO loop reserves one slot for the final HF re-eval; truncation
    # would silently drop HF anchors first (47.3b.1).
    resolved_n_init = n_init if n_init is not None else 5 * d + 3
    if budget_evals is not None and budget_evals - 1 < resolved_n_init:
        raise ValueError(
            f"budget_evals={budget_evals} too small for initial design "
            f"n_init={resolved_n_init}: need budget_evals >= n_init + 1 "
            f"(one slot reserved for the final HF re-eval at the incumbent). "
            f"Either raise budget_evals or pass a smaller n_init."
        )

    X_init, fid_init = build_initial_design(
        bounds_t,
        fidelity_levels,
        n_init=n_init,
        hf_anchors=hf_anchors_resolved,
        seed=seed,
    )

    eval_history: list[BOEval] = []
    n_evals_per_fid: dict[int, int] = {f: 0 for f in fidelity_levels}
    wall_time_per_fid: dict[int, float] = {f: 0.0 for f in fidelity_levels}

    t_start = time.perf_counter()

    def evaluate(x: np.ndarray, fid_internal: int) -> BOEval:
        ext_layer = int_to_ext[int(fid_internal)]
        x_arr = np.asarray(x, dtype=float)
        value, wt, rep = objective(x_arr, ext_layer)
        try:
            params = params_from_vector(kernel, x_arr)
        except Exception:
            params = {}
        ev = BOEval(
            x=x_arr.copy(),
            params=params,
            fidelity=ext_layer,
            value=float(value),
            wall_time=float(wt),
            report=rep,
        )
        eval_history.append(ev)
        n_evals_per_fid[ext_layer] += 1
        wall_time_per_fid[ext_layer] += float(wt)
        if verbose:
            print(
                f"[run_mfbo] eval #{len(eval_history)} layer={ext_layer} "
                f"value={value:.6g} wt={wt:.3f}s"
            )
        return ev

    def acq_budget_exhausted() -> bool:
        # Reserve one slot for the final HF re-evaluation at the incumbent.
        if budget_evals is not None and len(eval_history) >= budget_evals - 1:
            return True
        if (
            budget_seconds is not None
            and (time.perf_counter() - t_start) >= budget_seconds
        ):
            return True
        return False

    # Initial design (truncated when the budget has no room for it).
    for x_i, m_i in zip(X_init, fid_init):
        if acq_budget_exhausted():
            break
        evaluate(x_i, int(m_i))
    # 47.3f: post-init checkpoint so the variance guard can short-circuit
    # until ``min_acq_iters`` acquisition iterations have completed.
    n_init_actual = len(eval_history)

    stop_reason = "budget"
    converged = False
    final_model: MultiTaskGP | None = None

    while not acq_budget_exhausted():
        finite_mask = np.array(
            [
                np.isfinite(e.value) and e.value < _BO_SENTINEL / 2
                for e in eval_history
            ]
        )
        n_finite = int(finite_mask.sum())
        # Need at least K finite rows to identify the per-task hyperparameters,
        # and at least 2 to fit the marginal likelihood at all.
        if n_finite < max(2, K):
            stop_reason = "error"
            break

        # 47.6b.3.1: include sentinel rows in the GP fit at a per-fidelity
        # clamped value when ``clamp_sentinel_rows=True`` (default).  See
        # ``run_mfbo``'s ``clamp_sentinel_rows`` docstring entry for the
        # motivation.  Per-fidelity is essential because Y scales differ
        # across fidelities (e.g. layer1.boundary_gv_err ~ 0.02 vs
        # layer3.max_stab_eig ~ -1e-4); a global clamp would destabilise
        # the ICM kernel.  When a fidelity has fewer than 2 finite rows,
        # that fidelity's sentinels are filtered (insufficient data to
        # compute a meaningful clamp).  Row order in Y_train is preserved
        # from eval_history so the GP fit is bytewise-equivalent to the
        # pre-fix filter path when no sentinels exist (mutation guard:
        # changing row order can alter the marginal-likelihood optimiser's
        # convergence path on smooth synthetic objectives).
        if clamp_sentinel_rows:
            # Per-fidelity clamp values + which fidelities have ≥ 2 finite
            # rows (the others fall back to filter).
            clamp_vals: dict[int, float] = {}
            clamp_eligible: dict[int, bool] = {}
            for fid_int in range(K):
                fid_ext = int_to_ext[fid_int]
                fin_vals = np.array(
                    [
                        e.value
                        for e in eval_history
                        if e.fidelity == fid_ext
                        and np.isfinite(e.value)
                        and e.value < _BO_SENTINEL / 2
                    ]
                )
                if fin_vals.size >= 2:
                    clamp_vals[fid_ext] = float(fin_vals.max()) + 3.0 * float(
                        fin_vals.std(ddof=0)
                    )
                    clamp_eligible[fid_ext] = True
                else:
                    clamp_eligible[fid_ext] = False

            kept_evals: list[BOEval] = []
            kept_values: list[float] = []
            for e in eval_history:
                is_finite = (
                    np.isfinite(e.value) and e.value < _BO_SENTINEL / 2
                )
                if is_finite:
                    kept_evals.append(e)
                    kept_values.append(float(e.value))
                elif clamp_eligible.get(e.fidelity, False):
                    kept_evals.append(e)
                    kept_values.append(clamp_vals[e.fidelity])
                # else: fall back to filter (drop this row from the GP fit).
            if len(kept_evals) < max(2, K):
                stop_reason = "error"
                break
            X_train = np.array(
                [
                    np.concatenate([e.x, [float(ext_to_int[e.fidelity])]])
                    for e in kept_evals
                ]
            )
            Y_train = np.array(kept_values)
        else:
            finite_evals = [e for e, m in zip(eval_history, finite_mask) if m]
            X_train = np.array(
                [
                    np.concatenate([e.x, [float(ext_to_int[e.fidelity])]])
                    for e in finite_evals
                ]
            )
            Y_train = np.array([e.value for e in finite_evals])

        try:
            model = build_mf_gp(
                X_train, Y_train, fidelity_dim=d, num_fidelities=K
            )
        except Exception:
            stop_reason = "error"
            break
        final_model = model

        # Incumbent + variance guard.  The variance check is best-effort: a
        # numerical hiccup in posterior() should not abort the run.
        #
        # Scale: ``model.posterior()`` returns variance in the original Y
        # scale (BoTorch's Standardize outcome_transform auto-untransforms).
        # The absolute threshold uses ``spread_hf`` — the range of HF-only
        # Y values — rather than the full-multi-fidelity ``spread``.  In
        # MF-BO the GP posterior at the incumbent lives at the target
        # fidelity, so the relevant signal scale is HF-only.  Using full-Y
        # spread inflates the threshold whenever per-fidelity bias
        # dominates Y_train (e.g. cheap layers offset by hundreds while HF
        # lives near zero — a common pattern in the cascade), causing the
        # guard to fire after the initial design before any acquisition
        # iteration runs.
        #
        # The guard also requires a *relative* criterion: ``var_inc`` must
        # be small compared to the maximum posterior variance over a Sobol'
        # grid at HF.  This defends against the GP collapsing uniformly to
        # the noise floor on synthetic noise-free data — where every point
        # looks confident but no actual convergence has occurred (the
        # marginal-likelihood optimiser drives the noise hyperparameter to
        # the ``GreaterThan(1e-9)`` floor when there is no observable
        # signal noise, so absolute-only criteria fire spuriously).  When
        # the GP has genuine exploration uncertainty, ``max_var >> var_inc``
        # and both conditions can fire; when the GP is uniformly
        # overconfident, ``max_var ≈ var_inc`` and the relative condition
        # blocks the exit.
        #
        # The guard is skipped while fewer than 2 finite HF observations
        # exist; with 0–1 HF rows the GP cannot constrain HF-only spread.
        try:
            # 47.3f: short-circuit guard until enough acquisition iterations
            # have run.  Defends against the GP collapsing uniformly to its
            # noise floor on smooth synthetic objectives where the absolute
            # + relative variance criterion can fire after just the initial
            # design.  Counts only acquisition iterations, NOT init evals.
            n_acq_iters_done = len(eval_history) - n_init_actual
            if n_acq_iters_done < min_acq_iters:
                raise _SkipGuard
            Y_hf = Y_train[X_train[:, d] == target_fid_idx]
            if Y_hf.size < 2:
                raise _SkipGuard
            # 47.6b.3.2c.2: thread sentinel x's into the helper for the
            # ``"voronoi"`` strategy.  ``_collect_sentinel_x`` filters
            # ``eval_history`` by the existing 47.3b finite-mask convention
            # (``np.isfinite(value) and value < _BO_SENTINEL/2``); the
            # helper internally no-ops when the strategy is ``"mean"`` or
            # the array is empty.
            sentinel_x_loop = _collect_sentinel_x(eval_history)
            x_inc_loop = _recommend_incumbent(
                model,
                bounds_t,
                target_fid_idx,
                d,
                seed,
                strategy=recommendation_strategy,
                sentinel_x=sentinel_x_loop,
                voronoi_radius=voronoi_radius,
                ucb_beta=ucb_beta,
            )
            X_inc_full = torch.cat(
                [
                    torch.as_tensor(x_inc_loop, dtype=torch.float64),
                    torch.tensor([float(target_fid_idx)], dtype=torch.float64),
                ]
            ).unsqueeze(0)
            with torch.no_grad():
                var_inc = float(model.posterior(X_inc_full).variance.item())
                # Sobol' grid at HF for the relative-variance criterion.
                grid_sobol = torch.quasirandom.SobolEngine(
                    d, scramble=True, seed=seed
                )
                grid_raw = grid_sobol.draw(256).double()
                bounds_arr = torch.tensor(
                    [list(b) for b in bounds_t], dtype=torch.float64
                )
                lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]
                grid_X = lo + grid_raw * (hi - lo)
                grid_full = torch.cat(
                    [
                        grid_X,
                        torch.full(
                            (256, 1),
                            float(target_fid_idx),
                            dtype=torch.float64,
                        ),
                    ],
                    dim=1,
                )
                max_var = float(
                    model.posterior(grid_full).variance.max().item()
                )
            spread_hf = max(
                float(np.max(Y_hf)) - float(np.min(Y_hf)), 1e-12
            )
            absolute_fired = var_inc < 1e-6 * spread_hf ** 2
            relative_fired = _variance_guard_relative_fired(
                var_inc, max_var, variance_guard_relative_threshold
            )
            if absolute_fired and relative_fired:
                stop_reason = "variance"
                converged = True
                break
        except _SkipGuard:
            pass
        except Exception:
            pass

        # Stagnation guard: at least ``window + 1`` finite HF evals + the
        # running best is older than the last ``window``.  Logic factored
        # into :func:`_stagnation_triggered` for unit-testability (47.3e).
        hf_finite = [
            e
            for e in eval_history
            if e.fidelity == hf_level
            and np.isfinite(e.value)
            and e.value < _BO_SENTINEL / 2
        ]
        if _stagnation_triggered(hf_finite):
            stop_reason = "stagnation"
            converged = True
            break

        # 47.3i: adaptive HF cost floor.  When the HF posterior is still
        # uncertain AND the cheap surrogate is well-fit, lift the effective
        # HF cost toward the cheap cost so the cost-aware utility doesn't
        # under-sample HF.  Composes with hf_priority_warmup (47.3h, runs
        # first) and hf_explore_bias (47.3g, runs after — both jointly steer
        # ``fidelity_choices`` once the cost model is built).  Best-effort:
        # any failure in the predicate computation falls back to the static
        # cost table so a numerical hiccup cannot abort the loop.
        # Shared HF uncertainty signal (47.3j.1 Gap 3): both adaptive
        # mechanisms below consume the same ``(var_hf_grid, spread_hf)``
        # pair, computed once via :func:`_compute_hf_uncertainty_signal`.
        # ``None`` indicates the signal is unavailable (insufficient HF
        # rows or a numerical hiccup) — both mechanisms then fall back to
        # their static branches.
        if adaptive_hf_floor is not None or adaptive_hf_explore_bias is not None:
            hf_uncertainty_signal = _compute_hf_uncertainty_signal(
                model, X_train, Y_train, target_fid_idx, d, bounds_t, seed
            )
        else:
            hf_uncertainty_signal = None

        effective_cost_table = cost_table_resolved
        if adaptive_hf_floor is not None and hf_uncertainty_signal is not None:
            try:
                var_hf_grid, spread_hf_now = hf_uncertainty_signal
                n_cheap_finite = int(
                    np.sum(X_train[:, d] != target_fid_idx)
                )
                hf_uncertain = (
                    var_hf_grid
                    > _ADAPTIVE_HF_FLOOR_TAU * spread_hf_now ** 2
                )
                cheap_well_fit = n_cheap_finite >= max(2 * d, K)
                if hf_uncertain and cheap_well_fit:
                    cheap_costs = [
                        cost_table_resolved[ext]
                        for ext in fidelity_levels[:-1]
                    ]
                    if cheap_costs:
                        cheap_min = min(cheap_costs)
                        effective_hf_cost = min(
                            cost_table_resolved[hf_level],
                            adaptive_hf_floor * cheap_min,
                        )
                        effective_cost_table = dict(cost_table_resolved)
                        effective_cost_table[hf_level] = effective_hf_cost
            except Exception:
                effective_cost_table = cost_table_resolved

        # Cost-aware acquisition + mixed continuous/discrete optimiser.
        try:
            cost_util = build_cost_model(effective_cost_table, fidelity_dim=d)
            _, optimize = build_acquisition(
                model,
                cost_util,
                target_fid_idx,
                num_fantasies=num_fantasies,
                hf_acquisition_bonus=hf_acquisition_bonus,
            )
            # 47.3h: HF priority warmup runs first.  When
            # ``hf_priority_warmup > 0``, force the first that-many
            # acquisition iterations to HF regardless of cost-aware
            # utility or explore-bias quota.  Composes with the 47.3g
            # quota: warmup wins for the first N picks, then quota
            # kicks in (or both leave the choices unrestricted).
            #
            # 47.3g: HF explore-bias quota.  When ``hf_explore_bias > 0``,
            # restrict the acquisition's ``fidelity_choices`` to ``[HF]``
            # only whenever the running HF fraction (among acquisition
            # picks made so far) is below the target.  qMFKG then picks
            # the optimal ``x`` for HF specifically rather than re-using
            # an ``x_next`` chosen for a cheaper layer.  The initial
            # design's stratification is excluded from the fraction —
            # ``build_initial_design`` already governs that.
            #
            # 47.3j: when ``adaptive_hf_explore_bias`` is set, scale the
            # static quota by HF posterior uncertainty.  ``effective_bias
            # = max(hf_explore_bias, β * var_hf_grid / (var_hf_grid +
            # spread_hf**2))``: the second term tends toward β when the
            # GP says HF is still uncertain, toward 0 when it tightens.
            # Best-effort: any failure in the predicate computation
            # falls back to the static ``hf_explore_bias``.
            fidelity_choices: list[int] = list(range(K))
            n_acq_done = len(eval_history) - n_init_actual
            effective_bias = hf_explore_bias
            if (
                adaptive_hf_explore_bias is not None
                and hf_uncertainty_signal is not None
            ):
                try:
                    var_hf_bias, spread_hf_bias = hf_uncertainty_signal
                    denom = var_hf_bias + spread_hf_bias ** 2
                    if denom > 0.0:
                        adaptive_term = (
                            adaptive_hf_explore_bias
                            * var_hf_bias
                            / denom
                        )
                        # Snap essentially-zero values to a clean 0
                        # so the schedule reverts cleanly to the
                        # static ``hf_explore_bias`` when the GP is
                        # certain.  Without this, a finite-precision
                        # ``var_hf_bias / spread_hf**2`` ratio of
                        # ~1e-17 leaves ``effective_bias`` strictly
                        # positive and ``effective_bias > 0.0``
                        # would still trigger the quota check, with
                        # ``projected_no_hf = 0 < ~1e-17`` forcing
                        # HF on the first acquisition iteration.
                        if adaptive_term < 1e-6:
                            adaptive_term = 0.0
                        effective_bias = max(
                            hf_explore_bias, adaptive_term
                        )
                except Exception:
                    effective_bias = hf_explore_bias
            if hf_priority_warmup > 0 and n_acq_done < hf_priority_warmup:
                fidelity_choices = [target_fid_idx]
            elif effective_bias > 0.0:
                acq_evals = eval_history[n_init_actual:]
                n_acq_hf_done = sum(
                    1 for e in acq_evals if e.fidelity == hf_level
                )
                # Even if we pick HF this iteration, the post-iter fraction
                # is ``(n_hf + 1) / (n_done + 1)``.  We force HF whenever
                # the no-HF projection ``n_hf / (n_done + 1)`` is below
                # the target — i.e. when picking anything other than HF
                # would leave the running fraction below quota.
                projected_no_hf = n_acq_hf_done / (n_acq_done + 1)
                if projected_no_hf < effective_bias:
                    fidelity_choices = [target_fid_idx]
            x_next, fid_next, _ = optimize(bounds_t, fidelity_choices)
        except Exception:
            stop_reason = "error"
            break

        evaluate(x_next, fid_next)

    # Final recommendation: posterior mean over a 1024-pt Sobol' grid at HF.
    # 47.6b.3.2c.2: ``info_out`` collects the optional ``voronoi_fallback``
    # flag from the helper and we propagate it to ``BOResult.extras`` below.
    final_recommend_info: dict = {}
    if final_model is not None:
        try:
            sentinel_x_final = _collect_sentinel_x(eval_history)
            x_inc = _recommend_incumbent(
                final_model,
                bounds_t,
                target_fid_idx,
                d,
                seed,
                strategy=recommendation_strategy,
                sentinel_x=sentinel_x_final,
                voronoi_radius=voronoi_radius,
                ucb_beta=ucb_beta,
                info_out=final_recommend_info,
            )
        except Exception:
            x_inc = X_init[0].copy() if len(X_init) else np.zeros(d)
    else:
        # GP never fit (initial design entirely infeasible, or budget exhausted
        # before any iteration).  Fall back to best observed HF, else first
        # design point.
        hf_obs = [
            e
            for e in eval_history
            if e.fidelity == hf_level
            and np.isfinite(e.value)
            and e.value < _BO_SENTINEL / 2
        ]
        if hf_obs:
            x_inc = min(hf_obs, key=lambda e: e.value).x.copy()
        elif len(X_init):
            x_inc = X_init[0].copy()
        else:
            x_inc = np.zeros(d)

    final_eval = evaluate(x_inc, target_fid_idx)
    total_time = time.perf_counter() - t_start

    hf_eval_history = tuple(e for e in eval_history if e.fidelity == hf_level)

    if final_model is not None:
        with torch.no_grad():
            ls = final_model.covar_module.kernels[0].lengthscale.detach().cpu().numpy()
            W = final_model.covar_module.kernels[1].covar_factor.detach().cpu().numpy()
            v = final_model.covar_module.kernels[1].var.detach().cpu().numpy()
            noise_arr = final_model.likelihood.noise.detach().cpu().numpy()
        gp_hyp = {
            "lengthscale": np.atleast_1d(ls.squeeze()).tolist(),
            "icm_W": W.tolist(),
            "icm_var": np.atleast_1d(v.squeeze()).tolist(),
            "noise": float(np.atleast_1d(noise_arr.squeeze())[0]),
        }
    else:
        gp_hyp = {}

    sentinel_count = sum(
        1
        for e in eval_history
        if not (np.isfinite(e.value) and e.value < _BO_SENTINEL / 2)
    )
    if clamp_sentinel_rows:
        # 47.6b.3.1 / 47.6b.3.1.1: per-fidelity sentinel *occurrence* counts
        # (treatment-agnostic: rows at a fidelity with ≥ 2 finite siblings
        # were clamped, rows at a fidelity with < 2 finite siblings fell
        # back to filter — both are tallied here under the same key).  The
        # field name is ``n_sentinel_per_fidelity`` rather than
        # ``...clamped...`` because the count would otherwise overstate
        # what happened on the fallback path.  The ``clamp_sentinel_rows
        # =False`` branch keeps the pre-47.6b.3.1 ``n_sentinel_filtered``
        # global tally for backwards compatibility.
        sentinel_per_fid: dict[int, int] = {f: 0 for f in fidelity_levels}
        for e in eval_history:
            if not (np.isfinite(e.value) and e.value < _BO_SENTINEL / 2):
                sentinel_per_fid[e.fidelity] += 1
        extras_payload = {
            "n_sentinel_per_fidelity": sentinel_per_fid,
        }
    else:
        extras_payload = {"n_sentinel_filtered": sentinel_count}

    # 47.6b.3.2c.2: when the ``"voronoi"`` recommendation fell back to
    # unmasked argmin (every Sobol' grid point was within ``voronoi_radius``
    # of some sentinel), surface the flag in extras so the caller can
    # detect the degenerate case.  Absent under any other strategy or when
    # the masking step found feasible candidates.
    if final_recommend_info.get("voronoi_fallback"):
        extras_payload["voronoi_fallback"] = True

    return BOResult(
        best_x=np.asarray(x_inc, dtype=float).copy(),
        best_params=final_eval.params,
        best_objective=final_eval.value,
        best_report=final_eval.report,
        method="BoTorch-qMFKG",
        scheme=scheme,
        kernel=kernel,
        bounds=bounds_t,
        fidelity_levels=fidelity_levels,
        hf_level=hf_level,
        report_fields_by_layer=dict(report_fields_by_layer),
        cost_model=floored_costs,
        n_evals_per_fidelity=dict(n_evals_per_fid),
        wall_time_per_fidelity=dict(wall_time_per_fid),
        total_compute_time=total_time,
        eval_history=tuple(eval_history),
        hf_eval_history=hf_eval_history,
        gp_hyperparameters=gp_hyp,
        seed=seed,
        converged=converged,
        stop_reason=stop_reason,
        extras=extras_payload,
    )


__all__: list[str] = [
    "_BO_SENTINEL",
    "BOEval",
    "BOResult",
    "DEFAULT_COST_TABLE",
    "apply_cost_floor",
    "build_acquisition",
    "build_cost_model",
    "build_initial_design",
    "build_mf_gp",
    "make_multi_fidelity_objective",
    "run_mfbo",
]
