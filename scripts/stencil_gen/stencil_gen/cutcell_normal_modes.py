"""psi-parameterized GKS Kreiss check for cut-cell closures (plan 48 item 1.2).

Bridges the cached cut-cell evaluator (``cutcell_eval``, plan 48 item 1.1)
to the rigorous Kreiss determinant check (``gks_kreiss``): closure rows are
assembled numerically at a (psi, alpha) point and fed — together with their
grid-node ``row_indices`` — to ``kreiss_stability_check``, which dispatches
to the augmented Kreiss matrix (plan 48 item 0.2) with the item-0.5
witness-search robustness fixes (column equilibration, eigensolution
|s|-bound, structural near-origin discrimination).

The **Dirichlet (wall-eliminated) block is the canonical analysis object**
(plan 48 constraint): the evaluator returns the (R-1) closure rows with the
wall column dropped, which is exactly what ``derivative.cpp`` time-steps.
Sign convention: ``gks_kreiss`` analyzes ``u_t = -sum_k a_k u_{n+k}`` (the
``kappa_roots`` docstring convention); the evaluator's raw
derivative-operator rows are passed through unflipped, matching what
``brady2d_stability._extract_stencil_data`` feeds to the same check.

Feasibility guard (plan 48 item 1.2): before any assembly the closure's
structural denominators (``CutCellEvaluator.denominators``) are evaluated at
(psi, alpha); if any vanishes (|denominator| < ``feasibility_tol``) the
check returns a distinct :class:`CutCellInfeasible` marker instead of a
stability verdict.  Denominator vanishing is a *parameterization* failure —
the symbolic solve degenerates there — and must never be conflated with GKS
instability (the marker deliberately has no ``is_stable`` attribute).

Worst-case-over-psi margin (plan 48 item 1.3): sweeping
:func:`kreiss_cutcell_check` over a psi grid clustered near 0 (the ILW
literature's critical regime) and matched to the C++ clamp
``[1e-4, 1 - 1e-4]`` yields a single worst-cased margin
(:class:`CutCellPsiSweep`) ready for the optimizer (wiring is plan 49).
Infeasible psi points are partitioned out — never folded into the margin.

No time integration happens here (plan 48 hard constraint): the check is
purely resolvent/root-locus based via ``kreiss_stability_check``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
from sympy import Expr, lambdify

from stencil_gen.cutcell_eval import CutCellEvaluator, make_cutcell_evaluator
from stencil_gen.gks_kreiss import KreissResult, kreiss_stability_check
from stencil_gen.temo import SchemeParams

__all__ = [
    "CutCellInfeasible",
    "CutCellPsiSweep",
    "default_psi_grid",
    "kreiss_cutcell_check",
    "kreiss_margin_over_psi",
]


FEASIBILITY_TOL = 1e-12
"""Default |denominator| threshold below which a (psi, alpha) point is
declared infeasible (plan 48 item 1.2)."""


@dataclass(frozen=True)
class CutCellInfeasible:
    """Distinct marker for an infeasible (psi, alpha) parameterization.

    Returned by :func:`kreiss_cutcell_check` when a structural denominator
    of the closure vanishes at the requested point.  This is NOT a
    stability verdict — the closure coefficients do not exist there, so
    neither "stable" nor "unstable" is meaningful — and the class
    deliberately carries no ``is_stable`` attribute so downstream code
    cannot silently treat it as one.  Callers discriminate with
    ``isinstance(result, CutCellInfeasible)`` or
    ``getattr(result, "infeasible", False)``.
    """

    scheme: SchemeParams
    """Scheme the check was requested for."""

    psi: float
    """Fractional wall position at which the parameterization fails."""

    alpha: tuple
    """The alpha point as ``(name, value)`` pairs in canonical
    ``alpha_symbols`` order."""

    vanishing: tuple
    """The vanishing denominators as ``(sympy_expr, value)`` pairs with
    ``|value| < feasibility_tol``."""

    feasibility_tol: float = FEASIBILITY_TOL
    """Threshold used for the vanishing test."""

    infeasible: bool = True
    """Always True; lets callers test ``getattr(r, "infeasible", False)``
    uniformly across :class:`CutCellInfeasible` and ``KreissResult``."""


@functools.lru_cache(maxsize=None)
def _denominator_funcs(
    denominators: frozenset, syms: tuple
) -> tuple[tuple[Expr, object], ...]:
    """Lambdify each denominator polynomial once per (denominators, syms).

    Keyed on the *value* of the frozenset (not the evaluator object) so a
    fresh compilation happens whenever the denominator set differs — e.g.
    in tests that substitute an engineered set on the cached evaluator.
    """
    return tuple((den, lambdify(syms, den)) for den in denominators)


def _vanishing_denominators(
    evaluator: CutCellEvaluator,
    psi: float,
    alpha: dict,
    tol: float,
) -> tuple[tuple[Expr, float], ...]:
    """Evaluate the closure's denominators at (psi, alpha).

    Returns the ``(expr, value)`` pairs with ``|value| < tol`` — empty when
    the parameterization is feasible.  Alpha validation (missing/unknown
    keys) is delegated to the evaluator's canonical ordering.
    """
    syms = (evaluator._psi_symbol, *evaluator.alpha_symbols)
    args = (float(psi), *evaluator._alpha_args(alpha))
    out = []
    for den, func in _denominator_funcs(evaluator.denominators, syms):
        value = float(func(*args))
        if abs(value) < tol:
            out.append((den, value))
    return tuple(out)


def _validate_closure(
    evaluator: CutCellEvaluator,
    boundary_rows: list,
    interior_offsets: np.ndarray,
) -> tuple[int, int, int]:
    """Validate the closure/interior dimension contract (plan 48 item 1.2).

    The augmented Kreiss matrix (item 0.2) needs m = R - 1 closure rows,
    p = L_left = -min(interior offsets) admissible roots for Re(s) > 0,
    and m - p >= 1 free nodal unknowns.  For the wall-eliminated Dirichlet
    blocks: E2_1 has 3 = 2 + 1 (m = nodal + p) and E4_1 has 4 = 2 + 2.
    All SHOCCS interiors are centered, so p must also equal the scheme
    half-width ``scheme.p``.

    Returns
    -------
    (m, p, n_nodal) : tuple[int, int, int]

    Raises
    ------
    ValueError
        If any part of the contract is violated (indicates an evaluator or
        derivation bug, not a user error).
    """
    m = len(boundary_rows)
    R = evaluator.dims.R
    if m != R - 1:
        raise ValueError(
            f"Expected m = R - 1 = {R - 1} Dirichlet closure rows, got {m}"
        )
    p = -int(np.min(np.asarray(interior_offsets, dtype=int)))
    if p != evaluator.scheme.p:
        raise ValueError(
            f"Admissible root count p = L_left = {p} != scheme half-width "
            f"p = {evaluator.scheme.p} (centered-interior contract)"
        )
    n_nodal = m - p
    if n_nodal < 1:
        raise ValueError(
            f"Augmented closure needs m - p >= 1 nodal unknowns; "
            f"got m = {m}, p = {p}"
        )
    return m, p, n_nodal


def kreiss_cutcell_check(
    scheme: SchemeParams,
    alpha: dict,
    psi: float,
    *,
    feasibility_tol: float = FEASIBILITY_TOL,
    **kreiss_kwargs,
) -> KreissResult | CutCellInfeasible:
    """Run the GKS Kreiss determinant check on a cut-cell Dirichlet closure.

    Assembles the (R-1) wall-eliminated Dirichlet closure rows at
    (psi, alpha) via the cached evaluator (plan 48 item 1.1) and calls
    ``kreiss_stability_check`` with their grid-node ``row_indices``
    (Dirichlet row i sits at grid node i), dispatching to the augmented
    Kreiss matrix (m = R - 1 rows > p admissible roots; item 0.2) with the
    item-0.5 robustness fixes.  Sign convention: the check analyzes
    ``u_t = -sum_k a_k u_{n+k}``; the evaluator's raw derivative-operator
    rows are passed through unflipped.

    Parameters
    ----------
    scheme : SchemeParams
        Cut-cell scheme (e.g. ``temo.E2_1``, ``temo.E4_1``).
    alpha : dict
        Mapping from alpha symbol (or name) to value; must cover all of
        the scheme's free parameters.
    psi : float
        Fractional wall position in (0, 1].
    feasibility_tol : float
        |denominator| threshold for the feasibility guard.
    **kreiss_kwargs
        Forwarded to ``kreiss_stability_check`` (e.g. ``s_grid_params``,
        ``sigma_tol``).  ``row_indices`` is supplied internally and may
        not be overridden.

    Returns
    -------
    KreissResult or CutCellInfeasible
        The stability verdict, or the distinct infeasible marker when any
        structural denominator vanishes at (psi, alpha) — a
        *parameterization* failure, never conflated with GKS instability.
    """
    if "row_indices" in kreiss_kwargs:
        raise TypeError(
            "row_indices is derived from the evaluator's Dirichlet row "
            "placement and may not be overridden"
        )
    psi = float(psi)
    if not 0.0 < psi <= 1.0:
        raise ValueError(f"psi must be in (0, 1], got {psi}")

    evaluator = make_cutcell_evaluator(scheme)

    # Feasibility guard BEFORE assembly (plan 48 item 1.2): a vanishing
    # denominator means the closure coefficients do not exist at this
    # point; return the distinct marker, not a stability verdict.
    vanishing = _vanishing_denominators(evaluator, psi, alpha, feasibility_tol)
    if vanishing:
        alpha_args = evaluator._alpha_args(alpha)
        return CutCellInfeasible(
            scheme=scheme,
            psi=psi,
            alpha=tuple(
                (s.name, v) for s, v in zip(evaluator.alpha_symbols, alpha_args)
            ),
            vanishing=vanishing,
            feasibility_tol=feasibility_tol,
        )

    boundary_rows, row_indices = evaluator.rows(psi, alpha)
    interior_weights, interior_offsets = evaluator.interior()
    _validate_closure(evaluator, boundary_rows, interior_offsets)

    return kreiss_stability_check(
        interior_weights,
        interior_offsets,
        boundary_rows,
        row_indices=[int(i) for i in row_indices],
        **kreiss_kwargs,
    )


# ---------------------------------------------------------------------------
# Plan 48 item 1.3: worst-case-over-psi margin
# ---------------------------------------------------------------------------


def default_psi_grid() -> np.ndarray:
    """The default psi grid for worst-case-over-psi sweeps (plan 48 item 1.3).

    Clustered near 0 — the ILW literature identifies psi -> 0 (the wall
    nearly on top of the first grid node) as the critical regime — and
    matched to the C++ clamp ``[1e-4, 1 - 1e-4]``:

        {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}
        ∪ Chebyshev-11 on [0.05, 0.95]
        ∪ {0.99, 0.999, 0.9999}

    The Chebyshev-11 block uses the 11 Chebyshev–Lobatto (second-kind /
    extrema) nodes ``0.5 + 0.45·cos(k·pi/10)``, k = 0..10, which include
    the interval endpoints 0.05 and 0.95 exactly, so the bulk block abuts
    the two clustered end blocks without a gap.

    Returns
    -------
    np.ndarray
        The 20 grid points, sorted ascending, all within
        ``[1e-4, 1 - 1e-4]``.
    """
    near_zero = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    cheb = 0.5 + 0.45 * np.cos(np.pi * np.arange(11) / 10)
    near_one = np.array([0.99, 0.999, 0.9999])
    return np.unique(np.concatenate([near_zero, cheb, near_one]))


@dataclass(frozen=True)
class CutCellPsiSweep:
    """Worst-case-over-psi Kreiss margin for a cut-cell closure (item 1.3).

    Aggregates per-psi :func:`kreiss_cutcell_check` results into the single
    worst-cased margin the optimizer consumes (wiring is plan 49).  Two
    orthogonal aggregations, never conflated:

    - **stability**: ``any_unstable`` is True iff any feasible psi returned
      ``is_stable=False`` (including anomaly-driven conservative fails,
      whose ``witness_sigma_min`` is inf and therefore does not enter the
      margin bookkeeping below);
    - **margin**: ``worst_sigma_min`` / ``worst_psi`` are the min (and
      argmin) over feasible psi of the per-psi ``witness_sigma_min`` —
      the column-equilibrated relative margin for these augmented m > p
      closures (plan 48 item 0.5a).  Only finite margins participate; if
      no psi produced a finite margin, ``worst_sigma_min`` is inf and
      ``worst_psi`` is None.

    Infeasible psi points (vanishing structural denominator — a
    *parameterization* failure, see :class:`CutCellInfeasible`) are listed
    in ``infeasible_psi`` and excluded from both aggregations.
    """

    worst_sigma_min: float
    """Min over feasible psi of the per-psi ``witness_sigma_min`` (inf if
    no psi produced a finite margin)."""

    worst_psi: float | None
    """The psi attaining ``worst_sigma_min`` (None if no finite margin)."""

    any_unstable: bool
    """True iff any feasible psi returned ``is_stable=False``."""

    infeasible_psi: list
    """psi values where the parameterization is infeasible (in grid
    order); their per-psi entries are :class:`CutCellInfeasible`."""

    per_psi: dict
    """Full per-psi results, keyed by float(psi):
    ``KreissResult`` (feasible) or :class:`CutCellInfeasible`."""


def kreiss_margin_over_psi(
    scheme: SchemeParams,
    alpha: dict,
    psi_grid=None,
    *,
    feasibility_tol: float = FEASIBILITY_TOL,
    **kreiss_kwargs,
) -> CutCellPsiSweep:
    """Worst-case the GKS Kreiss margin over a psi grid (plan 48 item 1.3).

    Runs :func:`kreiss_cutcell_check` at each psi of *psi_grid* (default:
    :func:`default_psi_grid` — clustered near 0, matched to the C++ clamp
    ``[1e-4, 1 - 1e-4]``) and aggregates into a :class:`CutCellPsiSweep`.
    Thanks to the cached evaluator (plan 48 item 1.1) the symbolic
    derivation cost is paid once per scheme, not per grid point.

    Parameters
    ----------
    scheme : SchemeParams
        Cut-cell scheme (e.g. ``temo.E2_1``, ``temo.E4_1``).
    alpha : dict
        Mapping from alpha symbol (or name) to value; must cover all of
        the scheme's free parameters.
    psi_grid : sequence of float, optional
        psi values to sweep; defaults to :func:`default_psi_grid`.
        Must be non-empty.
    feasibility_tol : float
        |denominator| threshold for the per-psi feasibility guard.
    **kreiss_kwargs
        Forwarded to :func:`kreiss_cutcell_check` (and thence to
        ``kreiss_stability_check``), e.g. ``s_grid_params``, ``sigma_tol``.

    Returns
    -------
    CutCellPsiSweep
        Worst-cased margin, stability flag, and the per-psi partition of
        feasible verdicts vs. infeasible parameterization points.
    """
    if psi_grid is None:
        psi_grid = default_psi_grid()
    psi_values = [float(p) for p in psi_grid]
    if not psi_values:
        raise ValueError("psi_grid must contain at least one psi value")

    per_psi: dict = {}
    infeasible_psi: list = []
    any_unstable = False
    worst_sigma_min = float("inf")
    worst_psi: float | None = None

    for psi in psi_values:
        result = kreiss_cutcell_check(
            scheme, alpha, psi,
            feasibility_tol=feasibility_tol,
            **kreiss_kwargs,
        )
        per_psi[psi] = result
        if isinstance(result, CutCellInfeasible):
            infeasible_psi.append(psi)
            continue
        if not result.is_stable:
            any_unstable = True
        sigma = float(result.witness_sigma_min)
        if np.isfinite(sigma) and sigma < worst_sigma_min:
            worst_sigma_min = sigma
            worst_psi = psi

    return CutCellPsiSweep(
        worst_sigma_min=worst_sigma_min,
        worst_psi=worst_psi,
        any_unstable=any_unstable,
        infeasible_psi=infeasible_psi,
        per_psi=per_psi,
    )
