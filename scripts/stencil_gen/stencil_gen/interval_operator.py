"""Interval-operator assembler for two-ended cut-cell lines (plan 48 item 3.1).

Assembles the N x N semi-discrete first-derivative operator for a fluid
interval (a single cut line) with cut-cell Dirichlet closures at BOTH
ends, mirroring ``derivative.cpp``'s per-line layout
(``src/operators/derivative.cpp:305-442``)::

    [ dense_left | circulant interior | dense_right ]

with the wall columns dropped.  In C++ the wall coefficients are stripped
into the CSR B matrix (they multiply the prescribed object-boundary
value); under the homogeneous Dirichlet analysis setting of plan 48 the
wall value is 0, so dropping the column is exact.  The **Dirichlet
(wall-eliminated) block is the canonical analysis object** (plan 48 hard
constraint) — it is what ``derivative.cpp`` time-steps.

Layout (unit grid spacing h = 1, matching every Python-side builder —
``brady2d_stability._build_classical_diff_matrix``,
``phs.build_diff_matrix_rbf``, and the test-local cut builders):

- **Left closure** (rows ``0 .. m_L-1``): the ``m = R-1`` wall-eliminated
  Dirichlet rows from the cached evaluator (plan 48 item 1.1),
  ``CutCellEvaluator.rows(psi_L, alpha)``; Dirichlet row i sits at grid
  node i with support columns ``0 .. t-1`` (t = T-1).
- **Interior** (rows ``m_L .. N-1-m_R``): the centered ``2p+1`` circulant
  stencil.
- **Right closure** (rows ``N-m_R .. N-1``): the stored left-end rows at
  ``psi_R`` mapped through the item-2.3 mirror convention —
  ``reflection.mirror_closure(..., from_physical_right=False)``, the
  production nu = 1 codegen revneg entry point.  The physical right-end
  row is the revneg image ``D[N-1-i, N-1-j] = -B[i, j]`` of stored
  left-end row i (``derivative.cpp``'s right blocks, built by
  ``st.nbs(..., right=true, ...)``); the negation is nu-odd-specific.

``psi_L = 1.0`` / ``psi_R = 1.0`` means a **uniform domain boundary** at
that end, matching the C++ convention of passing ``psi = 1.0`` to
``st.nbs`` for uncut grid boundaries: the cut-cell closure evaluated at
psi = 1 is exactly the production uniform-domain-boundary closure (at
psi = 1 the wall sits one grid spacing outside node 0; item 1.1
established that rows ``0 .. m-2`` then coincide with the uniform
boundary rows ``B_u[1..r-1]`` while the last, near-interior row retains
its wall coupling by construction).

Sign convention: the returned matrix is the raw derivative operator D
(same as all repo builders); semi-discrete advection stability is
analyzed on ``u_t = -D u``.  **No time integration happens here** (plan
48 hard constraint) — this module assembles (item 3.1) and evaluates
resolvent quantities (:func:`interval_stability_scan`, item 3.2: dense
eigenvalues and grid-sampled Kreiss constants; no matrix powers, no
stepping).

Minimum interval length (documented per plan item 3.1 test d): closure
ROWS may not overlap — a grid node cannot carry both a left and a right
closure equation — so the minimum assemblable length is
``N = m_L + m_R``.  At exactly that N the line has no interior row and
the operator is all-closure; this is exactly the dangerous
(short-cut-line) case and it still assembles.  Below it, or when a
closure's column support does not fit inside the domain, the assembler
raises :class:`IntervalTooShort` with a precise message.  Column-support
fit: the left support occupies columns ``0 .. t_L-1`` and the right
support columns ``N-t_R .. N-1``; both fit inside the domain iff
``max(t_L, t_R) <= N``, equivalently ``t_L + t_R - overlap <= N`` with
``overlap`` the realized column overlap of the two end-anchored supports
(column overlap between the two closures is permitted — left and right
closure rows may legally write into shared columns).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from stencil_gen.cutcell_eval import make_cutcell_evaluator
from stencil_gen.cutcell_normal_modes import (
    FEASIBILITY_TOL,
    _vanishing_denominators,
    default_psi_grid,
)
from stencil_gen.non_normality import kreiss_constant_estimate
from stencil_gen.reflection import mirror_closure
from stencil_gen.temo import SchemeParams

__all__ = [
    "IntervalCase",
    "IntervalScanResult",
    "IntervalTooShort",
    "build_interval_operator",
    "default_interval_psi_pairs",
    "interval_stability_scan",
    "refined_interval_s_grid",
]


class IntervalTooShort(ValueError):
    """The requested interval length N cannot host both closures.

    Raised by :func:`build_interval_operator` when closure rows would
    overlap (``N < m_L + m_R``) or a closure's column support does not
    fit inside the domain (``max(t_L, t_R) > N``).  See the module
    docstring for the minimum-N discussion.
    """


def build_interval_operator(
    scheme: SchemeParams,
    alpha: dict,
    psi_L: float,
    psi_R: float,
    N: int,
) -> np.ndarray:
    """Assemble the N x N two-ended cut-cell interval operator.

    Plan 48 item 3.1.  Mirrors ``derivative.cpp``'s
    ``[dense_left | circulant | dense_right]`` inner_block layout with
    wall columns dropped (homogeneous Dirichlet) and the right closure
    produced by the item-2.3 mirror convention
    (``mirror_closure(..., from_physical_right=False)`` — the nu = 1
    codegen revneg entry point).  ``psi_* = 1.0`` means a uniform domain
    boundary at that end (the C++ convention of passing psi = 1.0 to
    ``nbs``).

    Parameters
    ----------
    scheme : SchemeParams
        Cut-cell scheme (e.g. ``temo.E2_1``, ``temo.E4_1``); the closure
        rows come from the cached evaluator (item 1.1).
    alpha : dict
        Mapping from alpha symbol (or name) to numeric value; must cover
        all of the scheme's free parameters (validated by the evaluator).
    psi_L, psi_R : float
        Fractional wall positions in ``(0, 1]`` at the left / right end
        of the interval (1.0 = uniform domain boundary).
    N : int
        Number of fluid grid nodes on the line.

    Returns
    -------
    np.ndarray
        The N x N derivative matrix D (unit grid spacing); semi-discrete
        stability is analyzed on ``u_t = -D u``.

    Raises
    ------
    IntervalTooShort
        If ``N < m_L + m_R`` (closure rows may not overlap) or a
        closure's column support does not fit inside the domain.
    ValueError
        If a psi is outside ``(0, 1]``, N is not a positive integer, the
        alpha dict is malformed (via the evaluator), or the evaluated
        closure is non-finite at (psi, alpha) — a parameterization
        (denominator) failure; see ``cutcell_normal_modes`` for the
        item-1.2 feasibility guard callers should apply first.
    """
    for name, psi in (("psi_L", psi_L), ("psi_R", psi_R)):
        psi = float(psi)
        if not 0.0 < psi <= 1.0:
            raise ValueError(f"{name} must be in (0, 1], got {psi}")
    N = int(N)
    if N <= 0:
        raise ValueError(f"N must be a positive integer, got {N}")

    evaluator = make_cutcell_evaluator(scheme)
    interior_weights, interior_offsets = evaluator.interior()

    # Left closure: wall-eliminated Dirichlet rows at psi_L, row i at
    # grid node i, support columns 0 .. t-1.
    rows_L, idx_L = evaluator.rows(float(psi_L), alpha)

    # Right closure: stored left-end rows at psi_R mapped through the
    # item-2.3 mirror convention (nu = 1 codegen revneg entry point).
    rows_R_stored, idx_R = evaluator.rows(float(psi_R), alpha)
    mirrored = mirror_closure(
        rows_R_stored,
        (interior_weights, interior_offsets),
        from_physical_right=False,
        row_indices=[int(i) for i in idx_R],
    )

    m_L = len(rows_L)
    m_R = len(mirrored.boundary_rows)
    t_L = max(int(np.max(cols)) for _, cols in rows_L) + 1
    t_R = max(int(np.max(cols)) for _, cols in mirrored.boundary_rows) + 1

    # Minimum length: closure ROWS may not overlap (N >= m_L + m_R; at
    # equality the line has no interior row — the all-closure case — and
    # still assembles), and each closure's column support must fit inside
    # the domain (max(t_L, t_R) <= N; column overlap between the two
    # supports is permitted).  See module docstring.
    if N < m_L + m_R:
        raise IntervalTooShort(
            f"N = {N} is too short: the left closure occupies rows "
            f"0..{m_L - 1} and the right closure rows {N - m_R}..{N - 1}, "
            f"so closure rows overlap; the minimum assemblable length is "
            f"N = m_L + m_R = {m_L + m_R} (at which the line has no "
            f"interior row and the operator is all-closure)"
        )
    if max(t_L, t_R) > N:
        raise IntervalTooShort(
            f"N = {N} is too short for the closure column supports: the "
            f"left support occupies columns 0..{t_L - 1} and the right "
            f"support columns {N - t_R}..{N - 1}; both fit only for "
            f"N >= max(t_L, t_R) = {max(t_L, t_R)}"
        )

    D = np.zeros((N, N))

    # [dense_left]
    for (weights, cols), n_i in zip(rows_L, idx_L):
        D[int(n_i), np.asarray(cols, dtype=int)] = weights

    # [circulant interior]
    for i in range(m_L, N - m_R):
        D[i, i + interior_offsets] = interior_weights

    # [dense_right]: mirrored row at distance n_i from the right end,
    # mirrored offset c -> physical column N - 1 - c.
    for (weights, cols), n_i in zip(mirrored.boundary_rows,
                                    mirrored.row_indices):
        D[N - 1 - int(n_i), N - 1 - np.asarray(cols, dtype=int)] = weights

    if not np.all(np.isfinite(D)):
        raise ValueError(
            f"Non-finite closure coefficients at (psi_L={psi_L}, "
            f"psi_R={psi_R}) — a parameterization (denominator) failure; "
            "run the item-1.2 feasibility guard "
            "(cutcell_normal_modes) before assembling"
        )
    return D


# ---------------------------------------------------------------------------
# Plan 48 item 3.2: cheap tier — resolvent-based interval scan
# ---------------------------------------------------------------------------
#
# Bright line restated (plan 48 hard constraint): NO time integration, NO
# matrix powers S^m, NO expm, NO stepping of any update operator happens
# here.  The scan evaluates only resolvent quantities — dense eigenvalues
# of the assembled semi-discrete operator and sigma_min(sI - L) samples
# feeding the grid-sampled Kreiss constant K(L) = max Re(s)/sigma_min —
# per (psi_L, psi_R, N) triple.  Fully-discrete (RK4) reasoning is
# Phase 4; time stepping runs only in C++.


def default_interval_psi_pairs(
    psi_grid=None,
    *,
    n_pairs: int = 50,
    seed: int = 48,
) -> list[tuple[float, float]]:
    """Default (psi_L, psi_R) pairs for the interval scan (plan 48 item 3.2).

    The item-1.3 psi grid (:func:`cutcell_normal_modes.default_psi_grid`,
    clustered near 0 and matched to the C++ clamp) crossed with itself is
    400 pairs; the scan default thins it to ``n_pairs`` (~50) as

    - the four **corners** ``{lo, hi} x {lo, hi}`` (extreme-cut /
      near-uniform combinations),
    - the full **diagonal** ``(g, g)`` (symmetric cut lines), and
    - a **fixed-seed Latin sample**: random permutations ``pi`` of the
      grid indices contribute pairs ``(g[i], g[pi[i]])`` — each psi value
      appearing exactly once per permutation as a left and as a right
      value — until ``n_pairs`` unique pairs are collected.

    Deterministic for a given (psi_grid, n_pairs, seed); insertion order
    is corners, diagonal, then Latin fill.

    Parameters
    ----------
    psi_grid : sequence of float, optional
        Grid to draw from; defaults to the item-1.3 grid (20 points).
    n_pairs : int
        Target number of pairs (capped at ``len(psi_grid)**2``).
    seed : int
        Seed for the Latin-sample permutations.

    Returns
    -------
    list[tuple[float, float]]
        Unique (psi_L, psi_R) pairs, ``min(n_pairs, len(grid)**2)`` long.
    """
    if psi_grid is None:
        psi_grid = default_psi_grid()
    grid = sorted({float(p) for p in np.asarray(psi_grid, dtype=float).ravel()})
    n = len(grid)
    if n == 0:
        raise ValueError("psi_grid must contain at least one psi value")

    target = min(int(n_pairs), n * n)
    pairs: dict[tuple[float, float], None] = {}
    lo, hi = grid[0], grid[-1]
    for corner in ((lo, lo), (lo, hi), (hi, lo), (hi, hi)):
        pairs.setdefault(corner, None)
    for g in grid:
        pairs.setdefault((g, g), None)
    # Fixed-seed Latin fill: each permutation uses every grid value exactly
    # once as psi_L and once as psi_R.
    rng = np.random.default_rng(seed)
    attempts = 0
    while len(pairs) < target and attempts < 1000:
        attempts += 1
        for i, j in enumerate(rng.permutation(n)):
            pairs.setdefault((grid[i], grid[int(j)]), None)
            if len(pairs) >= target:
                break
    # Deterministic fallback (unreachable in practice): exhaust the product.
    if len(pairs) < target:
        for pair in itertools.product(grid, grid):
            pairs.setdefault(pair, None)
            if len(pairs) >= target:
                break
    return list(itertools.islice(pairs, target))


def refined_interval_s_grid(
    im_max: float,
    *,
    re_min: float = 1e-4,
    re_max: float = 2.0,
    n_re: int = 16,
    n_im: int = 121,
) -> np.ndarray:
    """Refined near-imaginary-axis s-grid for the Kreiss-constant estimate.

    Plan 48 item 3.2 requires an explicit refined grid — NOT the coarse
    8 x 12 default that ``compute_non_normality`` falls back to for large
    operators.  Real parts are log-spaced on ``[re_min, re_max]`` (dense
    against the imaginary axis, where the Kreiss ratio
    ``Re(s)/sigma_min(sI - L)`` peaks for stable non-normal operators);
    imaginary parts are uniform on ``[-im_max, im_max]``.

    Returns
    -------
    np.ndarray
        Complex grid of shape ``(n_im, n_re)``.
    """
    re_vals = np.logspace(np.log10(re_min), np.log10(re_max), n_re)
    im_vals = np.linspace(-im_max, im_max, n_im)
    Re, Im = np.meshgrid(re_vals, im_vals)
    return Re + 1j * Im


@dataclass(frozen=True)
class IntervalCase:
    """Resolvent diagnostics for one (psi_L, psi_R, N) interval operator.

    All quantities refer to the semi-discrete advection operator
    ``L = -D`` (module sign convention: stability is analyzed on
    ``u_t = -D u``); no time integration is involved.
    """

    psi_L: float
    """Left-end fractional wall position (1.0 = uniform boundary)."""

    psi_R: float
    """Right-end fractional wall position (1.0 = uniform boundary)."""

    N: int
    """Number of fluid grid nodes on the line."""

    max_re_lambda: float
    """Spectral abscissa of L = -D: max Re over the dense eigenvalues."""

    argmax_lambda: complex
    """The eigenvalue attaining ``max_re_lambda`` (an unstable interval
    eigenvalue when positive — the s* the item-3.3 round-trip criterion
    cross-validates against)."""

    kreiss_constant: float
    """Grid-sampled Kreiss constant estimate
    ``max Re(s)/sigma_min(sI - L)`` over the scan's s-grid (a coarse
    LOWER-bound-flavored estimate; grid-proximity-limited near
    right-half-plane eigenvalues, where ``max_re_lambda`` is the
    primary verdict)."""

    eigenvalues: np.ndarray = field(repr=False, default=None)
    """Full dense spectrum of L = -D (<= 40 values; kept for the
    item-3.3 cross-validation and for diagnostics)."""


@dataclass(frozen=True)
class IntervalScanResult:
    """Result of :func:`interval_stability_scan` (plan 48 item 3.2)."""

    worst_abscissa: float
    """Max over all feasible cases of ``max_re_lambda`` (the
    ``interval_worst_abscissa`` the item-3.4 report wiring consumes);
    -inf if no feasible case was evaluated."""

    worst_abscissa_case: tuple | None
    """The (psi_L, psi_R, N) triple attaining ``worst_abscissa``
    (first attainer in scan order; None if no feasible case)."""

    worst_kreiss_constant: float
    """Max over all feasible cases of the per-case Kreiss-constant
    estimate; -inf if no feasible case."""

    worst_kreiss_case: tuple | None
    """The (psi_L, psi_R, N) triple attaining ``worst_kreiss_constant``
    (None if no feasible case)."""

    per_case: dict
    """Full per-case diagnostics, keyed by ``(psi_L, psi_R, N)`` ->
    :class:`IntervalCase`, in scan order (pairs outer, N inner)."""

    infeasible_pairs: list
    """(psi_L, psi_R) pairs skipped because a structural denominator
    vanishes at either end psi (a *parameterization* failure — see
    ``cutcell_normal_modes.CutCellInfeasible`` — never conflated with
    a stability verdict)."""

    N_values: tuple
    """The interval lengths scanned."""

    s_grid: np.ndarray | None = field(repr=False, default=None)
    """The s-grid used for the Kreiss-constant estimates (None if no
    feasible case was evaluated)."""


def interval_stability_scan(
    scheme: SchemeParams,
    alpha: dict,
    *,
    psi_pairs=None,
    N_values=None,
    s_grid: np.ndarray | None = None,
    feasibility_tol: float = FEASIBILITY_TOL,
) -> IntervalScanResult:
    """Cheap-tier resolvent scan of two-ended cut-cell intervals (item 3.2).

    For every feasible (psi_L, psi_R) pair and every interval length N,
    assembles the interval operator (item 3.1) and computes, on
    ``L = -D`` (the semi-discrete advection convention):

    - ``max Re lambda`` by dense eigendecomposition (matrices are
      <= 40 x 40 — microseconds each), and
    - ``kreiss_constant_estimate(L, s_grid)`` on a refined
      near-imaginary-axis grid (:func:`refined_interval_s_grid`; NOT the
      coarse 8 x 12 default of ``compute_non_normality``).

    **Bright line (plan 48 hard constraint): no matrix powers, no
    stepping; resolvent quantities only.**

    Parameters
    ----------
    scheme : SchemeParams
        Cut-cell scheme (e.g. ``temo.E2_1``, ``temo.E4_1``).
    alpha : dict
        Mapping from alpha symbol (or name) to numeric value; must cover
        all of the scheme's free parameters.
    psi_pairs : sequence of (float, float), optional
        (psi_L, psi_R) pairs to scan; defaults to
        :func:`default_interval_psi_pairs` — the item-1.3 grid crossed
        with itself, thinned to ~50 pairs (corners + diagonal +
        fixed-seed Latin sample).
    N_values : sequence of int, optional
        Interval lengths; defaults to
        ``(m_min, m_min + 2, m_min + 5, m_min + 10, 20, 40)`` with
        ``m_min = m_L + m_R = 2 (R - 1)`` the minimum assemblable length
        from item 3.1 (all-closure at m_min — exactly the dangerous
        short-cut-line case).  Duplicates are dropped.
    s_grid : np.ndarray, optional
        Explicit s-grid for the Kreiss-constant estimates.  If None, a
        single shared :func:`refined_interval_s_grid` is built with
        ``im_max`` covering the spectral extent of every scanned case
        (``max(1, 1.1 max|Im lambda| + 0.5)``), so per-case estimates
        are mutually comparable.
    feasibility_tol : float
        |denominator| threshold for the per-end item-1.2 feasibility
        guard; pairs with a vanishing structural denominator at either
        end go to ``infeasible_pairs``, never into the margins.

    Returns
    -------
    IntervalScanResult

    Raises
    ------
    IntervalTooShort
        If a requested N is below the scheme's minimum assemblable
        length (propagated from :func:`build_interval_operator`).
    """
    evaluator = make_cutcell_evaluator(scheme)
    m = evaluator.dims.R - 1
    m_min = 2 * m

    if N_values is None:
        N_list = sorted({m_min, m_min + 2, m_min + 5, m_min + 10, 20, 40})
    else:
        N_list = list(dict.fromkeys(int(N) for N in N_values))
        if not N_list:
            raise ValueError("N_values must contain at least one length")

    if psi_pairs is None:
        psi_pairs = default_interval_psi_pairs()
    pair_list = list(
        dict.fromkeys((float(pl), float(pr)) for pl, pr in psi_pairs)
    )
    if not pair_list:
        raise ValueError("psi_pairs must contain at least one pair")

    # Item-1.2 feasibility guard, cached per end psi (a pair is infeasible
    # iff either end's parameterization fails).
    psi_ok: dict[float, bool] = {}

    def _feasible(psi: float) -> bool:
        if psi not in psi_ok:
            psi_ok[psi] = not _vanishing_denominators(
                evaluator, psi, alpha, feasibility_tol
            )
        return psi_ok[psi]

    infeasible_pairs: list = []
    evaluated: list[tuple[tuple, np.ndarray, np.ndarray]] = []
    for pair in pair_list:
        psi_L, psi_R = pair
        if not (_feasible(psi_L) and _feasible(psi_R)):
            infeasible_pairs.append(pair)
            continue
        for N in N_list:
            L = -build_interval_operator(scheme, alpha, psi_L, psi_R, N)
            eigs = np.linalg.eigvals(L)
            evaluated.append(((psi_L, psi_R, N), L, eigs))

    grid_used: np.ndarray | None = None
    if evaluated:
        if s_grid is not None:
            grid_used = np.asarray(s_grid, dtype=complex)
        else:
            im_extent = max(
                float(np.max(np.abs(eigs.imag))) for _, _, eigs in evaluated
            )
            grid_used = refined_interval_s_grid(
                max(1.0, 1.1 * im_extent + 0.5)
            )

    per_case: dict = {}
    worst_abscissa = float("-inf")
    worst_abscissa_case: tuple | None = None
    worst_kreiss = float("-inf")
    worst_kreiss_case: tuple | None = None
    for key, L, eigs in evaluated:
        i_max = int(np.argmax(eigs.real))
        max_re = float(eigs.real[i_max])
        K = float(kreiss_constant_estimate(L, grid_used))
        per_case[key] = IntervalCase(
            psi_L=key[0],
            psi_R=key[1],
            N=key[2],
            max_re_lambda=max_re,
            argmax_lambda=complex(eigs[i_max]),
            kreiss_constant=K,
            eigenvalues=eigs,
        )
        if max_re > worst_abscissa:
            worst_abscissa = max_re
            worst_abscissa_case = key
        if K > worst_kreiss:
            worst_kreiss = K
            worst_kreiss_case = key

    return IntervalScanResult(
        worst_abscissa=worst_abscissa,
        worst_abscissa_case=worst_abscissa_case,
        worst_kreiss_constant=worst_kreiss,
        worst_kreiss_case=worst_kreiss_case,
        per_case=per_case,
        infeasible_pairs=infeasible_pairs,
        N_values=tuple(N_list),
        s_grid=grid_used,
    )
