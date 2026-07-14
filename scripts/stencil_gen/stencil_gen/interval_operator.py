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
48 hard constraint) — this module only assembles; the resolvent-based
scan is item 3.2.

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

import numpy as np

from stencil_gen.cutcell_eval import make_cutcell_evaluator
from stencil_gen.reflection import mirror_closure
from stencil_gen.temo import SchemeParams

__all__ = ["IntervalTooShort", "build_interval_operator"]


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
