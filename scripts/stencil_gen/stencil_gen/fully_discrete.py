"""RK4 stability-region boundary containment — resolvent only (plan 48, 4.1).

Certifies fully-discrete (method-of-lines + classical RK4) stability of a
semi-discrete operator L by RESOLVENT quantities alone.  **Bright line
(plan 48 hard constraint): no time integration happens in Python** — no
matrix powers ``S^m``, no ``expm``, no stepping of any update operator.
RK4 itself runs only in C++ (plan 48 Phase 4 L8 items); here the RK4
method appears exclusively through its scalar amplification polynomial.

Math
----
Classical RK4 amplification::

    R(zeta) = 1 + zeta + zeta^2/2 + zeta^3/6 + zeta^4/24

stability region ``S = { zeta : |R(zeta)| <= 1 }`` (a CLOSED set — an
eigenvalue exactly on the boundary, e.g. the consistency/constant mode of
a periodic or conservative operator at zeta = 0, counts as inside but
correctly produces a zero containment margin).  The boundary ``dS`` is
computed once by marching the level set ``|R(zeta)| = 1`` from
``zeta = 0``: solve ``R(zeta) = e^{i theta}`` by Newton continuation
(previous point as the next initial guess; a companion-matrix root solve
is the fallback if Newton strays), and cache the resulting polyline at
module level (:func:`rk4_stability_boundary` is ``lru_cache``-d).  Pure
numpy — no operator is ever stepped.  One phase subtlety the naive
``theta in [0, 2 pi)`` recipe misses: by the argument principle
``arg R`` winds FOUR times around dS (R has 4 zeros, all trivially
inside S since ``|R| = 0`` there, and numerically all inside the single
traced component), so the continuation must run ``theta in [0, 8 pi)``
to close the loop — a quarter-traced polyline would silently
overestimate the margin.  The march asserts closure back to the origin.
The traced curve is a single closed conjugate-symmetric loop through the
origin, crossing the negative real axis at the RK4 real stability limit
x* ~ -2.7853 (the negative real root of ``R(x) = +1``) and passing
through ``+/- 2 sqrt(2) i`` (where ``|R(i y)| = 1``).

For an operator L (uniform or interval) and ``dt = CFL * h``::

    containment_margin(L, dt) = min over z in dS of sigma_min(z I - dt L)

and, alongside it, whether all eigenvalues of ``dt L`` lie in S — both
are reported (:class:`ContainmentResult`).

Interpretation
--------------
Eigenvalues-in-S is NECESSARY for fully-discrete stability.  The joint
criterion is stronger than "spectrum within an eps-neighborhood of S":
**eigs-inside + containment_margin >= eps certifies
``Lambda_eps(dt L) subset S`` outright**, because every (bounded)
connected component of the eps-pseudospectrum
``Lambda_eps = { z : sigma_min(z I - dt L) < eps }`` contains an
eigenvalue, so no component can lie wholly outside S, and margin >= eps
on dS forbids any component from crossing the boundary.  By
Reddy & Trefethen (1992, pseudospectra vs. the RK stability region) this
pseudospectral containment is what controls fully-discrete
power-boundedness for NON-NORMAL L — the spectrum alone certifies
nothing there.  Together with the semi-discrete GKS margin (plan 48
Phases 0-1) this instantiates the Kreiss & Wu (1993) hypothesis chain:
a resolvent-stable semi-discrete approximation combined with a locally
stable one-step method yields a stable fully-discrete scheme.  Note
explicitly: Kreiss & Wu covers the HALF-LINE problem — marginal
fixed-Delta-j two-boundary cases (short cut lines) are NOT covered by
that theorem; those remain the province of the interval scan /
round-trip criterion (plan 48 items 3.2/3.3) plus the C++ L8 rung.

Discretization honesty
----------------------
``sigma_min(z I - dt L)`` is 1-Lipschitz in z, and the grid minimum is a
minimum over a SUBSET of dS, so it can only OVERestimate the true
continuous minimum — by at most half the largest gap between adjacent
boundary samples (every boundary point lies within half a gap of some
sample).  The certified value

    margin_certified = margin_grid - max_arc_spacing / 2

is therefore reported alongside the raw grid minimum; only
``margin_certified >= eps`` supports the pseudospectral containment
claim above.  Default ``n_boundary = 1024``: at 256 points the spacing
correction is ~0.05 — material for small margins.

Sweep entry point
-----------------
:func:`rk4_containment_over_psi` reuses the item-3.1 interval assembler
at a fixed representative N (default 40) over a psi grid — cut closure
at the left (inflow under the ``u_t = -D u`` convention) end, uniform
domain boundary (psi = 1) at the right — PLUS an evaluation at the worst
(psi_L, psi_R, N) triple(s) reported by the item-3.2 interval scan: a
fixed N can miss N-specific resonances, and the psi sweep's fixed
right-uniform pairing deliberately mirrors the item-1.2/1.3 half-line
sweeps (the outflow-cut configurations are covered by the scan triples,
which include diagonal (psi, psi) pairs by default).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np

from stencil_gen.cutcell_eval import make_cutcell_evaluator
from stencil_gen.cutcell_normal_modes import (
    FEASIBILITY_TOL,
    _vanishing_denominators,
    default_psi_grid,
)
from stencil_gen.interval_operator import (
    IntervalScanResult,
    build_interval_operator,
    interval_stability_scan,
)
from stencil_gen.temo import SchemeParams

__all__ = [
    "ContainmentResult",
    "Rk4ContainmentSweep",
    "rk4_amplification",
    "rk4_containment",
    "rk4_containment_over_psi",
    "rk4_stability_boundary",
]


# R(zeta) and R'(zeta) coefficients, highest degree first (np.polyval /
# np.roots layout).
_R_COEFFS = np.array([1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0])
_RPRIME_COEFFS = np.array([1.0 / 6.0, 0.5, 1.0, 1.0])

_NEWTON_TOL = 1e-13
_NEWTON_MAX_ITER = 100
# Winding number of arg R around dS (argument principle: R has 4 zeros,
# each with |R| = 0 < 1 hence inside S; the boundary march runs
# theta in [0, 2 pi * _WINDING) and asserts closure).
_WINDING = 4
# Continuation-jump guard: min |R'| on the curve is ~0.98, so the true
# step along the curve is ~ (2 pi _WINDING / n_boundary) / |R'| — well
# below this for n_boundary >= 128; a larger jump means Newton hopped to
# another sheet of R(zeta) = e^{i theta} and the companion-matrix
# fallback (nearest root to the previous point) is used instead.
_MAX_CONTINUATION_JUMP = 0.5
_MIN_N_BOUNDARY = 128

_DEFAULT_EIG_TOL = 1e-10
"""Closed-region tolerance for the eigenvalue test: zeta counts as inside
S iff |R(zeta)| <= 1 + eig_tol (S is closed; boundary eigenvalues such as
the zeta = 0 consistency mode of periodic/conservative operators are
inside — and yield a zero margin, the honest verdict).  Caveat: a
DEFECTIVE eigenvalue exactly on dS (e.g. the Jordan-blocked zero of the
E2_1 alpha = 0 row-degenerate closure) is resolved numerically only to
~eps^(1/k) (k = Jordan block size, ~1e-8 for k = 2), so its computed
``eigs_inside`` can flap under this default; in such marginal cases the
MARGIN (which is then ~0) is the load-bearing certificate — pass a
looser ``eig_tol`` when the boundary mode itself is the object of
study."""


def rk4_amplification(zeta):
    """Classical RK4 amplification factor R(zeta).

    ``R(zeta) = 1 + zeta + zeta^2/2 + zeta^3/6 + zeta^4/24`` — the degree-4
    Taylor polynomial of ``e^zeta``.  Accepts scalars or arrays (numpy
    broadcasting).
    """
    return np.polyval(_R_COEFFS, zeta)


def _solve_level_point(z_prev: complex, target: complex) -> complex:
    """Solve ``R(zeta) = target`` on the branch through *z_prev*.

    Newton from *z_prev* (the continuation guess); if Newton fails to
    converge, hits a critical point of R, or jumps farther than the
    continuation guard allows (a sheet hop), fall back to the
    companion-matrix roots of the quartic and take the root nearest
    *z_prev*.
    """
    z = complex(z_prev)
    for _ in range(_NEWTON_MAX_ITER):
        resid = np.polyval(_R_COEFFS, z) - target
        if abs(resid) < _NEWTON_TOL:
            if abs(z - z_prev) <= _MAX_CONTINUATION_JUMP:
                return z
            break  # converged, but onto another sheet
        deriv = np.polyval(_RPRIME_COEFFS, z)
        if deriv == 0:
            break
        z = z - resid / deriv
    coeffs = _R_COEFFS.astype(complex).copy()
    coeffs[-1] -= target
    roots = np.roots(coeffs)
    return complex(roots[np.argmin(np.abs(roots - z_prev))])


@functools.lru_cache(maxsize=None)
def rk4_stability_boundary(n_boundary: int = 1024) -> np.ndarray:
    """The RK4 stability-region boundary dS as a cached complex polyline.

    Marches the level set ``|R(zeta)| = 1`` from ``zeta = 0`` by solving
    ``R(zeta) = e^{i theta_j}``, ``theta_j = 2 pi _WINDING j / n_boundary``,
    j = 0..n_boundary-1, with Newton continuation (see module docstring;
    ``arg R`` winds ``_WINDING = 4`` times around dS, so a single 2 pi
    sweep would trace only a quarter of the loop).  Pure numpy; computed
    once per *n_boundary* and cached at module level (``lru_cache``), the
    plan-48 item-4.1 requirement.  The first point is exactly 0; closure
    of the loop back to the origin is asserted.

    Returns
    -------
    np.ndarray
        Complex array of shape ``(n_boundary,)`` sampling the full closed
        boundary once, marked read-only (it is shared by every caller
        through the cache).

    Raises
    ------
    ValueError
        If ``n_boundary < 128`` (the continuation-jump guard assumes
        steps along the curve stay small).
    RuntimeError
        If the march fails to close back to the origin (branch-tracking
        failure — should be unreachable for RK4).
    """
    n_boundary = int(n_boundary)
    if n_boundary < _MIN_N_BOUNDARY:
        raise ValueError(
            f"n_boundary must be >= {_MIN_N_BOUNDARY} for a reliable "
            f"continuation march, got {n_boundary}"
        )
    thetas = 2.0 * np.pi * _WINDING * np.arange(n_boundary) / n_boundary
    pts = np.empty(n_boundary, dtype=complex)
    z = 0.0 + 0.0j  # R(0) = 1 = e^{i 0}: the march starts exactly at 0.
    for j, theta in enumerate(thetas):
        z = _solve_level_point(z, np.exp(1j * theta))
        pts[j] = z
    # Closure check: one more continuation step (theta = 2 pi _WINDING,
    # target 1) must land back on the starting point.
    z_close = _solve_level_point(z, 1.0 + 0.0j)
    if abs(z_close) > 2.0 * np.pi * _WINDING / n_boundary + 1e-9:
        raise RuntimeError(
            f"RK4 boundary march failed to close: final point {z_close} "
            f"(n_boundary = {n_boundary})"
        )
    pts.flags.writeable = False
    return pts


def _max_arc_spacing(boundary: np.ndarray) -> float:
    """Largest gap between adjacent samples of the CLOSED polyline
    (includes the wrap-around segment from the last point back to the
    first)."""
    gaps = np.abs(np.diff(boundary))
    wrap = abs(boundary[0] - boundary[-1])
    return float(max(gaps.max(), wrap))


@dataclass(frozen=True)
class ContainmentResult:
    """Result of :func:`rk4_containment` (plan 48 item 4.1).

    Joint criterion (module docstring): ``eigs_inside`` is necessary;
    ``eigs_inside and margin_certified >= eps`` certifies
    ``Lambda_eps(dt L) subset S``.
    """

    margin: float
    """Grid containment margin: min over the sampled boundary polyline of
    ``sigma_min(z I - dt L)``."""

    margin_certified: float
    """``margin - max_arc_spacing / 2`` — the value certified for the
    CONTINUOUS boundary via the 1-Lipschitz bound (may be negative when
    the boundary sampling is too coarse for the margin in question)."""

    eigs_inside: bool
    """True iff every eigenvalue zeta of ``dt L`` satisfies
    ``|R(zeta)| <= 1 + eig_tol`` (S is closed — see module docstring)."""

    argmin_z: complex
    """The sampled boundary point attaining ``margin``."""

    max_arc_spacing: float
    """Largest gap between adjacent boundary samples (closed polyline)."""

    n_boundary: int
    """Number of boundary samples used."""

    eigenvalues: np.ndarray = field(repr=False, default=None)
    """Dense spectrum of ``dt L`` (kept for diagnostics)."""


def rk4_containment(
    L: np.ndarray,
    dt: float,
    *,
    n_boundary: int = 1024,
    eig_tol: float = _DEFAULT_EIG_TOL,
) -> ContainmentResult:
    """RK4 stability-region containment margin of ``dt L`` (item 4.1).

    Computes ``min over z in dS of sigma_min(z I - dt L)`` on the cached
    boundary polyline, checks that all eigenvalues of ``dt L`` lie in the
    (closed) stability region S, and reports both together with the
    Lipschitz-certified margin (module docstring).  Resolvent quantities
    only — one dense eigendecomposition plus ``n_boundary`` singular-value
    evaluations; **no time integration, no matrix powers**.

    Parameters
    ----------
    L : np.ndarray
        Square semi-discrete operator (uniform or interval), already in
        the ``u_t = L u`` sign convention (e.g. ``L = -D`` for advection).
    dt : float
        Time step (> 0); for the CFL parameterization ``dt = CFL * h``.
    n_boundary : int
        Boundary samples (default 1024; at 256 the spacing correction
        ~0.05 is material for small margins).
    eig_tol : float
        Closed-region tolerance for the eigenvalue test.

    Returns
    -------
    ContainmentResult
    """
    L = np.asarray(L)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be a square matrix, got shape {L.shape}")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be a positive finite float, got {dt}")

    boundary = rk4_stability_boundary(n_boundary)
    M = dt * L
    eye = np.eye(M.shape[0], dtype=complex)

    sigmas = np.array(
        [np.linalg.svd(z * eye - M, compute_uv=False)[-1] for z in boundary]
    )
    i_min = int(np.argmin(sigmas))
    margin = float(sigmas[i_min])
    spacing = _max_arc_spacing(boundary)

    eigenvalues = np.linalg.eigvals(M)
    eigs_inside = bool(
        np.all(np.abs(rk4_amplification(eigenvalues)) <= 1.0 + eig_tol)
    )

    return ContainmentResult(
        margin=margin,
        margin_certified=margin - spacing / 2.0,
        eigs_inside=eigs_inside,
        argmin_z=complex(boundary[i_min]),
        max_arc_spacing=spacing,
        n_boundary=int(n_boundary),
        eigenvalues=eigenvalues,
    )


# ---------------------------------------------------------------------------
# psi-swept containment for cut-cell interval operators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rk4ContainmentSweep:
    """Result of :func:`rk4_containment_over_psi` (plan 48 item 4.1).

    Aggregates the fixed-N psi-sweep cases AND the worst-scan-triple
    evaluations into single worst-cased headline numbers (the plan-49
    optimizer consumables); the two families stay separately inspectable
    in ``per_psi`` / ``scan_cases``.
    """

    worst_margin: float
    """Min over all evaluated cases (psi sweep and scan triples) of the
    grid ``margin``; inf if nothing was evaluated."""

    worst_margin_certified: float
    """Min over all evaluated cases of ``margin_certified``; inf if
    nothing was evaluated."""

    worst_case: object
    """The case attaining ``worst_margin_certified``: a float psi (sweep
    family, pair ``(psi, psi_R)`` at the representative N) or a
    ``(psi_L, psi_R, N)`` triple (scan family).  None if nothing was
    evaluated."""

    any_eigs_outside: bool
    """True iff any evaluated case (either family) has an eigenvalue of
    ``dt L`` outside the closed stability region S."""

    infeasible_psi: list
    """Sweep psi values skipped by the item-1.2 feasibility guard (a
    *parameterization* failure — never conflated with a verdict)."""

    per_psi: dict
    """Sweep family: ``float(psi) -> ContainmentResult`` for the interval
    ``(psi_L = psi, psi_R, N)`` at the representative N."""

    scan_cases: dict
    """Scan family: ``(psi_L, psi_R, N) -> ContainmentResult`` at the
    worst triple(s) reported by the item-3.2 scan (worst spectral
    abscissa, and worst Kreiss constant when it differs)."""

    N: int
    """Representative interval length of the psi sweep."""

    psi_R: float
    """Fixed right-end psi of the sweep family (1.0 = uniform domain
    boundary; the outflow-cut configurations are covered by the scan
    triples)."""

    cfl: float
    """CFL number."""

    h: float
    """Grid spacing (``dt = cfl * h``; since the assembler works at unit
    spacing, ``dt L = -cfl * D`` and h cancels — kept for interface
    fidelity to the physical parameters)."""

    dt: float
    """Time step ``cfl * h``."""


def rk4_containment_over_psi(
    scheme: SchemeParams,
    alpha: dict,
    cfl: float,
    h: float,
    psi_grid=None,
    *,
    N: int = 40,
    psi_R: float = 1.0,
    n_boundary: int = 1024,
    eig_tol: float = _DEFAULT_EIG_TOL,
    scan: IntervalScanResult | None = None,
    scan_kwargs: dict | None = None,
    feasibility_tol: float = FEASIBILITY_TOL,
) -> Rk4ContainmentSweep:
    """Worst-case RK4 containment over psi for a cut-cell scheme (item 4.1).

    Two case families, both through the item-3.1 interval assembler
    (semi-discrete operator ``L = -D / h``, time step ``dt = cfl * h``;
    the unit-spacing D makes ``dt L = -cfl * D`` mathematically — h
    cancels up to rounding):

    1. **psi sweep** at the fixed representative *N* (default 40): for
       each feasible psi in *psi_grid* (default: the item-1.3 grid), the
       interval ``(psi_L = psi, psi_R)`` — cut closure at the left
       (inflow) end, uniform domain boundary (``psi_R = 1.0``) at the
       right, mirroring the item-1.2/1.3 half-line sweeps.
    2. **worst scan triple(s)**: an evaluation at the worst
       (psi_L, psi_R, N) triple reported by the item-3.2 interval scan —
       a fixed N can miss N-specific resonances.  Both of the scan's
       reported worst cases (spectral abscissa and Kreiss constant) are
       evaluated when they differ.  If *scan* is None, the scan is run
       here (``interval_stability_scan(scheme, alpha, **scan_kwargs)`` —
       the full default scan is expensive; pass a precomputed *scan* or
       restricted *scan_kwargs* in tight loops).

    Infeasible psi points (item-1.2 guard: a structural denominator
    vanishes) are partitioned into ``infeasible_psi`` and never enter the
    margins.  **No time integration anywhere** — each case costs one
    eigendecomposition plus ``n_boundary`` sigma_min evaluations.

    Parameters
    ----------
    scheme, alpha
        Cut-cell scheme and its free-parameter values (item-1.1 evaluator
        conventions).
    cfl : float
        CFL number (> 0); ``dt = cfl * h``.
    h : float
        Grid spacing (> 0); cancels in ``dt L`` (see above).
    psi_grid : sequence of float, optional
        Sweep psi values; defaults to
        ``cutcell_normal_modes.default_psi_grid()``.
    N : int
        Representative interval length for the sweep family.
    psi_R : float
        Fixed right-end psi of the sweep family (default 1.0 — uniform
        domain boundary).
    n_boundary, eig_tol
        Forwarded to :func:`rk4_containment`.
    scan : IntervalScanResult, optional
        Precomputed item-3.2 scan whose worst triples are evaluated;
        mutually exclusive with *scan_kwargs*.
    scan_kwargs : dict, optional
        Arguments for the internally run ``interval_stability_scan``
        when *scan* is not supplied.
    feasibility_tol : float
        |denominator| threshold of the per-psi feasibility guard.

    Returns
    -------
    Rk4ContainmentSweep
    """
    cfl = float(cfl)
    if not np.isfinite(cfl) or cfl <= 0.0:
        raise ValueError(f"cfl must be a positive finite float, got {cfl}")
    h = float(h)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError(f"h must be a positive finite float, got {h}")
    psi_R = float(psi_R)
    if not 0.0 < psi_R <= 1.0:
        raise ValueError(f"psi_R must be in (0, 1], got {psi_R}")
    if scan is not None and scan_kwargs is not None:
        raise TypeError("pass either scan or scan_kwargs, not both")

    if psi_grid is None:
        psi_grid = default_psi_grid()
    psi_values = [float(p) for p in psi_grid]
    if not psi_values:
        raise ValueError("psi_grid must contain at least one psi value")
    for psi in psi_values:
        if not 0.0 < psi <= 1.0:
            raise ValueError(f"psi must be in (0, 1], got {psi}")

    dt = cfl * h
    evaluator = make_cutcell_evaluator(scheme)

    def _contain(psi_L: float, psi_r: float, n: int) -> ContainmentResult:
        D = build_interval_operator(scheme, alpha, psi_L, psi_r, n)
        return rk4_containment(
            -D / h, dt, n_boundary=n_boundary, eig_tol=eig_tol
        )

    # Family 1: the psi sweep at the representative N.
    per_psi: dict = {}
    infeasible_psi: list = []
    psi_R_feasible = not _vanishing_denominators(
        evaluator, psi_R, alpha, feasibility_tol
    )
    for psi in psi_values:
        if not psi_R_feasible or _vanishing_denominators(
            evaluator, psi, alpha, feasibility_tol
        ):
            infeasible_psi.append(psi)
            continue
        per_psi[psi] = _contain(psi, psi_R, N)

    # Family 2: the worst (psi_L, psi_R, N) triple(s) from the 3.2 scan.
    if scan is None:
        scan = interval_stability_scan(scheme, alpha, **(scan_kwargs or {}))
    triples: list[tuple] = []
    for triple in (scan.worst_abscissa_case, scan.worst_kreiss_case):
        if triple is not None and triple not in triples:
            triples.append(triple)
    scan_cases: dict = {
        triple: _contain(triple[0], triple[1], int(triple[2]))
        for triple in triples
    }

    # Headline aggregation over BOTH families.
    worst_margin = float("inf")
    worst_certified = float("inf")
    worst_case: object = None
    any_outside = False
    for case_key, result in [*per_psi.items(), *scan_cases.items()]:
        worst_margin = min(worst_margin, result.margin)
        if result.margin_certified < worst_certified:
            worst_certified = result.margin_certified
            worst_case = case_key
        if not result.eigs_inside:
            any_outside = True

    return Rk4ContainmentSweep(
        worst_margin=worst_margin,
        worst_margin_certified=worst_certified,
        worst_case=worst_case,
        any_eigs_outside=any_outside,
        infeasible_psi=infeasible_psi,
        per_psi=per_psi,
        scan_cases=scan_cases,
        N=int(N),
        psi_R=psi_R,
        cfl=cfl,
        h=h,
        dt=dt,
    )
