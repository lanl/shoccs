"""Reflection-coefficient analysis for semi-discrete boundary closures.

Plan 48 Phase 2 (Trefethen 1984, "Instability of difference models for
hyperbolic initial boundary value problems", and Trefethen 1985, Math.
Comp. 45, for the two-boundary round-trip in Phase 3). This module starts
with the mode partition (item 2.1): splitting the kappa roots of the
interior characteristic polynomial at a Laplace frequency s into
rightgoing and leftgoing families, the basis on which per-boundary
reflection matrices R(s) are built.

Sign convention: this module analyzes u_t = -sum_k a_k u_{n+k} (the
``gks_kreiss.kappa_roots`` docstring convention); under the "raw operator
weights" reading all w-signs flip.

Mode partition
--------------
For the left-half-line problem, the mode ansatz u_n(t) = e^{st} kappa^n
turns the interior stencil into the characteristic polynomial
Q(kappa; s) = 0 (see ``kappa_roots``). At a given s the roots split into:

- **rightgoing** (|kappa| < 1 - tol): decaying with increasing n — the
  admissible family for Re(s) > 0 on the left half-line. At the left
  boundary these are the modes *radiated* into the interior (Trefethen's
  outgoing modes: group velocity pointing away from the boundary).
- **leftgoing** (|kappa| > 1 + tol): growing with increasing n — for the
  reflection problem these are the modes *incident* on the left boundary
  from the interior.
- **borderline** (unit-modulus within tol): resolved by the shared
  perturbation test ``gks_kreiss.perturbed_root_directions`` (Trefethen
  1983 p. 207): a root that moves inward under Re(s) += delta is the
  half-plane limit of the rightgoing family; outward, of the leftgoing
  family. Coalescing borderline roots (sonic / branch points, e.g. the E2
  interior at s = +/- i) raise DefectiveKappaError — they must route
  through the defective path (plan 48 items 0.1 / 2.2), never report as a
  clean partition.

For a centered 2p+1-point interior the characteristic polynomial has
degree 2p. For a real antisymmetric (first-derivative-consistent,
anti-Hermitian-symbol) centered interior and Re(s) > 0, exactly p roots
lie strictly inside the unit circle and p strictly outside: a unimodular
kappa = e^{i theta} would force s = -sum_k a_k e^{ik theta}, which is
purely imaginary, contradicting Re(s) > 0; the count then follows by
continuity from |s| -> infinity, where p roots go to 0 and p to infinity.
``mode_split`` enforces this p/p count for every centered interior at
Re(s) > 0 (plan 48 item 2.1) and raises ValueError otherwise — an
engineered non-derivative centered stencil can genuinely violate it, and
the error surfaces that rather than returning a defective basis.

Imaginary-axis evaluation: evaluate at s = i*omega + eps_imag —
consistent with ``make_s_grid``'s strip (eps_imag = 1e-6) — so borderline
roots are resolved by the half-plane limit itself: the O(eps_imag)
modulus displacement exceeds the default ``tol`` and the strict partition
applies without invoking the perturbation test.

Reflection matrix (plan 48 item 2.2)
------------------------------------
With m closure rows, nodal unknowns phi_0 .. phi_{m-p-1}, rightgoing
amplitudes alpha in C^p, and a unit-amplitude incident leftgoing mode
mu_k (|mu_k| > 1), the reflected field is

    phi_n = mu_k^n + sum_l alpha_l kappa_l^n   (n >= m - p),
    phi_q free                                 (q < m - p).

Substituting into the m closure equations gives M(s) x = -b_k(s) with
M(s) the augmented Kreiss matrix of item 0.2 (``kreiss_matrix_general``)
and the incident load vector

    b_k[i] = s mu_k^{n_i} [n_i >= m-p]
             + sum_{j >= m-p, j in support(i)} w_{i,j} mu_k^j

— i.e. b_k = c(mu_k) where c(z) is exactly M's root-column function.
Solving for x, the k-th column of the reflection matrix is the
alpha-block of x: R(s) is p x n_leftgoing (p x p for a centered interior
at Re(s) > 0; p x 0 for one-sided interiors like first-order upwind,
which have nothing to reflect). ``max_reflection`` reports the margin

    max_reflection = max over the imaginary-axis s-grid of ||R(s)||_2.

Why R beats sigma_min as a margin: R is invariant under scaling an
assembled *equation* — multiplying row i of [M(s) | b_k(s)] jointly by a
constant leaves the solution x (hence R) unchanged — while sigma_min(M)
changes under the same joint row scaling. (Caution: scaling only the
stored BoundaryRow *weights* does NOT preserve R, because the temporal
coefficient stays 1 — the spatial parts of M and b scale but the s-parts
do not.) And ||R(s)|| -> infinity exactly at the det M(s0) = 0 points the
GKS check flags (singularity equivalence).

Structural near-origin poles: the benign determinant zeros of item
0.4/0.5d (the kappa = 1 consistency zero of conservative closures; the
tail-constant double zero of wall-eliminated Dirichlet cut blocks) are
det M(s) zeros, so R(s) has a *pole* at s = 0 there — a parameterization
artifact of the mode basis, not a GKS instability. The default
``max_reflection`` grid therefore excludes |omega| < min_omega = 1e-3,
mirroring kreiss_stability_check's min_s_magnitude mask.

Sonic points: where interior roots coalesce ON the unit circle (E2:
s = +/- i) the mode basis is defective and the partition undefined —
``reflection_matrix`` raises DefectiveKappaError (via ``mode_split``'s
borderline handling, plus an explicit within-family coalescence guard)
and ``max_reflection`` records the point in ``defective_points`` with
``finite=False`` rather than reporting a clean value (plan 48 item 0.1
routing). Note (measured; documented for the 2.2 reviewer): at the
eps_imag offset next to a sonic point the *amplitude-normalized* scalar
R stays O(1) rather than spiking like eps^{-1/2} — b_k = c(mu_k) is M's
own root-column function, so as mu -> kappa the solve degenerates to
x -> -e_kappa (R -> -1) by continuity. The eps^{-1/2} blowup mechanism
requires two *rightgoing* roots to coalesce (a genuinely defective M),
which is exactly what the DefectiveKappaError guard routes away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stencil_gen.gks_kreiss import (
    DefectiveKappaError,
    kappa_roots,
    perturbed_root_directions,
)

__all__ = [
    "DEFAULT_UNIT_TOL",
    "ModeSplit",
    "mode_split",
    "ReflectionResult",
    "reflection_system",
    "reflection_matrix",
    "max_reflection",
]


DEFAULT_UNIT_TOL = 1e-7
"""Default half-width of the borderline modulus band [1 - tol, 1 + tol].

Chosen below the O(eps_imag) = O(1e-6) modulus displacement of roots
evaluated on the ``make_s_grid`` strip s = i*omega + eps_imag (so strip
points partition strictly, "resolved by the half-plane limit"), but above
numpy.roots' ~sqrt(machine eps) ~ 1e-8 rounding error for a *double*
root, so coalescing sonic roots land in the borderline band and raise
DefectiveKappaError instead of being silently mis-partitioned.
"""


def _sort_roots(roots: np.ndarray) -> np.ndarray:
    """Deterministic ordering: lexicographic by (real, imag)."""
    roots = np.asarray(roots, dtype=complex)
    if len(roots) == 0:
        return roots
    order = np.lexsort((roots.imag, roots.real))
    return roots[order]


@dataclass(frozen=True)
class ModeSplit:
    """Partition of the interior kappa roots at a Laplace frequency s.

    Sign convention: u_t = -sum_k a_k u_{n+k} (see module docstring).

    The three buckets are disjoint and together contain every root of the
    characteristic polynomial: ``rightgoing`` and ``leftgoing`` hold the
    strictly partitioned roots; ``borderline`` holds unit-modulus roots
    (within ``tol``) together with the family the perturbation test
    resolved each one into. ``all_rightgoing`` / ``all_leftgoing`` append
    the resolved borderline roots to the strict buckets.
    """

    rightgoing: np.ndarray
    """Roots with |kappa| < 1 - tol, sorted lexicographically by
    (real, imag). Decaying with n: the admissible family for Re(s) > 0,
    radiated from the left boundary into the interior."""

    leftgoing: np.ndarray
    """Roots with |kappa| > 1 + tol, sorted lexicographically by
    (real, imag). Growing with n: incident on the left boundary."""

    borderline: tuple[tuple[complex, str], ...]
    """Unit-modulus roots (|kappa| within tol of 1) as (root, family)
    pairs, family in {'rightgoing', 'leftgoing'} per the shared
    perturbation test (gks_kreiss.perturbed_root_directions)."""

    s: complex
    """The Laplace frequency at which the partition was computed."""

    tol: float
    """Half-width of the borderline modulus band used."""

    @property
    def all_rightgoing(self) -> np.ndarray:
        """Strict rightgoing roots plus borderline roots resolved inward."""
        extra = [k for k, family in self.borderline if family == "rightgoing"]
        if not extra:
            return self.rightgoing
        return np.concatenate(
            [self.rightgoing, np.asarray(extra, dtype=complex)]
        )

    @property
    def all_leftgoing(self) -> np.ndarray:
        """Strict leftgoing roots plus borderline roots resolved outward."""
        extra = [k for k, family in self.borderline if family == "leftgoing"]
        if not extra:
            return self.leftgoing
        return np.concatenate(
            [self.leftgoing, np.asarray(extra, dtype=complex)]
        )

    @property
    def n_rightgoing(self) -> int:
        """Rightgoing count including resolved borderline roots."""
        return len(self.all_rightgoing)

    @property
    def n_leftgoing(self) -> int:
        """Leftgoing count including resolved borderline roots."""
        return len(self.all_leftgoing)


def mode_split(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    s: complex,
    *,
    tol: float = DEFAULT_UNIT_TOL,
    delta: float = 1e-4,
) -> ModeSplit:
    """Partition the interior kappa roots at s into mode families.

    Plan 48 item 2.1. Sign convention: u_t = -sum_k a_k u_{n+k} (the
    ``kappa_roots`` docstring convention); under the "raw operator
    weights" reading all w-signs flip.

    Roots of the characteristic polynomial Q(kappa; s) = 0 are split into
    ``rightgoing`` (|kappa| < 1 - tol), ``leftgoing`` (|kappa| > 1 + tol),
    and ``borderline`` (unit-modulus within tol). Borderline roots are
    classified by the shared perturbation test
    ``gks_kreiss.perturbed_root_directions``: inward under Re(s) += delta
    means the rightgoing family, outward the leftgoing family.

    Evaluate on the imaginary axis at s = i*omega + eps_imag (consistent
    with ``make_s_grid``'s strip) so borderline roots are resolved by the
    half-plane limit — the strict partition — rather than the
    perturbation test.

    For a centered 2p+1-point interior (offsets spanning [-p, p], so the
    characteristic polynomial has degree 2p) with Re(s) > 0 the partition
    must contain exactly p rightgoing and p leftgoing roots; this is
    checked (borderline roots counted via their resolved family) and a
    ValueError raised on violation.

    Parameters
    ----------
    interior_weights, interior_offsets
        Interior stencil specification.
    s : complex
        Laplace frequency.
    tol : float
        Half-width of the borderline modulus band (see DEFAULT_UNIT_TOL).
    delta : float
        Re(s) perturbation magnitude for resolving borderline roots.

    Raises
    ------
    DefectiveKappaError
        If borderline roots coalesce (sonic / branch point, e.g. the E2
        interior at s = +/- i): the partition is not defined there and
        callers must route through the defective path (plan 48 item 0.1).
    ValueError
        If a centered interior at Re(s) > 0 does not split exactly p/p.
    """
    offsets = np.asarray(interior_offsets, dtype=int)
    p = -int(np.min(offsets))
    s = complex(s)

    all_roots, _, _ = kappa_roots(interior_weights, interior_offsets, s)
    moduli = np.abs(all_roots)

    inside = _sort_roots(all_roots[moduli < 1.0 - tol])
    outside = _sort_roots(all_roots[moduli > 1.0 + tol])
    border = _sort_roots(
        all_roots[(moduli >= 1.0 - tol) & (moduli <= 1.0 + tol)]
    )

    borderline: tuple[tuple[complex, str], ...] = ()
    if len(border) > 0:
        directions = perturbed_root_directions(
            interior_weights, interior_offsets, s, border, delta=delta
        )
        borderline = tuple(
            (complex(k), "rightgoing" if d == "inward" else "leftgoing")
            for k, d in zip(border, directions)
        )

    split = ModeSplit(
        rightgoing=inside,
        leftgoing=outside,
        borderline=borderline,
        s=s,
        tol=float(tol),
    )

    # Centered 2p+1 interior at Re(s) > 0: exactly p rightgoing and p
    # leftgoing (degree-2p characteristic polynomial; plan 48 item 2.1).
    centered = int(np.max(offsets)) == p
    if centered and s.real > 0:
        if split.n_rightgoing != p or split.n_leftgoing != p:
            raise ValueError(
                f"Centered interior (p={p}) at s={s} with Re(s) > 0 must "
                f"split into exactly p rightgoing and p leftgoing modes; "
                f"got {split.n_rightgoing} rightgoing / "
                f"{split.n_leftgoing} leftgoing "
                f"(borderline: {len(borderline)})"
            )

    return split


# ---------------------------------------------------------------------------
# Reflection matrix (plan 48 item 2.2)
# ---------------------------------------------------------------------------

_COALESCE_TOL = 1e-7
"""Pairwise separation below which two rightgoing roots make the mode
basis defective (matches kappa_roots' repeat_tol): the kappa-columns of
M(s) degenerate (Vandermonde-like) and R would spike like the reciprocal
separation — such points must route through the DefectiveKappaError path
(plan 48 items 0.1 / 2.2), never report as clean values."""


def reflection_system(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list,
    s: complex,
    *,
    row_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, ModeSplit]:
    """Assemble the reflection system (M(s), B(s)) at Laplace frequency s.

    Plan 48 item 2.2. Sign convention: u_t = -sum_k a_k u_{n+k} (see
    module docstring); under the "raw operator weights" reading all
    w-signs flip.

    M(s) is the augmented m x m Kreiss matrix of item 0.2 (identical
    entries to ``gks_kreiss.kreiss_matrix_general``, with the alpha-block
    columns ordered by ``ModeSplit``'s deterministic rightgoing ordering)
    and B(s) is the m x n_leftgoing incident-load matrix whose k-th
    column is

        b_k[i] = s mu_k^{n_i} [n_i >= m-p]
                 + sum_{j >= m-p, j in support(i)} w_{i,j} mu_k^j

    for the k-th leftgoing root mu_k — i.e. B's columns are M's own
    root-column function c(z) evaluated at the leftgoing roots.

    Exposed separately from ``reflection_matrix`` so tests can exercise
    the normalization-invariance property at the assembled-equation
    level: scaling row i of [M | B] jointly leaves the solution of
    M x = -B (hence R) unchanged, while sigma_min(M) changes.

    Parameters
    ----------
    interior_weights, interior_offsets
        Interior stencil specification.
    boundary_rows : list[BoundaryRow]
        m closure rows as (weights, column_offsets) pairs, m >= p.
    s : complex
        Laplace frequency (typically on the make_s_grid strip
        s = i*omega + eps_imag).
    row_indices : list[int], optional
        Grid node n_i of each closure row (default 0..m-1).

    Returns
    -------
    (M, B, split) : tuple[np.ndarray, np.ndarray, ModeSplit]
        M is m x m, B is m x n_leftgoing, and split carries the mode
        partition whose ``all_rightgoing`` ordering indexes the
        alpha-block columns of M (and the rows of R) and whose
        ``all_leftgoing`` ordering indexes the columns of B (and of R).

    Raises
    ------
    DefectiveKappaError
        If the mode partition is defective at this s (coalescing
        borderline roots, e.g. the E2 sonic points s = +/- i) or two
        rightgoing roots coalesce within _COALESCE_TOL.
    ValueError
        If m < p, row_indices has the wrong length, a closure row
        references a negative grid node, or the rightgoing-root count
        differs from p (M would not be square).
    """
    offsets = np.asarray(interior_offsets, dtype=int)
    p = -int(np.min(offsets))
    m = len(boundary_rows)
    if m < p:
        raise ValueError(
            f"Number of boundary rows ({m}) < admissible root count p={p}; "
            "the reflection system requires m >= p"
        )
    if row_indices is None:
        row_indices = list(range(m))
    if len(row_indices) != m:
        raise ValueError(
            f"row_indices has length {len(row_indices)} != number of "
            f"boundary rows ({m})"
        )

    split = mode_split(interior_weights, interior_offsets, s)
    rightgoing = split.all_rightgoing
    leftgoing = split.all_leftgoing
    if len(rightgoing) != p:
        raise ValueError(
            f"Rightgoing root count ({len(rightgoing)}) != p = L_left "
            f"({p}) at s={s}; the reflection system matrix would not be "
            "square"
        )
    # Within-family coalescence: two rightgoing roots within _COALESCE_TOL
    # make M's kappa-columns degenerate — route through the defective path
    # (plan 48 items 0.1 / 2.2), never report a clean (spiking) R.
    for a in range(len(rightgoing)):
        for b in range(a + 1, len(rightgoing)):
            if abs(rightgoing[a] - rightgoing[b]) < _COALESCE_TOL:
                raise DefectiveKappaError(
                    f"Coalescing rightgoing kappa roots at s={s}: "
                    f"{rightgoing[a]} and {rightgoing[b]} within "
                    f"{_COALESCE_TOL}"
                )

    n_nodal = m - p
    n_left = len(leftgoing)
    s = complex(s)
    M = np.zeros((m, m), dtype=complex)
    B = np.zeros((m, n_left), dtype=complex)
    for i, (weights, col_offsets) in enumerate(boundary_rows):
        weights = np.asarray(weights, dtype=complex)
        col_offsets = np.asarray(col_offsets, dtype=int)
        n_i = int(row_indices[i])
        # Temporal contribution s * phi_{n_i}
        if n_i < n_nodal:
            M[i, n_i] += s
        else:
            M[i, n_nodal:] += s * rightgoing**n_i
            B[i, :] += s * leftgoing**n_i
        # Spatial operator contribution sum_j w_{ij} phi_j
        for w, j in zip(weights, col_offsets):
            if j < 0:
                raise ValueError(
                    f"Boundary row {i} references negative grid node j={j}; "
                    "left half-line closure rows must have offsets >= 0"
                )
            if j < n_nodal:
                M[i, j] += w
            else:
                M[i, n_nodal:] += w * rightgoing**j
                B[i, :] += w * leftgoing**j

    return M, B, split


def reflection_matrix(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list,
    s: complex,
    *,
    row_indices: list[int] | None = None,
) -> np.ndarray:
    """Compute the reflection matrix R(s) of a left-boundary closure.

    Plan 48 item 2.2 (see module docstring for the math): solves
    M(s) x_k = -b_k(s) for each unit-amplitude incident leftgoing mode
    mu_k and returns the alpha-block rows of the solutions,

        R[l, k] = amplitude of rightgoing mode kappa_l reflected from
                  incident leftgoing mode mu_k,

    with rows ordered by ``ModeSplit.all_rightgoing`` and columns by
    ``ModeSplit.all_leftgoing`` (both deterministic). R is
    p x n_leftgoing: p x p for centered interiors at Re(s) > 0, and
    p x 0 for one-sided interiors with no leftgoing modes (first-order
    upwind — nothing to reflect).

    Raises
    ------
    DefectiveKappaError
        Defective mode basis at this s (see reflection_system).
    numpy.linalg.LinAlgError
        det M(s) = 0 — a GKS eigensolution point (the same points
        ``kreiss_stability_check`` flags); near such s, ||R|| -> infinity
        (singularity equivalence).
    """
    M, B, split = reflection_system(
        interior_weights, interior_offsets, boundary_rows, s,
        row_indices=row_indices,
    )
    p = len(split.all_rightgoing)
    n_nodal = M.shape[0] - p
    if B.shape[1] == 0:
        return np.zeros((p, 0), dtype=complex)
    x = np.linalg.solve(M, -B)
    return x[n_nodal:, :]


@dataclass(frozen=True)
class ReflectionResult:
    """Result of the reflection margin sweep (plan 48 item 2.2)."""

    max_norm: float
    """max over the s-grid of ||R(s)||_2 (spectral norm). 0.0 when there
    are no leftgoing modes (R is p x 0 — nothing to reflect); inf when
    det M(s) = 0 at a grid point; nan when no grid point was cleanly
    evaluable."""

    argmax_s: complex | None
    """The s at which the maximum was attained (None if no grid point was
    cleanly evaluable)."""

    finite: bool
    """True iff every grid point evaluated to a finite clean value: no
    det M(s) = 0 blowup AND no defective mode basis anywhere on the grid
    (a nonempty ``defective_points`` forces False — sonic/coalescence
    points must route through the defective path, never count as clean;
    plan 48 items 0.1 / 2.2)."""

    defective_points: tuple[complex, ...] = ()
    """Grid points where the mode basis was defective
    (DefectiveKappaError: coalescing borderline roots at sonic points, or
    coalescing rightgoing roots). Their norms entries are nan."""

    norms: np.ndarray | None = field(default=None, repr=False)
    """||R(s)||_2 per grid point (nan at defective points, inf at
    det-zero points)."""

    s_grid: np.ndarray | None = field(default=None, repr=False)
    """The s-grid swept (flattened)."""


def default_reflection_s_grid(
    *,
    imag_max: float = 20.0,
    n_points: int = 241,
    eps_imag: float = 1e-6,
    min_omega: float = 1e-3,
) -> np.ndarray:
    """Imaginary-axis strip grid s = i*omega + eps_imag for max_reflection.

    Matches ``make_s_grid``'s strip convention (Re(s) = eps_imag = 1e-6).
    Points with |omega| < min_omega are excluded, mirroring
    kreiss_stability_check's min_s_magnitude mask: the structural benign
    determinant zeros at s = 0 (items 0.4 / 0.5d — the conservative
    consistency zero and the Dirichlet tail-constant zero) are poles of
    R(s), a parameterization artifact of the mode basis rather than a GKS
    instability, and would otherwise dominate the margin.
    """
    omega = np.linspace(-imag_max, imag_max, n_points)
    omega = omega[np.abs(omega) >= min_omega]
    return eps_imag + 1j * omega


def max_reflection(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list,
    *,
    row_indices: list[int] | None = None,
    s_grid: np.ndarray | None = None,
) -> ReflectionResult:
    """Reflection margin: max over the imaginary-axis s-grid of ||R(s)||_2.

    Plan 48 item 2.2. Defaults to ``default_reflection_s_grid()`` (the
    make_s_grid strip with the near-origin structural-pole exclusion).

    Grid points where the mode basis is defective (sonic-point
    coalescence, e.g. E2 at s = +/- i exactly) are recorded in
    ``defective_points`` and set ``finite=False`` — they route through
    the defective path (plan 48 items 0.1 / 2.2), never report as clean
    values. Grid points with det M(s) = 0 give norm inf (and
    ``finite=False``): the singularity-equivalence witness of a GKS
    eigensolution on the grid.
    """
    if s_grid is None:
        s_grid = default_reflection_s_grid()
    s_flat = np.asarray(s_grid, dtype=complex).ravel()

    norms = np.full(len(s_flat), np.nan)
    defective: list[complex] = []
    for i, s in enumerate(s_flat):
        try:
            R = reflection_matrix(
                interior_weights, interior_offsets, boundary_rows, s,
                row_indices=row_indices,
            )
        except DefectiveKappaError:
            defective.append(complex(s))
            continue
        except np.linalg.LinAlgError:
            norms[i] = np.inf
            continue
        if R.shape[0] == 0 or R.shape[1] == 0:
            norms[i] = 0.0
        else:
            norms[i] = float(np.linalg.norm(R, ord=2))

    if np.all(np.isnan(norms)):
        return ReflectionResult(
            max_norm=float("nan"),
            argmax_s=None,
            finite=False,
            defective_points=tuple(defective),
            norms=norms,
            s_grid=s_flat,
        )

    argmax = int(np.nanargmax(norms))
    max_norm = float(norms[argmax])
    finite = bool(np.isfinite(max_norm)) and not defective
    return ReflectionResult(
        max_norm=max_norm,
        argmax_s=complex(s_flat[argmax]),
        finite=finite,
        defective_points=tuple(defective),
        norms=norms,
        s_grid=s_flat,
    )
