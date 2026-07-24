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

Right-end closures: mirror convention (plan 48 item 2.3)
--------------------------------------------------------
The coordinate mirror n -> N - n' is **pure index reversal, no
negation**: interior weights map a_k -> a_{-k} (offsets negated, weights
unchanged), which maps kappa -> 1/kappa; closure rows likewise get
offsets negated and node indices remapped to distance-from-the-right-end.
Do NOT reverse-and-negate: revneg gives kappa'(s) = 1/kappa(-s) — for
the first-order upwind toy revneg yields kappa' = 1 - s while the true
mirror is 1/kappa = 1 + s. (The negation belongs to *codegen's
construction* of physical right-wall coefficients from left-wall ones —
a statement about the physical rows, not about the coordinate mirror.)
``mirror_closure`` covers both entry points:

- ``from_physical_right=True``: the inputs ARE the physical right-end
  rows (column offsets measured from the right-end node, non-positive
  into the domain — e.g. read off the bottom rows of an assembled D
  matrix). Pure reversal is applied and the result feeds
  ``reflection_matrix`` unchanged. Diagnostics/tests use this entry.
- ``from_physical_right=False``: the inputs are the STORED LEFT-END rows
  (``CutCellEvaluator.rows``, ``phs.uniform_boundary_weights_rbf``) and
  the physical right end is represented via the nu = 1 codegen revneg
  symmetry ``D[n-1-i, n-1-j] = -B[i, j]`` (``phs.build_diff_matrix_rbf``,
  ``brady2d_stability._build_classical_diff_matrix``, and
  ``derivative.cpp``'s right blocks). The composite map
  (reversal of revneg) is: interior reversed, closure weights negated,
  offsets and node indices unchanged. This is the entry the production
  pipeline uses — plan 48 items 2.4 (``reflection_over_psi`` with
  ``side="right"``) and 3.1 (``build_interval_operator``'s right block).
  The negation is nu-odd-specific: for even derivative orders the
  codegen symmetry has no sign flip and this entry point must not be
  used (plan 48 scope is nu = 1; the parabolic program is a non-goal).

Axis identity of the mirrored reflection matrix (measured; documented
for the 2.3 reviewer): for a real ANTISYMMETRIC interior the mirrored
interior equals the original with s -> -s, so A_R(s) is A_L(-s) up to
the modulus-based mode-family matching. Where the kappa moduli split
strictly across the imaginary axis (the super-sonic segment; E2:
|omega| > 1) that matching commutes with conjugation and
A_R(i omega) = conj(A_L(i omega)) after canonical ordering of the mode
basis — the norm equality ||A_R(i omega)|| = ||A_L(i omega)|| of plan 48
item 2.3(b) holds there exactly (to O(eps_imag) on the strip). On the
SUB-sonic segment (unimodular axis roots; E2: |omega| < 1) the
Re(s) -> 0+ admissibility limits at the two ends classify the
propagating modes from OPPOSITE sides of the sonic branch: the
incident/reflected roles swap (time reversal), and the measured relation
is the reciprocal one — A_R(i omega) * conj(A_L(i omega)) -> 1 for
p = 1, i.e. ||A_R|| ||A_L|| -> 1 as eps -> 0 — NOT norm equality. Both
behaviors are pinned by the mirror tests in tests/test_reflection.py.

psi-swept reflection margin (plan 48 item 2.4)
----------------------------------------------
``reflection_over_psi`` worst-cases the reflection margin over the
fractional wall position psi for a cut-cell closure, reusing item 1.3's
infrastructure exactly: the default psi grid is
``cutcell_normal_modes.default_psi_grid`` (clustered near 0, matched to
the C++ clamp [1e-4, 1 - 1e-4]) and each psi point passes through the
item-1.2 feasibility guard (``_vanishing_denominators`` ->
:class:`~stencil_gen.cutcell_normal_modes.CutCellInfeasible`) before any
assembly — denominator vanishing is a *parameterization* failure and is
partitioned out, never folded into the margin. Closure rows come from
the cached evaluator (item 1.1), so the symbolic derivation cost is paid
once per scheme. ``side="left"`` analyzes the stored left-end Dirichlet
rows directly; ``side="right"`` maps them through
``mirror_closure(..., from_physical_right=False)`` (the production
revneg entry point above). The result is a
:class:`CutCellReflectionSweep` — a
:class:`~stencil_gen.cutcell_normal_modes.CutCellPsiSweep` subtype whose
reflection margin lives in ``max_reflection`` / ``argmax_psi``.

Report wiring (same item): ``brady2d_stability.cutcell_stability_report``
populates the opt-in ``StabilityReport.layer_cutcell`` dict from this
sweep plus item 1.3's Kreiss sweep. It is NOT part of the L1-L8 cascade;
objective/alias wiring is plan 49.

Two-boundary round-trip criterion (plan 48 item 3.3)
----------------------------------------------------
Trefethen 1985 (Math. Comp. 45) Lemma 1: on a finite interval the two
boundary closures interact only through the interior modes, and the pair
supports a GKS eigensolution at s exactly when one round trip of the
mode amplitudes returns them unchanged. With the mode split at s
(K = diag rightgoing kappa, |kappa| <= 1; Mdiag = diag leftgoing mu,
|mu| >= 1) and the per-end reflection matrices A_L(s), A_R(s), the
round-trip cycle applies (right-to-left) A_L first, then the rightgoing
transit, then A_R, then the leftgoing return:

    E_Dj(s) = Mdiag^{-Dj} . A_R(s) . K^{Dj} . A_L(s)

(any cyclic rotation is similar and has the same spectrum). Derivation
pinning the ordering AND the reference nodes (an off-by-one rephases E):
let beta be the leftgoing amplitudes referenced at the LEFT reference
node (grid node 0 — the node the left closure's mode ansatz kappa^n,
mu^n counts from) and let the RIGHT reference node be grid node Dj (the
mirrored frame's n' = 0 node, n' = Dj - n), so Dj = N - 1 for an N-node
interval. In the modal window the interval solution is
phi_n = sum_k beta_k mu_k^n + sum_l alpha_l kappa_l^n; the left closure
equations give alpha = A_L beta (item 2.2); re-referencing at node Dj
turns the amplitudes into (K^{Dj} alpha, Mdiag^{Dj} beta), and the right
closure (in the item-2.3 mirrored frame, where mirrored-rightgoing =
physically leftgoing) gives Mdiag^{Dj} beta = A_R K^{Dj} alpha. Chaining:
beta = Mdiag^{-Dj} A_R K^{Dj} A_L beta = E_Dj beta, so a pair
eigensolution exists iff 1 is an eigenvalue of E_Dj(s). The transit
factors are contractions on the axis (|kappa| <= 1 <= |mu| for
Re(s) >= 0), so E is numerically safe to form and
rho(E_Dj) <= ||A_L|| ||A_R|| there — the decoupled screen
``reflection_product`` upper-bounds the round-trip radius.

Hypothesis (Lemma 1): each end must be separately half-line stable at s
(M_L(s), M_R(s) invertible) for the criterion to be meaningful — a
per-end det-zero is a SINGLE-boundary GKS eigensolution and is the
per-end tiers' job (items 1.2 / 2.2 on the mirrored closure), not the
round trip's. The 3.2 scan's outflow law is exactly such a case: the
E2_1 cut Dirichlet closure at the RIGHT (outflow) end alone carries a
right-localized eigenvalue at s = (1 - psi_R)/(1 + psi_R), so cut-right
Dirichlet pairs are unstable regardless of rho(E_Dj) on the axis.
(Dirichlet at an outflow end is ill-posed at the continuous level —
data imposed against the outgoing characteristic — so that eigenvalue
is the expected symptom, not a closure defect; such configurations are
used only as validation anchors. Leeward ends take the outflow/floating
closure in practice.)

Do not assume monotonicity of rho(E_Dj) in Dj — it is false: on the
axis K and Mdiag are pure phases for the propagating modes, so
rho(E_Dj) oscillates with Dj (Dj-specific resonances). The reported
``round_trip_radius`` therefore maxes over BOTH the s-grid and a Dj
range {Dj_min, ..., Dj_min + 5}.

Minimum separation: the criterion requires each closure's column support
to lie inside the other end's modal region, i.e. the two supports may
not overlap: Dj >= t_L + t_R - 1 (and closure rows may not share a grid
node: Dj >= max_node_L + max_node_R + 1). This is STRICTER than item
3.1's assembly minimum (which permits column overlap); violations raise
the same :class:`~stencil_gen.interval_operator.IntervalTooShort`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stencil_gen.cutcell_eval import make_cutcell_evaluator
from stencil_gen.cutcell_normal_modes import (
    FEASIBILITY_TOL,
    CutCellInfeasible,
    CutCellPsiSweep,
    _vanishing_denominators,
    default_psi_grid,
)
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
    "MirroredClosure",
    "mirror_closure",
    "CutCellReflectionSweep",
    "reflection_over_psi",
    "RoundTripFactors",
    "RoundTripResult",
    "min_round_trip_delta_j",
    "round_trip_factors",
    "round_trip_matrix",
    "round_trip_check",
    "cutcell_round_trip",
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


# ---------------------------------------------------------------------------
# Right-end closures: mirror convention (plan 48 item 2.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MirroredClosure:
    """A right-end closure mapped to the left-half-line analysis frame.

    Produced by :func:`mirror_closure`. The fields plug directly into the
    left-half-line machinery — ``reflection_matrix`` /
    ``reflection_system`` / ``max_reflection`` (and
    ``gks_kreiss.kreiss_stability_check``) — to yield the right-end
    reflection matrix A_R(s) / right-end GKS verdict:

        A_R = reflection_matrix(mc.interior_weights, mc.interior_offsets,
                                mc.boundary_rows, s,
                                row_indices=mc.row_indices)

    In the mirrored frame the mode families keep their left-half-line
    meaning, which corresponds to the physically correct right-end roles:
    mirrored-rightgoing (|kappa| < 1, the R-rows) are the modes radiated
    by the right boundary (physically leftgoing), and mirrored-leftgoing
    (the R-columns) are the modes incident on it (physically rightgoing).
    """

    interior_weights: np.ndarray
    """Mirrored interior weights (pure reversal: values unchanged,
    reordered to match ``interior_offsets``)."""

    interior_offsets: np.ndarray
    """Mirrored interior offsets (negated, sorted ascending)."""

    boundary_rows: list
    """Mirrored closure rows as ``BoundaryRow`` (weights, col_offsets)
    pairs with col_offsets >= 0 renumbered to distance from the right
    end."""

    row_indices: list
    """Grid node (distance from the right end) of each mirrored row."""


def mirror_closure(
    boundary_rows: list,
    interior: tuple[np.ndarray, np.ndarray],
    *,
    from_physical_right: bool,
    row_indices: list[int] | None = None,
) -> MirroredClosure:
    """Map a right-end closure into the left-half-line analysis frame.

    Plan 48 item 2.3. The coordinate mirror n -> N - n' is **pure index
    reversal, no negation** (see module docstring; revneg is the
    historical sign-error trap — it belongs to codegen's construction of
    the physical right-wall rows, not to the coordinate mirror). The
    interior is mirrored identically for both entry points: offsets
    negated, weights unchanged (kappa -> 1/kappa).

    Parameters
    ----------
    boundary_rows : list[BoundaryRow]
        Closure rows as (weights, column_offsets) pairs; interpretation
        depends on ``from_physical_right``.
    interior : (weights, offsets)
        The interior stencil in the ORIGINAL (physical, left-to-right)
        coordinate.
    from_physical_right : bool
        Selects the entry point:

        - ``True`` — ``boundary_rows`` are the physical right-end rows in
          the physical coordinate, column offsets measured from the
          right-end node (non-positive, pointing into the domain) and
          ``row_indices`` likewise non-positive (default ``[0, -1, -2,
          ...]``: row i sits i nodes in from the right end). This is how
          rows read off the bottom of an assembled operator look. Pure
          reversal is applied: offsets and node indices negated, weights
          unchanged.
        - ``False`` — ``boundary_rows`` are the STORED LEFT-END rows
          (column offsets >= 0, ``row_indices`` default ``[0, 1, ...]``),
          and the physical right end is their nu = 1 codegen revneg image
          ``D[n-1-i, n-1-j] = -B[i, j]`` (``phs.build_diff_matrix_rbf``,
          ``brady2d_stability._build_classical_diff_matrix``,
          ``derivative.cpp``). The composite map (pure reversal of the
          revneg rows) is: closure weights negated, offsets and node
          indices unchanged. Production callers use this entry point
          (plan 48 items 2.4 ``reflection_over_psi(side="right")`` and
          3.1 ``build_interval_operator``); it is nu-odd-specific — for
          even nu the codegen symmetry carries no sign flip.
    row_indices : list[int], optional
        Grid node of each closure row in the entry point's convention
        (see above).

    Returns
    -------
    MirroredClosure
        Mirrored interior + closure rows + row indices, ready for the
        left-half-line machinery.

    Raises
    ------
    ValueError
        If ``row_indices`` has the wrong length, or offsets / node
        indices have the wrong sign for the selected entry point.
    """
    int_weights, int_offsets = interior
    int_weights = np.asarray(int_weights)
    int_offsets = np.asarray(int_offsets, dtype=int)
    # Pure index reversal of the interior (both entry points): offsets
    # negated, weights unchanged, sorted ascending by mirrored offset.
    order = np.argsort(-int_offsets)
    mirrored_iw = int_weights[order].copy()
    mirrored_io = -int_offsets[order]

    m = len(boundary_rows)
    if row_indices is None:
        row_indices = [-i for i in range(m)] if from_physical_right else list(
            range(m)
        )
    if len(row_indices) != m:
        raise ValueError(
            f"row_indices has length {len(row_indices)} != number of "
            f"boundary rows ({m})"
        )

    rows_out: list = []
    idx_out: list = []
    for i, (weights, col_offsets) in enumerate(boundary_rows):
        weights = np.asarray(weights)
        col_offsets = np.asarray(col_offsets, dtype=int)
        n_i = int(row_indices[i])
        if from_physical_right:
            if np.any(col_offsets > 0):
                raise ValueError(
                    f"Physical right-end row {i} has positive column "
                    f"offsets {col_offsets[col_offsets > 0]}; offsets are "
                    "measured from the right-end node and must be <= 0 "
                    "(support inside the domain)"
                )
            if n_i > 0:
                raise ValueError(
                    f"Physical right-end row {i} has positive node index "
                    f"{n_i}; node indices are measured from the right-end "
                    "node and must be <= 0"
                )
            row_order = np.argsort(-col_offsets)
            rows_out.append(
                (weights[row_order].copy(), -col_offsets[row_order])
            )
            idx_out.append(-n_i)
        else:
            if np.any(col_offsets < 0):
                raise ValueError(
                    f"Stored left-end row {i} has negative column offsets "
                    f"{col_offsets[col_offsets < 0]}; left closure rows "
                    "must have offsets >= 0"
                )
            if n_i < 0:
                raise ValueError(
                    f"Stored left-end row {i} has negative node index "
                    f"{n_i}"
                )
            rows_out.append((-weights, col_offsets.copy()))
            idx_out.append(n_i)

    return MirroredClosure(
        interior_weights=mirrored_iw,
        interior_offsets=mirrored_io,
        boundary_rows=rows_out,
        row_indices=idx_out,
    )


# ---------------------------------------------------------------------------
# psi-swept reflection margin (plan 48 item 2.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CutCellReflectionSweep(CutCellPsiSweep):
    """Worst-case-over-psi reflection margin for a cut-cell closure.

    Plan 48 item 2.4: a :class:`~stencil_gen.cutcell_normal_modes.
    CutCellPsiSweep` subtype (so ``reflection_over_psi`` returns the item
    1.3 sweep container, as the plan specifies) whose reflection margin
    lives in the fields below. The two aggregations mirror the base class,
    with "worse" now meaning LARGER reflection:

    - **margin**: ``max_reflection`` / ``argmax_psi`` are the max (and
      argmax) over feasible psi of the per-psi
      ``ReflectionResult.max_norm``. An inf per-psi norm (det M(s) = 0 on
      the s-grid — a GKS-eigensolution witness) propagates: the margin is
      inf at that psi. Per-psi nan (no cleanly evaluable grid point) is
      excluded; if NO feasible psi produced a clean value,
      ``max_reflection`` is nan and ``argmax_psi`` / ``argmax_s`` are
      None (the ``ReflectionResult`` convention).
    - **stability routing**: ``any_unstable`` is True iff any feasible
      psi returned ``ReflectionResult.finite is False`` — either an inf
      blowup or defective grid points (sonic/coalescence), both of which
      must route through the non-clean path (items 0.1 / 2.2), never
      count as clean margins.

    The inherited Kreiss-margin fields (``worst_sigma_min`` /
    ``worst_psi``) are NOT computed by the reflection sweep; they carry
    the base class's no-finite-margin convention (inf / None). The Kreiss
    margin comes from ``kreiss_margin_over_psi`` (item 1.3);
    ``brady2d_stability.cutcell_stability_report`` reports both side by
    side.

    ``infeasible_psi`` / ``per_psi`` keep their base-class meaning:
    infeasible psi points (vanishing structural denominator — a
    *parameterization* failure) carry
    :class:`~stencil_gen.cutcell_normal_modes.CutCellInfeasible` in
    ``per_psi`` and are excluded from both aggregations; feasible points
    carry :class:`ReflectionResult`.
    """

    max_reflection: float
    """Max over feasible psi of the per-psi ``ReflectionResult.max_norm``
    (inf if det M(s) = 0 on the grid at some psi; nan if no feasible psi
    produced a clean value)."""

    argmax_psi: float | None
    """The psi attaining ``max_reflection`` (None if no clean value)."""

    argmax_s: complex | None
    """The s-grid point attaining ``max_reflection`` at ``argmax_psi``
    (None if no clean value)."""

    side: str
    """Which physical end was analyzed: "left" (stored left-end Dirichlet
    rows, direct) or "right" (their revneg image via
    ``mirror_closure(..., from_physical_right=False)``)."""


def reflection_over_psi(
    scheme,
    alpha: dict,
    psi_grid=None,
    side: str = "left",
    *,
    feasibility_tol: float = FEASIBILITY_TOL,
    s_grid: np.ndarray | None = None,
) -> CutCellPsiSweep:
    """Worst-case the reflection margin over a psi grid (plan 48 item 2.4).

    Runs :func:`max_reflection` on the cut-cell Dirichlet closure at each
    psi of *psi_grid* and aggregates into a :class:`CutCellReflectionSweep`
    (a ``CutCellPsiSweep`` subtype). Reuses item 1.3's infrastructure
    exactly: the default grid is
    ``cutcell_normal_modes.default_psi_grid()`` and each point passes the
    item-1.2 feasibility guard before assembly — a vanishing structural
    denominator partitions the psi into ``infeasible_psi`` (a
    *parameterization* failure, never conflated with a reflection margin).
    Closure rows come from the cached evaluator (item 1.1), so the
    symbolic derivation cost is paid once per scheme.

    The **Dirichlet (wall-eliminated) block is the canonical analysis
    object** (plan 48 constraint). Sign convention: the raw
    derivative-operator rows are passed through unflipped, exactly as
    ``kreiss_cutcell_check`` feeds them to the module-wide
    ``u_t = -sum_k a_k u_{n+k}`` convention.

    Parameters
    ----------
    scheme : SchemeParams
        Cut-cell scheme (e.g. ``temo.E2_1``, ``temo.E4_1``).
    alpha : dict
        Mapping from alpha symbol (or name) to value; must cover all of
        the scheme's free parameters.
    psi_grid : sequence of float, optional
        psi values to sweep; defaults to ``default_psi_grid()``. Must be
        non-empty; every psi must lie in (0, 1].
    side : str
        "left" (default): analyze the stored left-end rows directly.
        "right": analyze the physical right end they represent via the
        nu = 1 codegen revneg symmetry, i.e.
        ``mirror_closure(rows, interior, from_physical_right=False)``
        (item 2.3's production entry point).
    feasibility_tol : float
        |denominator| threshold for the per-psi feasibility guard.
    s_grid : np.ndarray, optional
        Forwarded to :func:`max_reflection` (default: the
        ``default_reflection_s_grid()`` strip).

    Returns
    -------
    CutCellReflectionSweep
        Worst-cased reflection margin, non-clean routing flag, and the
        per-psi partition of :class:`ReflectionResult` vs.
        :class:`~stencil_gen.cutcell_normal_modes.CutCellInfeasible`.
    """
    if side not in ("left", "right"):
        raise ValueError(f'side must be "left" or "right", got {side!r}')
    if psi_grid is None:
        psi_grid = default_psi_grid()
    psi_values = [float(p) for p in psi_grid]
    if not psi_values:
        raise ValueError("psi_grid must contain at least one psi value")

    evaluator = make_cutcell_evaluator(scheme)
    interior = evaluator.interior()

    per_psi: dict = {}
    infeasible_psi: list = []
    any_unstable = False
    max_norm = float("nan")
    argmax_psi: float | None = None
    argmax_s: complex | None = None

    for psi in psi_values:
        if not 0.0 < psi <= 1.0:
            raise ValueError(f"psi must be in (0, 1], got {psi}")

        # Feasibility guard BEFORE assembly (item 1.2, reused verbatim):
        # a vanishing denominator means the closure coefficients do not
        # exist at this point — partition it out, never fold it in.
        vanishing = _vanishing_denominators(
            evaluator, psi, alpha, feasibility_tol
        )
        if vanishing:
            per_psi[psi] = CutCellInfeasible(
                scheme=scheme,
                psi=psi,
                alpha=tuple(
                    (sym.name, v)
                    for sym, v in zip(
                        evaluator.alpha_symbols, evaluator._alpha_args(alpha)
                    )
                ),
                vanishing=vanishing,
                feasibility_tol=feasibility_tol,
            )
            infeasible_psi.append(psi)
            continue

        boundary_rows, row_indices = evaluator.rows(psi, alpha)
        row_idx = [int(i) for i in row_indices]
        if side == "right":
            mc = mirror_closure(
                boundary_rows,
                interior,
                from_physical_right=False,
                row_indices=row_idx,
            )
            iw, io = mc.interior_weights, mc.interior_offsets
            boundary_rows, row_idx = mc.boundary_rows, mc.row_indices
        else:
            iw, io = interior

        result = max_reflection(
            iw, io, boundary_rows, row_indices=row_idx, s_grid=s_grid
        )
        per_psi[psi] = result
        if not result.finite:
            any_unstable = True
        if not np.isnan(result.max_norm) and (
            argmax_psi is None or result.max_norm > max_norm
        ):
            max_norm = float(result.max_norm)
            argmax_psi = psi
            argmax_s = result.argmax_s

    return CutCellReflectionSweep(
        # Kreiss-margin fields: not computed here (base-class
        # no-finite-margin convention; see class docstring).
        worst_sigma_min=float("inf"),
        worst_psi=None,
        any_unstable=any_unstable,
        infeasible_psi=infeasible_psi,
        per_psi=per_psi,
        max_reflection=max_norm,
        argmax_psi=argmax_psi,
        argmax_s=argmax_s,
        side=side,
    )


# ---------------------------------------------------------------------------
# Two-boundary round-trip criterion (plan 48 item 3.3,
# Trefethen 1985 Math. Comp. 45 Lemma 1)
# ---------------------------------------------------------------------------

_MODE_MATCH_TOL = 1e-6
"""Tolerance for matching a mirrored-frame root z to the reciprocal of an
original-frame root w via |z*w - 1| < tol.  The mirrored characteristic
polynomial is Q_m(z; s) = Q(1/z; s) (reversed coefficient vector), so its
roots are exact reciprocals of the original roots up to numpy.roots
rounding (~1e-10 for the well-conditioned degree-2p polynomials here);
1e-6 leaves headroom while staying far below the 1e-7 coalescence guard
that routes genuinely ambiguous bases through DefectiveKappaError."""


def _support_extent(boundary_rows: list) -> int:
    """t = 1 + max column offset over the closure rows (support columns
    0..t-1 in the rows' own frame)."""
    return 1 + max(
        int(np.max(np.asarray(cols, dtype=int))) for _, cols in boundary_rows
    )


def min_round_trip_delta_j(
    left_boundary_rows: list,
    right_mirrored: MirroredClosure,
    *,
    left_row_indices: list[int] | None = None,
) -> int:
    """Minimum reference-node separation Dj for the round-trip criterion.

    Plan 48 item 3.3. The criterion is exact only when the two closures
    interact exclusively through the interior modes, which requires

    - **column supports not to overlap**: the left support occupies grid
      nodes 0..t_L-1 and the right support nodes Dj-(t_R-1)..Dj (right
      reference node = grid node Dj), so Dj >= t_L + t_R - 1; and
    - **closure rows not to share a grid node**:
      Dj >= max_node_L + max_node_R + 1.

    ``Dj_min = max`` of the two. For an N-node interval Dj = N - 1, so
    this is STRICTER than item 3.1's assembly minimum
    ``N >= max(m_L + m_R, t_L, t_R)`` — assembly permits the two column
    supports to overlap, the modal round trip does not (violations raise
    the same :class:`~stencil_gen.interval_operator.IntervalTooShort`).
    """
    if not left_boundary_rows or not right_mirrored.boundary_rows:
        raise ValueError(
            "Both ends need at least one closure row for a round trip"
        )
    m_L = len(left_boundary_rows)
    if left_row_indices is None:
        left_row_indices = list(range(m_L))
    if len(left_row_indices) != m_L:
        raise ValueError(
            f"left_row_indices has length {len(left_row_indices)} != "
            f"number of left boundary rows ({m_L})"
        )
    t_L = _support_extent(left_boundary_rows)
    t_R = _support_extent(right_mirrored.boundary_rows)
    max_node_L = max(int(i) for i in left_row_indices)
    max_node_R = max(int(i) for i in right_mirrored.row_indices)
    return max(t_L + t_R - 1, max_node_L + max_node_R + 1)


def _check_delta_j(delta_j: int, dj_min: int) -> int:
    """Validate a requested Dj against the round-trip minimum.

    Raises the item-3.1 :class:`IntervalTooShort` (lazy import — 3.1
    imports this module) on violation, tying the two minima together.
    """
    from stencil_gen.interval_operator import IntervalTooShort

    delta_j = int(delta_j)
    if delta_j < dj_min:
        raise IntervalTooShort(
            f"Round-trip separation Dj = {delta_j} is too short: the "
            f"closure column supports (and rows) must not overlap, "
            f"requiring Dj >= {dj_min} (equivalently an interval of "
            f"N = Dj + 1 >= {dj_min + 1} nodes — stricter than the "
            f"item-3.1 assembly minimum, which permits column overlap)"
        )
    return delta_j


@dataclass(frozen=True)
class RoundTripFactors:
    """The matched factors of the round-trip cycle at one s.

    All arrays are indexed by the ORIGINAL frame's deterministic mode
    ordering (``ModeSplit.all_rightgoing`` for kappa / A_L rows / A_R
    columns; ``ModeSplit.all_leftgoing`` for mu / A_L columns / A_R
    rows): the mirrored-frame reflection matrix has been re-indexed onto
    the original mode basis via the exact reciprocal-root correspondence
    (mirror maps kappa -> 1/kappa), so the factors compose directly as
    ``Mdiag^{-Dj} A_R K^{Dj} A_L``.
    """

    A_L: np.ndarray
    """Left-end reflection matrix (p x n_leftgoing): incident leftgoing
    (mu) amplitudes -> reflected rightgoing (kappa) amplitudes, both
    referenced at grid node 0."""

    A_R: np.ndarray
    """Right-end reflection matrix (n_leftgoing x p) in the ORIGINAL mode
    ordering: incident rightgoing (kappa) amplitudes -> reflected
    leftgoing (mu) amplitudes, both referenced at the right reference
    node (grid node Dj)."""

    kappa: np.ndarray
    """Rightgoing roots (|kappa| <= 1), original deterministic order;
    K = diag(kappa)."""

    mu: np.ndarray
    """Leftgoing roots (|mu| >= 1), original deterministic order;
    Mdiag = diag(mu)."""

    s: complex
    """The Laplace frequency at which the factors were computed."""


def _match_reciprocal(
    mirrored_roots: np.ndarray,
    original_roots: np.ndarray,
    family: str,
    s: complex,
) -> list[int]:
    """Index the mirrored-frame roots by the original roots they mirror.

    Returns ``idx`` with ``mirrored_roots[idx[k]] ~ 1/original_roots[k]``
    (relative residual |z*w - 1| < _MODE_MATCH_TOL, assignment bijective).
    A failed or ambiguous match means the mode bases cannot be paired —
    route through the defective path (plan 48 items 0.1 / 2.2 / 3.3).
    """
    idx: list[int] = []
    for w in original_roots:
        residuals = np.abs(mirrored_roots * w - 1.0)
        a = int(np.argmin(residuals))
        if residuals[a] > _MODE_MATCH_TOL:
            raise DefectiveKappaError(
                f"Cannot match {family} root {w} to a reciprocal "
                f"mirrored-frame root at s={s}: best residual "
                f"{residuals[a]:.3e} > {_MODE_MATCH_TOL}"
            )
        idx.append(a)
    if len(set(idx)) != len(idx):
        raise DefectiveKappaError(
            f"Ambiguous reciprocal matching of {family} roots at s={s} "
            f"(two original roots matched the same mirrored root — "
            f"near-coalescent mode basis)"
        )
    return idx


def round_trip_factors(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    left_boundary_rows: list,
    right_mirrored: MirroredClosure,
    s: complex,
    *,
    left_row_indices: list[int] | None = None,
) -> RoundTripFactors:
    """Compute the matched round-trip factors A_L, A_R, kappa, mu at s.

    Plan 48 item 3.3 (see module docstring for the cycle math). The left
    end is analyzed directly (item 2.2); the right end is analyzed in the
    item-2.3 mirrored frame — where mirrored-rightgoing modes are the
    physically leftgoing family — and its reflection matrix is re-indexed
    onto the ORIGINAL mode ordering through the exact reciprocal-root
    correspondence of the mirror (kappa -> 1/kappa), so the returned
    factors compose directly (:func:`round_trip_matrix`).

    Parameters
    ----------
    interior_weights, interior_offsets
        Interior stencil in the ORIGINAL (left-to-right) frame.
    left_boundary_rows : list[BoundaryRow]
        Left-end closure rows (offsets >= 0, node 0 = left reference).
    right_mirrored : MirroredClosure
        The right-end closure in the mirrored analysis frame (from
        :func:`mirror_closure`; its own interior fields are used for the
        mirrored-frame solve).
    s : complex
        Laplace frequency (typically on the make_s_grid strip).
    left_row_indices : list[int], optional
        Grid node of each left closure row (default 0..m_L-1).

    Raises
    ------
    DefectiveKappaError
        Defective mode basis at s in either frame, or a failed/ambiguous
        reciprocal matching (near-coalescent roots) — route through the
        defective path, never report a clean value.
    numpy.linalg.LinAlgError
        det M_L(s) = 0 or det M_R(s) = 0: a SINGLE-boundary GKS
        eigensolution at s (Lemma 1's hypothesis is violated — the pair
        is already unstable through that end alone; the per-end tiers
        flag it).
    ValueError
        Inconsistent mode counts between the two frames (indicates a
        malformed closure, not a user error).
    """
    M_L, B_L, split_L = reflection_system(
        interior_weights, interior_offsets, left_boundary_rows, s,
        row_indices=left_row_indices,
    )
    kappa = split_L.all_rightgoing
    mu = split_L.all_leftgoing
    p = len(kappa)
    n_left = len(mu)
    n_nodal_L = M_L.shape[0] - p
    if n_left == 0:
        A_L = np.zeros((p, 0), dtype=complex)
    else:
        A_L = np.linalg.solve(M_L, -B_L)[n_nodal_L:, :]

    M_R, B_R, split_R = reflection_system(
        right_mirrored.interior_weights,
        right_mirrored.interior_offsets,
        right_mirrored.boundary_rows,
        s,
        row_indices=right_mirrored.row_indices,
    )
    mirrored_rg = split_R.all_rightgoing  # ~ 1/mu (physically leftgoing)
    mirrored_lg = split_R.all_leftgoing   # ~ 1/kappa (physically rightgoing)
    if len(mirrored_rg) != n_left or len(mirrored_lg) != p:
        raise ValueError(
            f"Mode-count mismatch between frames at s={s}: original "
            f"(rightgoing, leftgoing) = ({p}, {n_left}), mirrored = "
            f"({len(mirrored_rg)}, {len(mirrored_lg)}); the mirror maps "
            f"kappa -> 1/kappa so the counts must swap exactly"
        )
    if n_left == 0:
        A_R = np.zeros((0, p), dtype=complex)
    else:
        n_nodal_R = M_R.shape[0] - len(mirrored_rg)
        A_R_m = np.linalg.solve(M_R, -B_R)[n_nodal_R:, :]
        # Re-index the mirrored-frame matrix onto the ORIGINAL ordering:
        # rows by mu (mirrored-rightgoing ~ 1/mu), columns by kappa
        # (mirrored-leftgoing ~ 1/kappa).
        row_map = _match_reciprocal(mirrored_rg, mu, "leftgoing (mu)", s)
        col_map = _match_reciprocal(
            mirrored_lg, kappa, "rightgoing (kappa)", s
        )
        A_R = A_R_m[np.ix_(row_map, col_map)]

    return RoundTripFactors(
        A_L=A_L, A_R=A_R, kappa=kappa, mu=mu, s=complex(s)
    )


def _compose_round_trip(factors: RoundTripFactors, delta_j: int) -> np.ndarray:
    """E_Dj = Mdiag^{-Dj} A_R K^{Dj} A_L from matched factors.

    Both transit factors are contractions for Re(s) >= 0 (|kappa| <= 1,
    |mu| >= 1), so the composition is numerically safe (evanescent modes
    underflow toward 0, never overflow).
    """
    if factors.A_L.shape[1] == 0:
        return np.zeros((0, 0), dtype=complex)
    transit_right = (factors.kappa ** delta_j)[:, None] * factors.A_L
    transit_back = (factors.mu ** (-delta_j))[:, None] * factors.A_R
    return transit_back @ transit_right


def round_trip_matrix(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    left_boundary_rows: list,
    right_mirrored: MirroredClosure,
    s: complex,
    delta_j: int,
    *,
    left_row_indices: list[int] | None = None,
) -> np.ndarray:
    """The round-trip cycle matrix E_Dj(s) (Trefethen 1985 Lemma 1).

    Plan 48 item 3.3:

        E_Dj(s) = Mdiag^{-Dj} . A_R(s) . K^{Dj} . A_L(s),

    acting on the leftgoing amplitudes referenced at grid node 0; the
    right reference node is grid node Dj (= N - 1 for an N-node
    interval). The two-boundary pair supports a GKS eigensolution at s
    iff 1 is an eigenvalue of E_Dj(s) — PROVIDED each end is separately
    half-line stable at s (see module docstring; a per-end det-zero
    raises ``numpy.linalg.LinAlgError`` here and is the per-end tiers'
    verdict, not the round trip's).

    Raises
    ------
    IntervalTooShort
        If ``delta_j`` is below :func:`min_round_trip_delta_j` (closure
        supports would overlap — the modal ansatz breaks down).
    DefectiveKappaError, numpy.linalg.LinAlgError, ValueError
        Propagated from :func:`round_trip_factors`.
    """
    dj_min = min_round_trip_delta_j(
        left_boundary_rows, right_mirrored, left_row_indices=left_row_indices
    )
    delta_j = _check_delta_j(delta_j, dj_min)
    factors = round_trip_factors(
        interior_weights, interior_offsets, left_boundary_rows,
        right_mirrored, s, left_row_indices=left_row_indices,
    )
    return _compose_round_trip(factors, delta_j)


@dataclass(frozen=True)
class RoundTripResult:
    """Result of the two-boundary round-trip sweep (plan 48 item 3.3)."""

    round_trip_radius: float
    """max over the s-grid AND over the Dj values of rho(E_Dj(s)).
    Radius > 1 flags per-cycle amplification somewhere on the axis (a
    Dj-specific two-boundary resonance candidate); < 1 certifies the
    round trip PROVIDED each end is separately half-line stable (Lemma
    1's hypothesis — the per-end tiers' job). inf when a per-end
    det-zero sits on the grid; nan when no grid point was cleanly
    evaluable; 0.0 when there are no leftgoing modes (nothing to
    round-trip)."""

    argmax_s: complex | None
    """The s attaining the radius (None if no clean value)."""

    argmax_delta_j: int | None
    """The Dj attaining the radius (None if no clean value)."""

    reflection_product: float
    """Decoupled screen: max over the s-grid of ||A_L(s)|| ||A_R(s)||
    (pointwise product of spectral norms). Upper-bounds rho(E_Dj) on the
    axis for every Dj (||K^{Dj}|| = ||Mdiag^{-Dj}|| <= 1 for
    Re(s) >= 0), so product < 1 certifies the round trip for ALL Dj at
    once, while product > 1 alone cannot flag (see the item-2.5
    tension-closure control caveat in the plan's Findings log)."""

    product_argmax_s: complex | None
    """The s attaining ``reflection_product`` (None if no clean value)."""

    delta_j_min: int
    """The minimum admissible Dj (:func:`min_round_trip_delta_j`)."""

    delta_j_values: tuple
    """The Dj values swept (default {Dj_min, ..., Dj_min + 5})."""

    finite: bool
    """True iff every grid point evaluated to a finite clean value: no
    per-end det-zero (inf) and no defective mode basis anywhere on the
    grid (defective points must route through the non-clean path,
    plan 48 items 0.1 / 2.2)."""

    defective_points: tuple[complex, ...] = ()
    """Grid points with a defective mode basis or failed reciprocal
    matching; their rho / product entries are nan."""

    rho: np.ndarray | None = field(default=None, repr=False)
    """rho(E_Dj(s)) per grid point and Dj, shape (n_s, n_Dj) (nan at
    defective points, inf at per-end det-zero points)."""

    products: np.ndarray | None = field(default=None, repr=False)
    """||A_L(s)|| ||A_R(s)|| per grid point (same non-clean markers)."""

    s_grid: np.ndarray | None = field(default=None, repr=False)
    """The s-grid swept (flattened)."""


def round_trip_check(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    left_boundary_rows: list,
    right_mirrored: MirroredClosure,
    *,
    delta_j_values=None,
    n_delta_j: int = 6,
    s_grid: np.ndarray | None = None,
    left_row_indices: list[int] | None = None,
) -> RoundTripResult:
    """Sweep the round-trip radius over the imaginary-axis s-grid and Dj.

    Plan 48 item 3.3. Reports

        round_trip_radius = max over the s-grid AND over Dj of
                            rho(E_Dj(s)),

    with the default Dj range {Dj_min, ..., Dj_min + n_delta_j - 1}
    (n_delta_j = 6 per the plan) — rho(E_Dj) is NOT monotone in Dj (on
    the axis the transit factors are pure phases for propagating modes,
    so the radius oscillates with Dj: Dj-specific resonances), hence the
    max over a Dj window rather than a single length. Also reports the
    decoupled screen ``reflection_product = max_s ||A_L|| ||A_R||``,
    which upper-bounds the radius on the axis.

    Grid points with a defective mode basis are recorded in
    ``defective_points`` (rho = nan, ``finite=False``); grid points where
    an end's M(s) is exactly singular — a SINGLE-boundary GKS
    eigensolution, Lemma 1's hypothesis violated — get rho = inf (and
    ``finite=False``): both route through the non-clean path, never
    report as clean values.

    Parameters
    ----------
    interior_weights, interior_offsets
        Interior stencil in the ORIGINAL (left-to-right) frame.
    left_boundary_rows : list[BoundaryRow]
        Left-end closure rows.
    right_mirrored : MirroredClosure
        Right-end closure in the mirrored frame (:func:`mirror_closure`).
    delta_j_values : sequence of int, optional
        Explicit Dj values; every value must be >= Dj_min
        (:class:`IntervalTooShort` otherwise). Default: the plan's
        window {Dj_min, ..., Dj_min + n_delta_j - 1}.
    n_delta_j : int
        Width of the default Dj window.
    s_grid : np.ndarray, optional
        Imaginary-axis strip grid (default
        :func:`default_reflection_s_grid`).
    left_row_indices : list[int], optional
        Grid node of each left closure row (default 0..m_L-1).
    """
    dj_min = min_round_trip_delta_j(
        left_boundary_rows, right_mirrored, left_row_indices=left_row_indices
    )
    if delta_j_values is None:
        if n_delta_j < 1:
            raise ValueError(f"n_delta_j must be >= 1, got {n_delta_j}")
        delta_j_values = range(dj_min, dj_min + n_delta_j)
    dj_list = [_check_delta_j(d, dj_min) for d in delta_j_values]
    if not dj_list:
        raise ValueError("delta_j_values must contain at least one value")

    if s_grid is None:
        s_grid = default_reflection_s_grid()
    s_flat = np.asarray(s_grid, dtype=complex).ravel()

    rho = np.full((len(s_flat), len(dj_list)), np.nan)
    products = np.full(len(s_flat), np.nan)
    defective: list[complex] = []
    for i, s in enumerate(s_flat):
        try:
            factors = round_trip_factors(
                interior_weights, interior_offsets, left_boundary_rows,
                right_mirrored, s, left_row_indices=left_row_indices,
            )
        except DefectiveKappaError:
            defective.append(complex(s))
            continue
        except np.linalg.LinAlgError:
            # det M_L(s) = 0 or det M_R(s) = 0: single-boundary GKS
            # eigensolution ON the grid — the pair is unstable through
            # that end alone (Lemma 1 hypothesis violated).
            rho[i, :] = np.inf
            products[i] = np.inf
            continue
        if factors.A_L.shape[1] == 0:
            rho[i, :] = 0.0
            products[i] = 0.0
            continue
        products[i] = float(
            np.linalg.norm(factors.A_L, ord=2)
            * np.linalg.norm(factors.A_R, ord=2)
        )
        for jd, dj in enumerate(dj_list):
            E = _compose_round_trip(factors, dj)
            rho[i, jd] = float(np.max(np.abs(np.linalg.eigvals(E))))

    if np.all(np.isnan(rho)):
        return RoundTripResult(
            round_trip_radius=float("nan"),
            argmax_s=None,
            argmax_delta_j=None,
            reflection_product=float("nan"),
            product_argmax_s=None,
            delta_j_min=dj_min,
            delta_j_values=tuple(dj_list),
            finite=False,
            defective_points=tuple(defective),
            rho=rho,
            products=products,
            s_grid=s_flat,
        )

    flat_argmax = int(np.nanargmax(rho))
    i_max, j_max = np.unravel_index(flat_argmax, rho.shape)
    radius = float(rho[i_max, j_max])
    i_prod = int(np.nanargmax(products))
    finite = bool(np.isfinite(radius)) and not defective
    return RoundTripResult(
        round_trip_radius=radius,
        argmax_s=complex(s_flat[i_max]),
        argmax_delta_j=int(dj_list[j_max]),
        reflection_product=float(products[i_prod]),
        product_argmax_s=complex(s_flat[i_prod]),
        delta_j_min=dj_min,
        delta_j_values=tuple(dj_list),
        finite=finite,
        defective_points=tuple(defective),
        rho=rho,
        products=products,
        s_grid=s_flat,
    )


def cutcell_round_trip(
    scheme,
    alpha: dict,
    psi_L: float,
    psi_R: float,
    *,
    delta_j_values=None,
    n_delta_j: int = 6,
    s_grid: np.ndarray | None = None,
    feasibility_tol: float = FEASIBILITY_TOL,
):
    """Round-trip criterion for a two-ended cut-cell Dirichlet pair.

    Plan 48 item 3.3, cut-cell entry point: the left end is the
    wall-eliminated Dirichlet closure at psi_L (cached evaluator, item
    1.1); the right end is the stored left-end rows at psi_R mapped
    through the item-2.3 production mirror convention
    (``mirror_closure(..., from_physical_right=False)`` — the nu = 1
    codegen revneg entry point, exactly the right block
    ``build_interval_operator`` assembles). ``psi_* = 1.0`` means a
    uniform domain boundary at that end.

    The item-1.2 feasibility guard runs at BOTH end psi before any
    assembly: a vanishing structural denominator returns the distinct
    :class:`~stencil_gen.cutcell_normal_modes.CutCellInfeasible` marker
    (for the first failing end) — a *parameterization* failure, never a
    stability verdict.

    Returns
    -------
    RoundTripResult or CutCellInfeasible
    """
    evaluator = make_cutcell_evaluator(scheme)
    for name, psi in (("psi_L", psi_L), ("psi_R", psi_R)):
        psi = float(psi)
        if not 0.0 < psi <= 1.0:
            raise ValueError(f"{name} must be in (0, 1], got {psi}")
        vanishing = _vanishing_denominators(
            evaluator, psi, alpha, feasibility_tol
        )
        if vanishing:
            return CutCellInfeasible(
                scheme=scheme,
                psi=psi,
                alpha=tuple(
                    (sym.name, v)
                    for sym, v in zip(
                        evaluator.alpha_symbols, evaluator._alpha_args(alpha)
                    )
                ),
                vanishing=vanishing,
                feasibility_tol=feasibility_tol,
            )

    interior = evaluator.interior()
    rows_L, idx_L = evaluator.rows(float(psi_L), alpha)
    rows_R, idx_R = evaluator.rows(float(psi_R), alpha)
    mirrored = mirror_closure(
        rows_R, interior,
        from_physical_right=False,
        row_indices=[int(i) for i in idx_R],
    )
    return round_trip_check(
        interior[0], interior[1], rows_L, mirrored,
        delta_j_values=delta_j_values,
        n_delta_j=n_delta_j,
        s_grid=s_grid,
        left_row_indices=[int(i) for i in idx_L],
    )
