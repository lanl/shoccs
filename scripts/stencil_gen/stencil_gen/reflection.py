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
"""

from __future__ import annotations

from dataclasses import dataclass

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
