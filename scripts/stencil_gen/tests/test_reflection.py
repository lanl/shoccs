"""Tests for stencil_gen.reflection (plan 48 Phase 2).

Item 2.1 (mode partition): every test name contains ``mode_split`` so the
focused run ``pytest tests/test_reflection.py -k mode_split`` selects
exactly this item's tests.

Sign convention throughout: u_t = -sum_k a_k u_{n+k} (the
``gks_kreiss.kappa_roots`` docstring convention).
"""

import numpy as np
import pytest

from stencil_gen.gks_kreiss import DefectiveKappaError, _classify_imag_axis
from stencil_gen.reflection import ModeSplit, mode_split

# Centered E2 interior for u_t = -u_x: a_{-1} = -1/2, a_{+1} = 1/2.
# Characteristic polynomial (multiply s + sum a_k kappa^k = 0 by kappa):
#   kappa^2 + 2 s kappa - 1 = 0  =>  kappa = -s +/- sqrt(s^2 + 1),
# with root product -1, so the two moduli are exactly reciprocal.
E2_WEIGHTS = np.array([-0.5, 0.5])
E2_OFFSETS = np.array([-1, 1])

# Centered E4 interior (standard 4th-order central first derivative):
# p = L_left = 2, degree-4 characteristic polynomial.
E4_WEIGHTS = np.array([1.0 / 12.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 12.0])
E4_OFFSETS = np.array([-2, -1, 1, 2])


class TestModeSplitCounts:
    """Centered 2p+1 interiors at Re(s) > 0 split exactly p/p."""

    def test_mode_split_e2_counts_off_axis(self):
        """E2 (p=1) at Re(s) > 0: exactly 1 rightgoing, 1 leftgoing."""
        split = mode_split(E2_WEIGHTS, E2_OFFSETS, 0.5 + 0.3j)

        assert isinstance(split, ModeSplit)
        assert len(split.rightgoing) == 1
        assert len(split.leftgoing) == 1
        assert split.borderline == ()
        assert split.n_rightgoing == 1
        assert split.n_leftgoing == 1
        # Strict modulus partition
        assert abs(split.rightgoing[0]) < 1.0
        assert abs(split.leftgoing[0]) > 1.0

    def test_mode_split_e4_counts_off_axis(self):
        """E4 (p=2) at Re(s) > 0: exactly 2 rightgoing, 2 leftgoing."""
        split = mode_split(E4_WEIGHTS, E4_OFFSETS, 0.3 + 1.1j)

        assert len(split.rightgoing) == 2
        assert len(split.leftgoing) == 2
        assert split.borderline == ()
        assert all(abs(k) < 1.0 for k in split.rightgoing)
        assert all(abs(k) > 1.0 for k in split.leftgoing)

    def test_mode_split_e4_counts_on_strip(self):
        """E4 on the make_s_grid strip s = i*omega + eps_imag: still 2/2.

        The O(eps_imag) modulus displacement resolves the roots by the
        half-plane limit itself — no borderline classification needed.
        """
        split = mode_split(E4_WEIGHTS, E4_OFFSETS, 1e-6 + 0.2j)

        assert len(split.rightgoing) == 2
        assert len(split.leftgoing) == 2
        assert split.borderline == ()

    def test_mode_split_count_violation_raises(self):
        """A centered non-derivative stencil violating p/p raises ValueError.

        Engineered stencil (kappa - 1)(kappa - 0.5) at s = 0 (offsets
        [-1, 0, 1], weights [0.5, -1.5, 1.0]): at s = 0.01 the roots are
        ~0.9796 and ~0.5104 — BOTH strictly inside the unit circle, so a
        centered p = 1 interior would claim 2 rightgoing / 0 leftgoing.
        mode_split must surface this instead of returning a defective
        mode basis.
        """
        weights = np.array([0.5, -1.5, 1.0])
        offsets = np.array([-1, 0, 1])

        with pytest.raises(ValueError, match="rightgoing"):
            mode_split(weights, offsets, 0.01 + 0.0j)


class TestModeSplitAnalytic:
    """Hand-derivable E2 root values."""

    def test_mode_split_e2_analytic_roots(self):
        """E2 roots are kappa = -s +/- sqrt(s^2 + 1); moduli reciprocal.

        The characteristic polynomial kappa^2 + 2 s kappa - 1 = 0 has
        root product -1, so |kappa_+| |kappa_-| = 1: one root strictly
        inside, one strictly outside for Re(s) > 0.
        """
        s = 0.4 + 0.7j
        root_a = -s + np.sqrt(s**2 + 1)
        root_b = -s - np.sqrt(s**2 + 1)
        k_in, k_out = (
            (root_a, root_b) if abs(root_a) < 1 else (root_b, root_a)
        )

        split = mode_split(E2_WEIGHTS, E2_OFFSETS, s)

        assert split.rightgoing[0] == pytest.approx(k_in, rel=1e-12)
        assert split.leftgoing[0] == pytest.approx(k_out, rel=1e-12)
        # Reciprocal moduli (root product is -1)
        assert abs(split.rightgoing[0]) * abs(split.leftgoing[0]) == (
            pytest.approx(1.0, rel=1e-12)
        )

    def test_mode_split_upwind_no_leftgoing(self):
        """First-order upwind: degree-1 polynomial, zero leftgoing modes.

        u_t = -(u_n - u_{n-1}): weights [-1, 1], offsets [-1, 0], so
        kappa (1 + s) = 1 => kappa = 1/(1+s). At s = 1: kappa = 0.5 is
        the single (rightgoing) root — there is nothing to reflect
        (plan 48 item 2.2's assert-R-is-empty anchor).
        """
        weights = np.array([-1.0, 1.0])
        offsets = np.array([-1, 0])

        split = mode_split(weights, offsets, 1.0 + 0.0j)

        assert len(split.rightgoing) == 1
        assert split.rightgoing[0] == pytest.approx(0.5, rel=1e-12)
        assert len(split.leftgoing) == 0
        assert split.borderline == ()


class TestModeSplitBorderline:
    """Unit-modulus roots: strip resolution vs perturbation classification."""

    def test_mode_split_axis_strip_resolves_borderline(self):
        """On the strip s = i*omega + eps_imag the partition is strict.

        E2 at omega = 0.5 (sub-sonic, |omega| < 1): at s = i*omega
        exactly, BOTH roots are unimodular; at Re(s) = eps_imag = 1e-6
        their moduli are displaced by ~1.15e-6 — beyond the borderline
        band — so the half-plane limit resolves them without the
        perturbation test (plan 48 item 2.1).
        """
        split = mode_split(E2_WEIGHTS, E2_OFFSETS, 1e-6 + 0.5j)

        assert split.borderline == ()
        assert len(split.rightgoing) == 1
        assert len(split.leftgoing) == 1
        assert abs(split.rightgoing[0]) < 1.0 - split.tol
        assert abs(split.leftgoing[0]) > 1.0 + split.tol

    def test_mode_split_exact_axis_borderline_classified(self):
        """Exactly on the axis both E2 roots are borderline and resolved.

        E2 at s = 0.5j: kappa_+/- = +/- sqrt(3)/2 - 0.5j, both exactly
        unimodular. Implicit differentiation of
        kappa^2 + 2 s kappa - 1 = 0 gives kappa' = -kappa/(kappa + s),
        so d|kappa|^2/d Re(s) = 2 Re(conj(kappa) kappa') = -/+ 2/sqrt(0.75):
        kappa_+ (Re > 0) moves INWARD (rightgoing family), kappa_-
        (Re < 0) moves OUTWARD (leftgoing family).
        """
        c = np.sqrt(0.75)
        split = mode_split(E2_WEIGHTS, E2_OFFSETS, 0.5j)

        # Strict buckets empty; both roots in the borderline band.
        assert len(split.rightgoing) == 0
        assert len(split.leftgoing) == 0
        assert len(split.borderline) == 2

        families = {}
        for root, family in split.borderline:
            assert abs(abs(root) - 1.0) < split.tol
            families[family] = root
        assert set(families) == {"rightgoing", "leftgoing"}
        assert families["rightgoing"] == pytest.approx(c - 0.5j, rel=1e-12)
        assert families["leftgoing"] == pytest.approx(-c - 0.5j, rel=1e-12)

        # Resolved totals restore the p/p partition.
        assert split.n_rightgoing == 1
        assert split.n_leftgoing == 1
        assert split.all_rightgoing[0] == pytest.approx(c - 0.5j, rel=1e-12)
        assert split.all_leftgoing[0] == pytest.approx(-c - 0.5j, rel=1e-12)

    def test_mode_split_sonic_defective_raises(self):
        """E2 at the sonic point s = i has a double unimodular root.

        kappa^2 + 2i kappa - 1 = (kappa + i)^2: the coalescing borderline
        roots make nearest-neighbor perturbation matching ambiguous, so
        mode_split must raise DefectiveKappaError (routing through the
        item-0.1 defective path, never reporting a clean partition —
        plan 48 item 2.2 sonic-point requirement).
        """
        with pytest.raises(DefectiveKappaError):
            mode_split(E2_WEIGHTS, E2_OFFSETS, 1j)


class TestModeSplitSharedClassifier:
    """The borderline test is the SAME perturbation logic _classify_imag_axis
    uses (factored into gks_kreiss.perturbed_root_directions, plan 48 item
    2.1) — verdicts must be consistent on the classifier's own engineered
    cases (test_gks_kreiss.py TestClassifyImagAxis)."""

    def test_mode_split_agrees_with_classify_imag_axis_inward(self):
        """(kappa-1)(kappa-0.5) at s=0: kappa=1 perturbs inward.

        _classify_imag_axis calls this 'outgoing_mode_detected' (GKS
        violation); in the mode partition the same root resolves into the
        rightgoing family.
        """
        weights = np.array([0.5, -1.5, 1.0])
        offsets = np.array([-1, 0, 1])
        s = 0.0 + 0.0j

        assert _classify_imag_axis(weights, offsets, s) == (
            "outgoing_mode_detected"
        )

        split = mode_split(weights, offsets, s)
        assert len(split.borderline) == 1
        root, family = split.borderline[0]
        assert root == pytest.approx(1.0, rel=1e-12)
        assert family == "rightgoing"
        # The strictly-inside root kappa = 0.5 is rightgoing too.
        assert split.rightgoing[0] == pytest.approx(0.5, rel=1e-12)

    def test_mode_split_agrees_with_classify_imag_axis_outward(self):
        """(kappa-1)(kappa-2) at s=0: kappa=1 perturbs outward.

        _classify_imag_axis calls this 'all_incoming' (no violation); in
        the mode partition the same root resolves into the leftgoing
        family.
        """
        weights = np.array([2.0, -3.0, 1.0])
        offsets = np.array([-1, 0, 1])
        s = 0.0 + 0.0j

        assert _classify_imag_axis(weights, offsets, s) == "all_incoming"

        split = mode_split(weights, offsets, s)
        assert len(split.borderline) == 1
        root, family = split.borderline[0]
        assert root == pytest.approx(1.0, rel=1e-12)
        assert family == "leftgoing"
        # The strictly-outside root kappa = 2 is leftgoing too.
        assert split.leftgoing[0] == pytest.approx(2.0, rel=1e-12)


class TestModeSplitOrdering:
    """Deterministic bucket ordering (lexicographic by (real, imag))."""

    def test_mode_split_deterministic_ordering(self):
        """E4 buckets are sorted lexicographically by (real, imag)."""
        split = mode_split(E4_WEIGHTS, E4_OFFSETS, 0.3 + 1.1j)

        for bucket in (split.rightgoing, split.leftgoing):
            keys = [(k.real, k.imag) for k in bucket]
            assert keys == sorted(keys)
