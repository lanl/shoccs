"""Tests for stencil_gen.reflection (plan 48 Phase 2).

Item 2.1 (mode partition): every test name contains ``mode_split`` so the
focused run ``pytest tests/test_reflection.py -k mode_split`` selects
exactly this item's tests.

Item 2.2 (reflection matrix and margin): test names contain
``reflection`` and deliberately avoid the -k tokens reserved for other
items (``mode_split``, ``mirror``, ``round_trip``, ``gaussian``,
``cutcell``).

Item 2.3 (right-end closures, mirror convention): every test name
contains ``mirror`` so ``pytest tests/test_reflection.py -k mirror``
selects exactly this item's tests.

Sign convention throughout: u_t = -sum_k a_k u_{n+k} (the
``gks_kreiss.kappa_roots`` docstring convention).
"""

import numpy as np
import pytest

from stencil_gen.gks_kreiss import (
    DefectiveKappaError,
    _classify_imag_axis,
    kappa_roots,
    kreiss_matrix,
    kreiss_matrix_general,
    kreiss_stability_check,
)
from stencil_gen.reflection import (
    MirroredClosure,
    ModeSplit,
    ReflectionResult,
    default_reflection_s_grid,
    max_reflection,
    mirror_closure,
    mode_split,
    reflection_matrix,
    reflection_system,
)

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


# ---------------------------------------------------------------------------
# Item 2.2: reflection matrix and margin
# ---------------------------------------------------------------------------

# Forward-difference closure row at node 0 for u_t = -u_x:
# u_t|_0 = -(u_1 - u_0), i.e. -sum_j w_j u_j with w = [-1, 1] at {0, 1}.
E2_FWD_ROW = [(np.array([-1.0, 1.0]), np.array([0, 1]))]

# Engineered det-zero closure row (same reverse-engineering pattern as
# item 0.2's worked example, transplanted to the E2 interior because the
# upwind toy has NO leftgoing modes and hence an empty R): with m = p = 1,
#   det M(s) = s + w_0 + w_1*kappa(s),  kappa(s) = -s + sqrt(s^2 + 1),
# so w = [-sqrt(2), 1] at {0, 1} gives det M(s) = sqrt(s^2 + 1) - sqrt(2)
# exactly — a zero at s = 1 (Re > 0), a genuine GKS eigensolution.
E2_ENGINEERED_ROW = [(np.array([-np.sqrt(2.0), 1.0]), np.array([0, 1]))]

# First-order upwind interior u_t = -(u_n - u_{n-1}).
UPWIND_WEIGHTS = np.array([-1.0, 1.0])
UPWIND_OFFSETS = np.array([-1, 0])

# m = 3 augmented E2 closure (m > p = 1: two nodal unknowns phi_0, phi_1)
# for the assembled-equation-level normalization-invariance test.
E2_M3_ROWS = [
    (np.array([-1.5, 2.0, -0.5]), np.array([0, 1, 2])),
    (np.array([-0.5, 0.5]), np.array([0, 2])),
    (np.array([-0.6, 0.6]), np.array([1, 3])),
]


def _e2_kappa_mu(s: complex) -> tuple[complex, complex]:
    """Analytic E2 roots: kappa (|.| < 1, rightgoing), mu (|.| > 1)."""
    root_a = -s + np.sqrt(s**2 + 1)
    root_b = -s - np.sqrt(s**2 + 1)
    return (root_a, root_b) if abs(root_a) < abs(root_b) else (root_b, root_a)


class TestReflectionAnalyticAnchors:
    """Hand-derivable reflection matrices (plan 48 item 2.2)."""

    def test_reflection_upwind_r_is_empty(self):
        """First-order upwind: degree-1 characteristic polynomial, one
        admissible root, ZERO leftgoing modes — R is p x 0 and there is
        nothing to reflect (the plan's assert-R-is-empty anchor)."""
        rows = [(np.array([1.0]), np.array([0]))]

        R = reflection_matrix(UPWIND_WEIGHTS, UPWIND_OFFSETS, rows, 1.0 + 0.0j)

        assert R.shape == (1, 0)

    def test_reflection_e2_hand_derived(self):
        """E2 central interior + one one-sided closure row: R is 1x1.

        Hand derivation (m = p = 1, no nodal unknowns; node n_0 = 0):
        substitute phi_n = mu^n + R kappa^n into the closure equation
        s phi_0 + sum_j w_j phi_j = 0 with w = [-1, 1] at {0, 1}:

            s (1 + R) + (-1)(1 + R) + 1 (mu + R kappa) = 0
            => R (s - 1 + kappa) = -(s - 1 + mu)
            => R(s) = -(s - 1 + mu) / (s - 1 + kappa),

        with kappa = -s + sqrt(s^2 + 1) (the |kappa| < 1 branch) and
        mu = -s - sqrt(s^2 + 1).
        """
        for s in (0.5 + 0.3j, 1e-6 + 0.7j, 2.0 - 1.1j):
            kappa, mu = _e2_kappa_mu(s)
            R_hand = -(s - 1 + mu) / (s - 1 + kappa)

            R = reflection_matrix(E2_WEIGHTS, E2_OFFSETS, E2_FWD_ROW, s)

            assert R.shape == (1, 1)
            assert R[0, 0] == pytest.approx(R_hand, rel=1e-12)

    def test_reflection_e2_dirichlet_damping_row_is_minus_one(self):
        """Damping row du_0/dt = -u_0 (w = [1] at {0}) reflects R = -1.

        Hand check: s phi_0 + phi_0 = 0 => (s + 1)(1 + R) = 0 => R = -1
        for every s != -1 — frequency-independent perfect reflection.
        """
        for s in (0.5 + 0.3j, 1e-6 + 5.0j, 2.0 - 1.1j):
            rows = [(np.array([1.0]), np.array([0]))]

            R = reflection_matrix(E2_WEIGHTS, E2_OFFSETS, rows, s)

            assert R[0, 0] == pytest.approx(-1.0, abs=1e-12)


class TestReflectionSystemConsistency:
    """The assembled M(s) is item 0.2's Kreiss matrix (up to the
    deterministic alpha-column ordering, a unitary permutation that
    preserves singular values)."""

    def test_reflection_system_sigma_min_matches_kreiss_general(self):
        """Augmented m=3 > p=1: sigma_min(M_reflection) equals
        sigma_min(kreiss_matrix_general) — column permutations are
        unitary, so the singular values are identical."""
        s = 0.3 + 0.7j

        M, B, split = reflection_system(E2_WEIGHTS, E2_OFFSETS, E2_M3_ROWS, s)
        Mg = kreiss_matrix_general(E2_WEIGHTS, E2_OFFSETS, E2_M3_ROWS, s)

        assert M.shape == (3, 3)
        assert B.shape == (3, 1)
        sv = np.linalg.svd(M, compute_uv=False)
        svg = np.linalg.svd(Mg, compute_uv=False)
        np.testing.assert_allclose(sv, svg, rtol=1e-12)

    def test_reflection_system_m_equals_p_matches_kreiss_matrix(self):
        """Square m = p = 1 case: M is entrywise the classic kreiss_matrix."""
        s = 0.5 + 0.3j

        M, _, _ = reflection_system(E2_WEIGHTS, E2_OFFSETS, E2_FWD_ROW, s)
        Mk = kreiss_matrix(E2_WEIGHTS, E2_OFFSETS, E2_FWD_ROW, s)

        assert M[0, 0] == pytest.approx(Mk[0, 0], rel=1e-14)


class TestReflectionNormalizationInvariance:
    """The reason R beats sigma_min (plan 48 item 2.2): R is invariant
    under scaling an assembled *equation* while sigma_min(M) is not.

    Implemented at the assembled-system level per the plan's review
    caution: scaling only the stored BoundaryRow *weights* would NOT
    preserve R (the temporal coefficient stays 1 — the spatial parts of M
    and b scale but the s-parts do not), so the test scales row i of
    [M(s) | b_k(s)] jointly instead.
    """

    def test_reflection_invariant_under_joint_row_scaling(self):
        s = 0.3 + 0.7j
        n_nodal = 2  # m - p = 3 - 1

        M, B, _ = reflection_system(E2_WEIGHTS, E2_OFFSETS, E2_M3_ROWS, s)
        R0 = np.linalg.solve(M, -B)[n_nodal:, :]

        # Multiply row 1 of [M | b] jointly by 10: the equation set is
        # unchanged, so the solution x — and hence R — is exactly
        # unchanged.
        M_scaled = M.copy()
        B_scaled = B.copy()
        M_scaled[1, :] *= 10.0
        B_scaled[1, :] *= 10.0
        R1 = np.linalg.solve(M_scaled, -B_scaled)[n_nodal:, :]

        np.testing.assert_allclose(R1, R0, rtol=1e-12)
        # And the public entry point agrees with the by-hand solve.
        R_fun = reflection_matrix(E2_WEIGHTS, E2_OFFSETS, E2_M3_ROWS, s)
        np.testing.assert_allclose(R_fun, R0, rtol=1e-12)

        # Contrast: sigma_min(M) is NOT invariant under the same joint
        # row scaling (measured: 0.0547 -> 0.0796, a 45% change).
        sigma0 = np.linalg.svd(M, compute_uv=False)[-1]
        sigma1 = np.linalg.svd(M_scaled, compute_uv=False)[-1]
        assert abs(sigma1 - sigma0) > 0.1 * sigma0


class TestReflectionSingularityEquivalence:
    """||R(s)|| -> infinity exactly at the det M(s0) = 0 points the GKS
    check flags (plan 48 item 2.2)."""

    def test_reflection_blowup_at_engineered_det_zero(self):
        """E2_ENGINEERED_ROW: det M(s) = sqrt(s^2+1) - sqrt(2) = 0 at
        s0 = 1. Near s0, det M ~ (s - 1)/sqrt(2) and the numerator
        -(s - 1 + mu) -> 2*sqrt(2), so |R(1 + eps)| ~ 4/eps -> infinity.
        """
        norms = []
        for eps in (1e-2, 1e-3, 1e-4):
            R = reflection_matrix(
                E2_WEIGHTS, E2_OFFSETS, E2_ENGINEERED_ROW, 1.0 + eps
            )
            norms.append(abs(R[0, 0]))
            # |R| ~ 4/eps (constant derived above; 5% slack).
            assert norms[-1] == pytest.approx(4.0 / eps, rel=0.05)
        assert norms[0] < norms[1] < norms[2]

    def test_reflection_engineered_det_zero_flagged_by_gks(self):
        """The same closure is flagged unstable by kreiss_stability_check
        with the witness at s0 = 1 — R's pole and the GKS violation are
        the same point."""
        result = kreiss_stability_check(
            E2_WEIGHTS, E2_OFFSETS, E2_ENGINEERED_ROW
        )

        assert result.is_stable is False
        assert result.witness_sigma_min < 1e-8
        assert abs(result.witness_s - 1.0) < 1e-3


class TestReflectionSonicPoint:
    """Sonic points (interior roots coalescing ON the unit circle; E2 at
    s = +/- i) must route through the defective/borderline path — never
    report as clean values (plan 48 items 0.1 / 2.2)."""

    def test_reflection_exact_sonic_raises_defective(self):
        """At s = i the E2 characteristic polynomial is (kappa + i)^2:
        the mode basis is defective and reflection_matrix must raise."""
        with pytest.raises(DefectiveKappaError):
            reflection_matrix(E2_WEIGHTS, E2_OFFSETS, E2_FWD_ROW, 1j)

    def test_reflection_sonic_grid_point_flagged_not_clean(self):
        """max_reflection over a grid containing the exact sonic point
        records it in defective_points and reports finite=False — the
        defective point is observable, never a clean value — while still
        evaluating the clean points."""
        grid = np.array([1j, 1e-6 + 0.5j, 1e-6 + 2.0j])

        result = max_reflection(
            E2_WEIGHTS, E2_OFFSETS, E2_FWD_ROW, s_grid=grid
        )

        assert result.defective_points == (1j,)
        assert result.finite is False
        # The clean points were still evaluated: nan only at the sonic
        # point, and the max comes from a clean point.
        assert np.isnan(result.norms[0])
        assert np.all(np.isfinite(result.norms[1:]))
        assert np.isfinite(result.max_norm)
        assert result.argmax_s in (grid[1], grid[2])

    def test_reflection_near_sonic_amplitude_bounded(self):
        """Measured near-sonic behavior at the eps offset (documented for
        the 2.2 reviewer): the amplitude-normalized scalar R does NOT
        spike like eps^{-1/2} — it stays O(1) with |R| = 1 + 2 sqrt(eps)
        + O(eps).

        Mechanism: b_k = c(mu_k) is M's own root-column function, so as
        mu -> kappa the solve degenerates continuously to x -> -e_kappa,
        i.e. R -> -1 (the incident and reflected modes coalesce). The
        plan's eps^{-1/2} blowup requires two *rightgoing* roots to
        coalesce (a genuinely defective M), which routes through
        DefectiveKappaError instead (previous tests). The sqrt(eps)
        signature of the sonic branch point appears in the *deviation*
        |R| - 1 ~ 2 sqrt(eps), which this test pins down.
        """
        for eps in (1e-4, 1e-6, 1e-8):
            R = reflection_matrix(
                E2_WEIGHTS, E2_OFFSETS, E2_FWD_ROW, eps + 1j
            )
            r = abs(R[0, 0])
            assert r < 1.1  # bounded: no eps^{-1/2} spike
            # |R| - 1 = 2 sqrt(eps) + O(eps)
            assert r - 1.0 == pytest.approx(2.0 * np.sqrt(eps), rel=0.05)


class TestReflectionMargin:
    """max_reflection = max over the imaginary-axis s-grid of ||R(s)||_2."""

    def test_reflection_margin_damping_row_is_one(self):
        """The damping row reflects R = -1 at every frequency, so the
        margin over the default strip grid is exactly 1."""
        rows = [(np.array([1.0]), np.array([0]))]

        result = max_reflection(E2_WEIGHTS, E2_OFFSETS, rows)

        assert isinstance(result, ReflectionResult)
        assert result.max_norm == pytest.approx(1.0, rel=1e-10)
        assert result.finite is True
        assert result.defective_points == ()
        assert result.argmax_s is not None

    def test_reflection_margin_upwind_is_zero(self):
        """Upwind has no leftgoing modes: R is p x 0 everywhere, margin 0."""
        rows = [(np.array([1.0]), np.array([0]))]

        result = max_reflection(UPWIND_WEIGHTS, UPWIND_OFFSETS, rows)

        assert result.max_norm == 0.0
        assert result.finite is True

    def test_reflection_default_grid_matches_strip_convention(self):
        """The default grid is the make_s_grid strip (Re = 1e-6) with the
        near-origin structural-pole exclusion |omega| >= 1e-3 (the benign
        det-zeros of items 0.4/0.5d are poles of R at s = 0)."""
        grid = default_reflection_s_grid()

        assert np.all(grid.real == pytest.approx(1e-6, abs=0.0))
        assert np.all(np.abs(grid.imag) >= 1e-3)


# ---------------------------------------------------------------------------
# Item 2.3: right-end closures (mirror convention)
# ---------------------------------------------------------------------------

# 3-point form of the E2 central interior, matching the assembled-matrix
# layout of phs.build_diff_matrix_rbf (the zero center weight changes
# neither the characteristic polynomial nor p = L_left = 1).
E2_INTERIOR_3PT = (np.array([-0.5, 0.0, 0.5]), np.array([-1, 0, 1]))


@pytest.fixture(scope="module")
def e2_classical_closure():
    """E2 classical closure (tension sigma = 0 — the repo's classical-E2
    family; ``derive_boundary(p=1)`` has zero free alphas, so sigma = 0
    is the alpha = 0 configuration) plus the assembled D matrix whose
    right end is the nu = 1 codegen revneg image of the left rows
    (``D[n-1-i, n-1-j] = -B[i, j]``)."""
    from stencil_gen.phs import build_diff_matrix_rbf

    n, r, t = 20, 3, 4  # compute_dimensions(p=1, q=2, s=0, nextra=0, nu=1)
    D = build_diff_matrix_rbf(n=n, p=1, q=2, epsilon=0.0, kernel="tension",
                              nu=1, nextra=0)
    # Construction guard: the first non-closure row is the centered E2
    # interior stencil.
    np.testing.assert_allclose(D[r, r - 1:r + 2], [-0.5, 0.0, 0.5],
                               atol=1e-12)
    left_rows = [(D[i, :t].copy(), np.arange(t)) for i in range(r)]
    return D, left_rows, n, r, t


class TestMirrorConvention:
    """Plan 48 item 2.3: pure index reversal, NOT reverse-and-negate."""

    def test_mirror_upwind_interior_root_is_one_plus_s(self):
        """Plan worked check (a): the mirrored upwind interior's root is
        analytically 1/kappa = 1 + s.

        Upwind u_t = -(u_n - u_{n-1}) has kappa(s) = 1/(1 + s). The pure
        index reversal maps offsets [-1, 0] -> [0, 1] with weights
        unchanged, so the mirrored characteristic equation is
        s + 1 - kappa' = 0 => kappa' = 1 + s = 1/kappa. The historical
        sign-error trap, reverse-AND-negate, would instead give
        kappa' = 1/kappa(-s) = 1 - s — asserted against below.
        """
        mirrored = mirror_closure(
            [], (UPWIND_WEIGHTS, UPWIND_OFFSETS), from_physical_right=False
        )
        assert isinstance(mirrored, MirroredClosure)
        np.testing.assert_array_equal(mirrored.interior_offsets, [0, 1])
        np.testing.assert_allclose(mirrored.interior_weights, [1.0, -1.0])

        for s in (0.5 + 0.0j, 0.3 + 0.2j, 2.0 - 1.5j):
            roots, _, _ = kappa_roots(
                mirrored.interior_weights, mirrored.interior_offsets, s
            )
            assert len(roots) == 1
            kappa_left = 1.0 / (1.0 + s)
            assert roots[0] == pytest.approx(1.0 / kappa_left, rel=1e-12)
            # Revneg trap: 1/kappa(-s) = 1 - s is NOT the mirror root.
            assert abs(roots[0] - (1.0 - s)) > 0.5

    def test_mirror_interior_pure_reversal_both_entry_points(self):
        """The interior map is identical for both entry points: offsets
        negated, weights unchanged. For the antisymmetric E2 interior the
        mirror equals its own negation (a_{-k} = -a_k)."""
        for fpr in (True, False):
            mirrored = mirror_closure(
                [], E2_INTERIOR_3PT, from_physical_right=fpr
            )
            np.testing.assert_array_equal(
                mirrored.interior_offsets, [-1, 0, 1]
            )
            np.testing.assert_allclose(
                mirrored.interior_weights, [0.5, 0.0, -0.5]
            )
            np.testing.assert_allclose(
                mirrored.interior_weights, -E2_INTERIOR_3PT[0]
            )

    def test_mirror_stored_left_composite_map(self):
        """from_physical_right=False (stored left-end rows relying on the
        codegen revneg symmetry): closure weights negated, offsets and
        node indices unchanged."""
        mirrored = mirror_closure(
            E2_M3_ROWS, E2_INTERIOR_3PT, from_physical_right=False
        )

        assert mirrored.row_indices == [0, 1, 2]
        assert len(mirrored.boundary_rows) == 3
        for (mw, mo), (w, o) in zip(mirrored.boundary_rows, E2_M3_ROWS):
            np.testing.assert_allclose(mw, -w)
            np.testing.assert_array_equal(mo, o)

    def test_mirror_physical_right_pure_reversal(self):
        """from_physical_right=True: offsets and node indices negated,
        weights unchanged (reordered to ascending mirrored offsets)."""
        # One physical right-end row at the right-end node itself
        # (node 0), support 2 nodes into the domain.
        phys_rows = [(np.array([1.0, -4.0, 3.0]), np.array([-2, -1, 0]))]

        mirrored = mirror_closure(
            phys_rows, E2_INTERIOR_3PT, from_physical_right=True
        )

        assert mirrored.row_indices == [0]
        mw, mo = mirrored.boundary_rows[0]
        np.testing.assert_array_equal(mo, [0, 1, 2])
        np.testing.assert_allclose(mw, [3.0, -4.0, 1.0])

    def test_mirror_entry_points_agree_on_assembled_operator(
        self, e2_classical_closure
    ):
        """The two entry points are consistent on the repo's actual
        assembly: rows read off the BOTTOM of ``build_diff_matrix_rbf``'s
        D (physical right end, built by revneg from the left rows) fed
        through from_physical_right=True equal the stored LEFT rows fed
        through from_physical_right=False."""
        D, left_rows, n, r, t = e2_classical_closure

        phys_right = []
        for i in range(r):
            row = n - 1 - i
            weights = D[row, n - t:].copy()
            offsets = np.arange(n - t, n) - (n - 1)  # -(t-1) .. 0
            phys_right.append((weights, offsets))

        via_physical = mirror_closure(
            phys_right, E2_INTERIOR_3PT, from_physical_right=True,
            row_indices=[0, -1, -2],
        )
        via_stored = mirror_closure(
            left_rows, E2_INTERIOR_3PT, from_physical_right=False
        )

        np.testing.assert_array_equal(
            via_physical.interior_offsets, via_stored.interior_offsets
        )
        np.testing.assert_array_equal(
            via_physical.interior_weights, via_stored.interior_weights
        )
        assert via_physical.row_indices == via_stored.row_indices == [0, 1, 2]
        for (wa, oa), (wb, ob) in zip(
            via_physical.boundary_rows, via_stored.boundary_rows
        ):
            np.testing.assert_array_equal(oa, ob)
            np.testing.assert_array_equal(wa, wb)  # exact: same floats

    def test_mirror_e2_classical_norm_equality_on_axis(
        self, e2_classical_closure
    ):
        """Plan worked check (b): antisymmetric interior + symmetric
        closure (E2 classical, alpha = 0) gives ||A_R(i omega)|| =
        ||A_L(i omega)|| on the axis.

        For a real antisymmetric interior the mirrored problem is the
        original at s -> -s, so A_R(i omega) = conj(A_L(i omega))
        wherever the mode-family matching is continuous across the axis —
        the super-sonic segment |omega| > 1, where the kappa moduli split
        strictly (on the sub-sonic segment the roles swap instead; see
        the companion reciprocal test). Norms are compared per the plan;
        for p = 1 the mode basis is 1x1, canonical ordering is trivial,
        and the entrywise conjugation identity is asserted too.
        """
        _, left_rows, _, r, _ = e2_classical_closure
        iw, io = E2_INTERIOR_3PT
        row_idx = list(range(r))
        mirrored = mirror_closure(
            left_rows, E2_INTERIOR_3PT, from_physical_right=False
        )

        # Exact axis (super-sonic: strict modulus split, identity exact).
        for omega in (1.3, 1.7, 2.6, -1.3, -2.6):
            s = 1j * omega
            A_L = reflection_matrix(iw, io, left_rows, s,
                                    row_indices=row_idx)
            A_R = reflection_matrix(
                mirrored.interior_weights, mirrored.interior_offsets,
                mirrored.boundary_rows, s,
                row_indices=mirrored.row_indices,
            )
            assert A_L.shape == A_R.shape == (1, 1)
            norm_L = np.linalg.norm(A_L, 2)
            norm_R = np.linalg.norm(A_R, 2)
            assert norm_R == pytest.approx(norm_L, rel=1e-10), (
                f"norm equality fails at omega={omega}"
            )
            assert A_R[0, 0] == pytest.approx(np.conj(A_L[0, 0]), rel=1e-10)

        # Strip points (make_s_grid convention): equality to O(eps_imag).
        for omega in (1.3, 1.7, 2.6, 4.0, -1.8):
            s = 1e-6 + 1j * omega
            A_L = reflection_matrix(iw, io, left_rows, s,
                                    row_indices=row_idx)
            A_R = reflection_matrix(
                mirrored.interior_weights, mirrored.interior_offsets,
                mirrored.boundary_rows, s,
                row_indices=mirrored.row_indices,
            )
            assert np.linalg.norm(A_R, 2) == pytest.approx(
                np.linalg.norm(A_L, 2), rel=1e-4
            )

    def test_mirror_e2_classical_sub_sonic_reciprocal(
        self, e2_classical_closure
    ):
        """Measured companion to the norm-equality check (documented for
        the 2.3 reviewer): on the SUB-sonic axis segment (E2:
        |omega| < 1, both roots unimodular at eps = 0) the Re(s) -> 0+
        admissibility limits at the two ends classify the propagating
        modes from opposite sides of the sonic branch — the
        incident/reflected roles swap (time reversal) and the relation is
        the reciprocal A_R * conj(A_L) -> 1 (p = 1), NOT norm equality.
        Measured: |A_L| ~ 177 vs |A_R| ~ 0.0056 at omega = 0.3 with
        product -> 1 like O(eps)."""
        _, left_rows, _, r, _ = e2_classical_closure
        iw, io = E2_INTERIOR_3PT
        row_idx = list(range(r))
        mirrored = mirror_closure(
            left_rows, E2_INTERIOR_3PT, from_physical_right=False
        )

        for omega in (0.3, 0.7, -0.4):
            s = 1e-6 + 1j * omega
            A_L = reflection_matrix(iw, io, left_rows, s,
                                    row_indices=row_idx)
            A_R = reflection_matrix(
                mirrored.interior_weights, mirrored.interior_offsets,
                mirrored.boundary_rows, s,
                row_indices=mirrored.row_indices,
            )
            norm_L = np.linalg.norm(A_L, 2)
            norm_R = np.linalg.norm(A_R, 2)
            # Reciprocal relation, entrywise and in norm ...
            assert A_R[0, 0] * np.conj(A_L[0, 0]) == pytest.approx(
                1.0, abs=1e-4
            )
            assert norm_L * norm_R == pytest.approx(1.0, abs=1e-4)
            # ... and emphatically NOT norm equality.
            assert norm_L / norm_R > 10.0

    def test_mirror_validation_rejects_wrong_conventions(self):
        """Sign/shape validation of the two entry points."""
        left_like = [(np.array([-1.0, 1.0]), np.array([0, 1]))]
        right_like = [(np.array([-1.0, 1.0]), np.array([-1, 0]))]

        # Physical-right entry: offsets measured from the right-end node
        # must be <= 0, node indices <= 0.
        with pytest.raises(ValueError, match="positive column offsets"):
            mirror_closure(left_like, E2_INTERIOR_3PT,
                           from_physical_right=True)
        with pytest.raises(ValueError, match="positive node index"):
            mirror_closure(right_like, E2_INTERIOR_3PT,
                           from_physical_right=True, row_indices=[1])

        # Stored-left entry: offsets and node indices must be >= 0.
        with pytest.raises(ValueError, match="negative column offsets"):
            mirror_closure(right_like, E2_INTERIOR_3PT,
                           from_physical_right=False)
        with pytest.raises(ValueError, match="negative node index"):
            mirror_closure(left_like, E2_INTERIOR_3PT,
                           from_physical_right=False, row_indices=[-1])

        # Row-index count must match the row count.
        with pytest.raises(ValueError, match="row_indices"):
            mirror_closure(left_like, E2_INTERIOR_3PT,
                           from_physical_right=False, row_indices=[0, 1])
