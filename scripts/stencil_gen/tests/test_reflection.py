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

Item 2.4 (psi-swept reflection margin + report wiring): every test name
contains ``cutcell`` (the plan's naming rule for this item) and avoids
the other reserved tokens.

Item 2.5 (Gaussian eps=0.1 mechanism calibration): every test name
contains ``gaussian`` so ``pytest tests/test_reflection.py -k gaussian``
selects exactly this item's tests.

Item 3.3 (two-boundary round-trip criterion, Trefethen 1985 Lemma 1):
every test name contains ``round_trip`` so ``-k round_trip`` selects
exactly this item's tests; they deliberately avoid the tokens reserved
by the other items (``mode_split``, ``mirror``, ``cutcell``,
``gaussian`` — the wall-eliminated eps=0.1 pair tests spell it
``wall_eliminated`` for that reason).

Sign convention throughout: u_t = -sum_k a_k u_{n+k} (the
``gks_kreiss.kappa_roots`` docstring convention).
"""

import numpy as np
import pytest
from sympy import Rational, Symbol

from stencil_gen.brady2d_stability import (
    StabilityReport,
    cutcell_stability_report,
)
from stencil_gen.cutcell_eval import make_cutcell_evaluator
from stencil_gen.cutcell_normal_modes import (
    CutCellInfeasible,
    CutCellPsiSweep,
    default_psi_grid,
)
from stencil_gen.gks_kreiss import (
    DefectiveKappaError,
    _classify_imag_axis,
    kappa_roots,
    kreiss_matrix,
    kreiss_matrix_general,
    kreiss_stability_check,
    min_singular_value_ex,
)
from stencil_gen.interval_operator import (
    IntervalTooShort,
    build_interval_operator,
)
from stencil_gen.reflection import (
    CutCellReflectionSweep,
    MirroredClosure,
    ModeSplit,
    ReflectionResult,
    RoundTripFactors,
    RoundTripResult,
    cutcell_round_trip,
    default_reflection_s_grid,
    max_reflection,
    min_round_trip_delta_j,
    mirror_closure,
    mode_split,
    reflection_matrix,
    reflection_over_psi,
    reflection_system,
    round_trip_check,
    round_trip_factors,
    round_trip_matrix,
)
from stencil_gen.temo import E2_1

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


# ---------------------------------------------------------------------------
# Item 2.4: psi-swept reflection margin + report wiring
# ---------------------------------------------------------------------------

# Reduced imaginary-axis strip for the psi sweeps: same construction as
# the default grid, smaller so 20-psi sweeps stay fast.
_CUTCELL_S_GRID = default_reflection_s_grid(imag_max=5.0, n_points=61)


@pytest.fixture(scope="module")
def e2_1_evaluator():
    """Module-cached E2_1 evaluator (lru_cache makes this near-free)."""
    return make_cutcell_evaluator(E2_1)


@pytest.fixture()
def e2_1_alpha_zero(e2_1_evaluator):
    """The plan-35 baseline configuration: all four E2_1 alphas zero."""
    return {s: 0.0 for s in e2_1_evaluator.alpha_symbols}


class TestCutcellReflectionSweep:
    """Plan 48 item 2.4: reflection_over_psi.

    Every test name contains ``cutcell`` (the plan's naming rule for this
    item's sweep/report tests).
    """

    def test_cutcell_sweep_e2_1_alpha0_default_psi_grid(self, e2_1_alpha_zero):
        """E2_1 alpha = 0 across the full default psi grid: every psi is
        feasible and cleanly evaluable, and the worst-case bookkeeping is
        consistent with the per-psi results."""
        sweep = reflection_over_psi(E2_1, e2_1_alpha_zero,
                                    s_grid=_CUTCELL_S_GRID)

        # The item-1.3 sweep container is reused (plan 48 item 2.4).
        assert isinstance(sweep, CutCellPsiSweep)
        assert isinstance(sweep, CutCellReflectionSweep)
        assert sweep.side == "left"

        # Default grid = item 1.3's default_psi_grid, keyed by float(psi).
        assert sorted(sweep.per_psi) == [float(p) for p in default_psi_grid()]
        assert sweep.infeasible_psi == []
        assert sweep.any_unstable is False

        for psi, res in sweep.per_psi.items():
            assert isinstance(res, ReflectionResult), f"psi={psi}"
            assert res.finite is True, f"non-clean sweep at psi={psi}"
            assert res.defective_points == ()

        # max_reflection / argmax_psi / argmax_s are the max (and argmax)
        # over feasible psi of the per-psi margins.
        per_psi_max = {p: r.max_norm for p, r in sweep.per_psi.items()}
        assert np.isfinite(sweep.max_reflection)
        assert sweep.max_reflection > 0.0
        assert sweep.max_reflection == max(per_psi_max.values())
        best = sweep.per_psi[sweep.argmax_psi]
        assert best.max_norm == sweep.max_reflection
        assert best.argmax_s == sweep.argmax_s

        # The inherited Kreiss-margin fields are NOT computed by the
        # reflection sweep: base-class no-finite-margin convention.
        assert np.isinf(sweep.worst_sigma_min)
        assert sweep.worst_psi is None

    def test_cutcell_sweep_rows_depend_on_psi(self, e2_1_alpha_zero):
        """The swept closures genuinely differ across psi: the per-psi
        norm profiles are not all identical (guards against sweeping a
        fixed closure by accident)."""
        sweep = reflection_over_psi(E2_1, e2_1_alpha_zero,
                                    psi_grid=[0.1, 0.9],
                                    s_grid=_CUTCELL_S_GRID)
        norms_a = sweep.per_psi[0.1].norms
        norms_b = sweep.per_psi[0.9].norms
        assert np.nanmax(np.abs(norms_a - norms_b)) > 0.0

    def test_cutcell_sweep_side_right_matches_revneg_frame(
        self, e2_1_evaluator, e2_1_alpha_zero
    ):
        """side="right" analyzes the stored left-end rows through the
        production revneg entry point (mirror_closure with
        from_physical_right=False) — cross-checked against a by-hand
        application of that map at one psi."""
        psi = 0.5
        sweep = reflection_over_psi(E2_1, e2_1_alpha_zero, psi_grid=[psi],
                                    side="right", s_grid=_CUTCELL_S_GRID)
        assert isinstance(sweep, CutCellReflectionSweep)
        assert sweep.side == "right"
        res = sweep.per_psi[psi]
        assert res.finite is True

        rows, row_indices = e2_1_evaluator.rows(psi, e2_1_alpha_zero)
        mc = mirror_closure(rows, e2_1_evaluator.interior(),
                            from_physical_right=False,
                            row_indices=[int(i) for i in row_indices])
        expected = max_reflection(mc.interior_weights, mc.interior_offsets,
                                  mc.boundary_rows,
                                  row_indices=mc.row_indices,
                                  s_grid=_CUTCELL_S_GRID)
        assert res.max_norm == expected.max_norm
        assert res.argmax_s == expected.argmax_s
        np.testing.assert_array_equal(res.norms, expected.norms)

    def test_cutcell_sweep_invalid_side_rejected(self, e2_1_alpha_zero):
        with pytest.raises(ValueError, match="side"):
            reflection_over_psi(E2_1, e2_1_alpha_zero, psi_grid=[0.5],
                                side="top")

    def test_cutcell_sweep_psi_out_of_range_rejected(self, e2_1_alpha_zero):
        for bad in (0.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="psi"):
                reflection_over_psi(E2_1, e2_1_alpha_zero, psi_grid=[bad],
                                    s_grid=_CUTCELL_S_GRID)

    def test_cutcell_sweep_empty_psi_grid_rejected(self, e2_1_alpha_zero):
        with pytest.raises(ValueError, match="psi_grid"):
            reflection_over_psi(E2_1, e2_1_alpha_zero, psi_grid=[])

    def test_cutcell_sweep_partitions_infeasible(
        self, e2_1_evaluator, e2_1_alpha_zero, monkeypatch
    ):
        """Infeasible psi points (item 1.2 guard, reused verbatim) are
        listed in infeasible_psi, carried as CutCellInfeasible in per_psi,
        and excluded from the margin — a parameterization failure never
        folds into max_reflection.

        E2_1's only structural denominator is psi + 1, which never
        vanishes on (0, 1], so an engineered denominator psi - 1/2 is
        substituted on the cached evaluator (monkeypatch restores it)."""
        monkeypatch.setattr(
            e2_1_evaluator, "denominators",
            frozenset({Symbol("psi") - Rational(1, 2)}),
        )
        sweep = reflection_over_psi(E2_1, e2_1_alpha_zero,
                                    psi_grid=[0.25, 0.5, 0.75],
                                    s_grid=_CUTCELL_S_GRID)
        assert sweep.infeasible_psi == [0.5]
        marker = sweep.per_psi[0.5]
        assert isinstance(marker, CutCellInfeasible)
        assert marker.infeasible is True
        assert not hasattr(marker, "is_stable")
        assert isinstance(sweep.per_psi[0.25], ReflectionResult)
        assert isinstance(sweep.per_psi[0.75], ReflectionResult)
        assert sweep.argmax_psi in (0.25, 0.75)
        assert np.isfinite(sweep.max_reflection)
        assert sweep.any_unstable is False

    def test_cutcell_sweep_all_infeasible(
        self, e2_1_evaluator, e2_1_alpha_zero, monkeypatch
    ):
        """All-infeasible sweep: no margin (nan / None), not 'unstable'."""
        monkeypatch.setattr(
            e2_1_evaluator, "denominators",
            frozenset({Symbol("psi") - Rational(1, 2)}),
        )
        sweep = reflection_over_psi(E2_1, e2_1_alpha_zero, psi_grid=[0.5],
                                    s_grid=_CUTCELL_S_GRID)
        assert sweep.infeasible_psi == [0.5]
        assert np.isnan(sweep.max_reflection)
        assert sweep.argmax_psi is None
        assert sweep.argmax_s is None
        assert sweep.any_unstable is False


class TestCutcellStabilityReport:
    """Plan 48 items 2.4 + 3.4: the opt-in layer_cutcell report wiring."""

    def test_cutcell_stability_report_populates_layer_cutcell(
        self, e2_1_alpha_zero
    ):
        """cutcell_stability_report fills exactly the plan-specified keys
        from the item-1.3 Kreiss sweep, the item-2.4 reflection sweep, and
        (item 3.4) the item-3.3 round-trip and item-3.2 interval tiers."""
        psi_grid = [0.5, 0.9]
        report = cutcell_stability_report(E2_1, e2_1_alpha_zero,
                                          psi_grid=psi_grid)
        assert isinstance(report, StabilityReport)
        lc = report.layer_cutcell
        assert lc is not None
        assert set(lc) == {
            "worst_kreiss_sigma",
            "worst_psi",
            "max_reflection",
            "argmax_psi",
            "reflection_product",
            "round_trip_radius",
            "interval_worst_abscissa",
            "infeasible_psi",
        }
        # Kreiss margin (item 1.3): finite positive witness sigma at one
        # of the swept psi (E2_1 alpha = 0 is the stable baseline).
        assert 0.0 < lc["worst_kreiss_sigma"] < np.inf
        assert lc["worst_psi"] in psi_grid
        # Reflection margin (item 2.4).
        assert np.isfinite(lc["max_reflection"])
        assert lc["max_reflection"] > 0.0
        assert lc["argmax_psi"] in psi_grid
        # Two-boundary tiers (item 3.4 wiring), worst-cased over the
        # default pair thinning of psi_grid x itself: here the 4 corner
        # pairs {0.5, 0.9}^2.  The axis round-trip radius stays below 1
        # for every pair (the marginally-stable near-origin consistency
        # physics; measured 1 - 6.1e-6) and the decoupled product screen
        # upper-bounds it.
        assert np.isfinite(lc["reflection_product"])
        assert lc["reflection_product"] > 0.0
        assert 0.99 < lc["round_trip_radius"] < 1.0
        assert lc["reflection_product"] >= lc["round_trip_radius"]
        # Interval tier: the scan catches the genuine cut-right (outflow)
        # eigenvalue at exactly (1 - psi_R)/(1 + psi_R) — worst pair here
        # has psi_R = 0.5, so the abscissa is 1/3 (item-3.2 outflow law;
        # N-independent) — the tier the axis radius correctly does NOT
        # flag (per-end det-zero off the axis, Lemma 1 hypothesis).
        assert lc["interval_worst_abscissa"] == pytest.approx(
            1.0 / 3.0, rel=1e-6
        )
        assert lc["infeasible_psi"] == []
        assert report.compute_time > 0.0

    def test_cutcell_stability_report_outside_cascade(self, e2_1_alpha_zero):
        """The report is opt-in: no L1-L8 layer runs, the verdict stays
        "unknown" (objective/alias wiring is plan 49), and an empty
        StabilityReport leaves layer_cutcell None by default."""
        report = cutcell_stability_report(E2_1, e2_1_alpha_zero,
                                          psi_grid=[0.5])
        for i in range(1, 9):
            assert getattr(report, f"layer{i}") is None
        assert report.layer_bl42 is None
        assert report.kreiss is None
        assert report.non_normality is None
        assert report.overall_verdict == "unknown"
        assert report.failed_layer is None

        assert StabilityReport.empty().layer_cutcell is None

    def test_cutcell_stability_report_partitions_infeasible(
        self, e2_1_evaluator, e2_1_alpha_zero, monkeypatch
    ):
        """Infeasible psi points surface in layer_cutcell["infeasible_psi"]
        (union over both sweeps), never as a margin."""
        monkeypatch.setattr(
            e2_1_evaluator, "denominators",
            frozenset({Symbol("psi") - Rational(1, 2)}),
        )
        report = cutcell_stability_report(E2_1, e2_1_alpha_zero,
                                          psi_grid=[0.5, 0.9])
        lc = report.layer_cutcell
        assert lc["infeasible_psi"] == [0.5]
        assert lc["worst_psi"] == 0.9
        assert lc["argmax_psi"] == 0.9
        # The two-boundary tiers exclude every pair with an infeasible
        # end up front (item 3.4): only (0.9, 0.9) survives, so the
        # interval abscissa is the psi_R = 0.9 outflow eigenvalue
        # (1 - 0.9)/(1 + 0.9) = 1/19 and the round-trip margins are
        # computed, not nan.
        assert lc["interval_worst_abscissa"] == pytest.approx(
            1.0 / 19.0, rel=1e-6
        )
        assert 0.99 < lc["round_trip_radius"] < 1.0
        assert np.isfinite(lc["reflection_product"])

    def test_cutcell_stability_report_all_infeasible(
        self, e2_1_evaluator, e2_1_alpha_zero, monkeypatch
    ):
        """All psi infeasible: every margin carries its no-clean-value
        sentinel (item-1.3 inf/None convention for the Kreiss margin, nan
        elsewhere) and nothing is ever conflated with a verdict."""
        monkeypatch.setattr(
            e2_1_evaluator, "denominators",
            frozenset({Symbol("psi") - Rational(1, 2)}),
        )
        report = cutcell_stability_report(E2_1, e2_1_alpha_zero,
                                          psi_grid=[0.5])
        lc = report.layer_cutcell
        assert lc["infeasible_psi"] == [0.5]
        assert lc["worst_kreiss_sigma"] == np.inf
        assert lc["worst_psi"] is None
        assert np.isnan(lc["max_reflection"])
        assert lc["argmax_psi"] is None
        assert np.isnan(lc["reflection_product"])
        assert np.isnan(lc["round_trip_radius"])
        assert np.isnan(lc["interval_worst_abscissa"])

    def test_cutcell_stability_report_explicit_pairs(self, e2_1_alpha_zero):
        """Explicit psi_pairs / N_values are honored (item 3.4): the
        stable cut-left + uniform-right configuration reports clean
        margins — radius < 1 and a tiny interval abscissa (the item-1.2
        half-line result carried to the finite interval)."""
        report = cutcell_stability_report(
            E2_1, e2_1_alpha_zero,
            psi_grid=[0.5],
            psi_pairs=[(0.5, 1.0)],
            N_values=(8,),
        )
        lc = report.layer_cutcell
        assert lc["infeasible_psi"] == []
        assert 0.99 < lc["round_trip_radius"] < 1.0
        assert lc["reflection_product"] >= lc["round_trip_radius"]
        assert lc["interval_worst_abscissa"] < 1e-6


# ---------------------------------------------------------------------------
# Item 2.5: calibration — the Gaussian eps = 0.1 case gets a mechanism
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gaussian_eps01_closure():
    """The repo's documented GKS-passes-but-eigenvalue-unstable case:
    uniform Gaussian eps = 0.1 (``build_diff_matrix_rbf``, n=20, p=2,
    q=3, nu=1, nextra=0; see test_gks_kreiss.py
    TestKreissIntegration.test_gaussian_known_unstable_passes_gks and the
    plan 48 item 0.3 Findings-log recalibration of this exact closure).

    Returns the assembled D, the 5-point interior read off the middle
    row, and the m = 4 stored left-end closure rows (support columns
    0..5) — the same all-differing-rows block item 0.3 feeds to the
    augmented Kreiss matrix. The right end of D is the nu = 1 codegen
    revneg image of these rows (D[n-1-i, n-1-j] = -B[i, j]), so A_R is
    obtained via ``mirror_closure(..., from_physical_right=False)``
    (item 2.3's production entry point).
    """
    from stencil_gen.phs import build_diff_matrix_rbf

    n, m, t = 20, 4, 6
    D = build_diff_matrix_rbf(n=n, p=2, q=3, epsilon=0.1,
                              kernel="gaussian", nu=1, nextra=0)
    io = np.arange(-2, 3)
    iw = D[n // 2, n // 2 + io].copy()
    # Construction guards, mirroring _extract_stencil_data_from_D's
    # outcome for this closure (item 0.3): rows 0..3 differ from the
    # shifted interior row, row 4 is the first interior row, and the
    # closure support ends at column t - 1 = 5.
    assert np.max(np.abs(D[m, m + io] - iw)) < 1e-10
    assert np.max(np.abs(D[m - 1, (m - 1) + io] - iw)) > 1e-6
    assert np.max(np.abs(D[:m, t:])) < 1e-15
    left_rows = [(D[i, :t].copy(), np.arange(t)) for i in range(m)]
    return D, iw, io, left_rows


@pytest.fixture(scope="module")
def gaussian_eps01_reflection(gaussian_eps01_closure):
    """Per-end reflection margins A_L / A_R for the Gaussian eps = 0.1
    closure on the shared default strip grid (plan 48 item 2.5)."""
    _, iw, io, left_rows = gaussian_eps01_closure
    row_idx = list(range(len(left_rows)))

    res_L = max_reflection(iw, io, left_rows, row_indices=row_idx)

    mirrored = mirror_closure(
        left_rows, (iw, io), from_physical_right=False, row_indices=row_idx
    )
    res_R = max_reflection(
        mirrored.interior_weights, mirrored.interior_offsets,
        mirrored.boundary_rows, row_indices=mirrored.row_indices,
    )
    return res_L, res_R


class TestGaussianEps01Mechanism:
    """Plan 48 item 2.5: the two-boundary mechanism for Gaussian eps=0.1.

    The closure passes the single-boundary GKS test at BOTH ends (left:
    test_gks_kreiss.py test_gaussian_known_unstable_passes_gks; right:
    the mirrored problem of the antisymmetric interior is the original at
    s -> -s and checks GKS-stable, witness sigma ~ 2.5e-2 at the E4 sonic
    frequency s ~ 1.3722j) yet is eigenvalue-unstable on the finite
    domain (max Re eig(-D_bc) ~ +9.83e-2, the documented L3 catch). Item
    2.5 attributes the finite-domain instability to the two-boundary
    round-trip: each end has a finite reflection margin, but the
    decoupled product ||A_L(s)|| * ||A_R(s)|| exceeds 1 on the imaginary
    axis, so per-bounce amplification cannot be ruled out — the full
    Delta-j-phased criterion rho(E_Dj) = 1 arrives in item 3.3.

    Measured (recorded in the plan 48 Findings log): max ||A_L|| ~ 3.7e7
    at s = 1e-6 - 0.1667j (the grid point nearest the excluded origin —
    the tail of the benign structural consistency pole at s = 0 plus the
    near det-zero at s ~ 0.1j documented in the item 0.3 findings);
    max ||A_R|| ~ 8.9e6 at the grid edge |omega| = 20 (evanescent-mode
    normalization growth ~ |mu|^j — cancelled in item 3.3 by the transit
    factors K^Dj / Mdiag^-Dj); pointwise product > 1 at ALL 240 default
    grid points, max ~ 1.2e12 inside the propagating band
    |omega| <= 1.3722, mid-band product ~ 1.1e10 at omega = 1.
    """

    def test_gaussian_each_end_max_norm_finite(
        self, gaussian_eps01_reflection
    ):
        """Each end individually is GKS-stable in the reflection sense:
        finite ||R|| on the whole strip grid — no det M(s) = 0 blowup and
        no defective grid points at either end (plan 48 item 2.5)."""
        res_L, res_R = gaussian_eps01_reflection

        for res, end in ((res_L, "left"), (res_R, "right")):
            assert isinstance(res, ReflectionResult)
            assert res.finite is True, f"{end} end not cleanly finite"
            assert res.defective_points == ()
            assert np.isfinite(res.max_norm)
            assert res.max_norm > 0.0
            assert res.argmax_s is not None

    def test_gaussian_two_boundary_product_exceeds_one(
        self, gaussian_eps01_closure, gaussian_eps01_reflection
    ):
        """The two-boundary mechanism: the pointwise reflection product
        ||A_L(s)|| * ||A_R(s)|| exceeds 1 on part of the imaginary axis —
        including strictly inside the propagating band |omega| <=
        omega_sonic, away from both the near-origin structural pole tail
        and the evanescent grid edge — while the dense eigenvalues
        confirm the finite-domain instability this explains. Had the
        product NOT exceeded 1, the round-trip explanation would be
        falsified (plan 48 item 2.5's control direction)."""
        D, iw, io, _ = gaussian_eps01_closure
        res_L, res_R = gaussian_eps01_reflection

        # Pointwise product on the SHARED default grid (a product of
        # maxima at different s would not bound any round-trip).
        np.testing.assert_array_equal(res_L.s_grid, res_R.s_grid)
        prod = res_L.norms * res_R.norms
        assert np.nanmax(prod) > 1.0

        # Propagating band: |omega| <= omega_sonic where the interior
        # supports unimodular modes (E4: omega_sonic ~ 1.3722), computed
        # from the interior symbol s(theta) = -sum_k a_k e^{i k theta}.
        theta = np.linspace(0.0, 2.0 * np.pi, 2001)
        symbol = -(
            np.asarray(iw, dtype=complex)
            @ np.exp(1j * np.outer(io, theta))
        )
        omega_sonic = float(np.max(np.abs(symbol.imag)))
        assert omega_sonic == pytest.approx(1.3722, abs=1e-3)

        band = np.abs(res_L.s_grid.imag) <= omega_sonic
        assert np.any(band)
        assert np.nanmax(prod[band]) > 1.0
        # Mid-band, far from the origin pole tail and the grid edge.
        i_mid = int(np.argmin(np.abs(res_L.s_grid.imag - 1.0)))
        assert prod[i_mid] > 1.0

        # Context anchor: the finite-domain eigenvalue instability that
        # this mechanism explains (the documented L3 catch).
        from stencil_gen.phs import stability_eigenvalue_from_matrix

        assert stability_eigenvalue_from_matrix(D) > 0.0


# ---------------------------------------------------------------------------
# Item 3.3: two-boundary round-trip criterion (Trefethen 1985 Lemma 1)
# ---------------------------------------------------------------------------

# Reduced strip grid for the round-trip sweeps (same construction as the
# default grid; 60 points keep the psi-pair sweeps fast).
_RT_S_GRID = default_reflection_s_grid(imag_max=5.0, n_points=61)


def _wall_eliminated_pair(n: int):
    """The wall-eliminated Gaussian eps=0.1 two-boundary pair at size n.

    ``phs.stability_eigenvalue_from_matrix`` — the documented L3 catch for
    this closure — eigendecomposes D_bc = D[1:, 1:] (inflow row/column
    removed), NOT the raw D. D_bc is an (n-1)-node interval whose LEFT
    closure is the wall-eliminated block: rows 1..3 of D on columns 1..5
    (m_L = 3 > p = 2, support t_L = 5), renumbered to grid nodes 0..2.
    The RIGHT end is untouched by the elimination: the nu = 1 codegen
    revneg image of the stored left rows (m_R = 4, t_R = 6), mapped by
    the item-2.3 production entry point.

    Returns (D_bc, interior_weights, interior_offsets, left_rows,
    left_row_indices, right_mirrored).
    """
    from stencil_gen.phs import build_diff_matrix_rbf

    m, t = 4, 6
    D = build_diff_matrix_rbf(n=n, p=2, q=3, epsilon=0.1,
                              kernel="gaussian", nu=1, nextra=0)
    io = np.arange(-2, 3)
    iw = D[n // 2, n // 2 + io].copy()
    left_rows = [(D[i, 1:t].copy(), np.arange(t - 1)) for i in range(1, m)]
    stored_left = [(D[i, :t].copy(), np.arange(t)) for i in range(m)]
    mirrored = mirror_closure(stored_left, (iw, io),
                              from_physical_right=False,
                              row_indices=list(range(m)))
    return D[1:, 1:], iw, io, left_rows, [0, 1, 2], mirrored


@pytest.fixture(scope="module")
def wall_eliminated_pair_n20():
    """n = 20: D_bc is the 19-node interval the L3 catch runs on."""
    return _wall_eliminated_pair(20)


@pytest.fixture(scope="module")
def wall_eliminated_pair_n12():
    """n = 12: D_bc is an 11-node interval at the round-trip minimum
    Dj = 10 = Dj_min — small enough that the evanescent mode pair still
    contributes through the transit factors, which is what gives the
    p = 2 ordering tripwire its teeth (at Dj = 18 the evanescent
    contributions underflow to ~1e-27 and E is effectively scalar)."""
    return _wall_eliminated_pair(12)


class TestRoundTripComposition:
    """E_Dj = Mdiag^{-Dj} A_R K^{Dj} A_L: composition and factor contracts."""

    def test_round_trip_scalar_composition_matches_factors(
        self, e2_1_evaluator, e2_1_alpha_zero
    ):
        """p = 1 (E2_1, psi_L = 0.5, psi_R = 0.9): E is 1x1 and equals the
        by-hand composition mu^{-Dj} * A_R * kappa^{Dj} * A_L assembled
        from the PUBLIC pieces (mode_split + reflection_matrix on each
        end) — pinning the transit-factor placement and the mode
        correspondence against independently computed factors."""
        interior = e2_1_evaluator.interior()
        rows_L, idx_L = e2_1_evaluator.rows(0.5, e2_1_alpha_zero)
        rows_R, idx_R = e2_1_evaluator.rows(0.9, e2_1_alpha_zero)
        mirrored = mirror_closure(rows_R, interior,
                                  from_physical_right=False,
                                  row_indices=[int(i) for i in idx_R])
        row_idx = [int(i) for i in idx_L]
        dj = 9

        for s in (1e-6 + 0.5j, 1e-6 + 2.0j, 0.3 + 0.7j):
            split = mode_split(interior[0], interior[1], s)
            kappa = split.all_rightgoing[0]
            mu = split.all_leftgoing[0]
            A_L = reflection_matrix(interior[0], interior[1], rows_L, s,
                                    row_indices=row_idx)[0, 0]
            A_R = reflection_matrix(
                mirrored.interior_weights, mirrored.interior_offsets,
                mirrored.boundary_rows, s,
                row_indices=mirrored.row_indices,
            )[0, 0]
            E_hand = mu ** (-dj) * A_R * kappa ** dj * A_L

            E = round_trip_matrix(interior[0], interior[1], rows_L,
                                  mirrored, s, dj, left_row_indices=row_idx)

            assert E.shape == (1, 1)
            assert E[0, 0] == pytest.approx(E_hand, rel=1e-12)

    def test_round_trip_factors_contract(self, wall_eliminated_pair_n20):
        """RoundTripFactors carries the matched factors in the ORIGINAL
        mode ordering: A_L is p x n_leftgoing, A_R is n_leftgoing x p,
        |kappa| <= 1 <= |mu| on the strip (Re(s) > 0)."""
        _, iw, io, left_rows, left_idx, mirrored = wall_eliminated_pair_n20
        s = 1e-6 + 0.9j

        f = round_trip_factors(iw, io, left_rows, mirrored, s,
                               left_row_indices=left_idx)

        assert isinstance(f, RoundTripFactors)
        assert f.s == s
        assert f.A_L.shape == (2, 2)
        assert f.A_R.shape == (2, 2)
        assert len(f.kappa) == len(f.mu) == 2
        assert np.all(np.abs(f.kappa) < 1.0)
        assert np.all(np.abs(f.mu) > 1.0)

    def test_round_trip_cyclic_rotations_share_spectrum(
        self, wall_eliminated_pair_n20
    ):
        """Any cyclic rotation of the cycle is similar to E_Dj (same
        spectrum) — while the original draft's NON-cyclic reordering
        Mdiag^{-Dj} A_L K^{Dj} A_R has a different spectrum (the plan 48
        item 3.3 review correction)."""
        _, iw, io, left_rows, left_idx, mirrored = wall_eliminated_pair_n20
        s = 1e-6 + 0.9j
        dj = 12

        f = round_trip_factors(iw, io, left_rows, mirrored, s,
                               left_row_indices=left_idx)
        K = np.diag(f.kappa ** dj)
        Mi = np.diag(f.mu ** (-float(dj)))
        E = round_trip_matrix(iw, io, left_rows, mirrored, s, dj,
                              left_row_indices=left_idx)
        np.testing.assert_allclose(E, Mi @ f.A_R @ K @ f.A_L, rtol=1e-12)

        spec = np.sort_complex(np.linalg.eigvals(E))
        rotations = (
            f.A_R @ K @ f.A_L @ Mi,
            K @ f.A_L @ Mi @ f.A_R,
            f.A_L @ Mi @ f.A_R @ K,
        )
        for rot in rotations:
            np.testing.assert_allclose(
                np.sort_complex(np.linalg.eigvals(rot)), spec,
                rtol=0, atol=1e-12,
            )

        # The draft-wrong ordering is NOT a cyclic rotation: its spectrum
        # differs grossly at this generic s (measured max diff ~ 2.9e7).
        E_draft = Mi @ f.A_L @ K @ f.A_R
        spec_draft = np.sort_complex(np.linalg.eigvals(E_draft))
        assert np.max(np.abs(spec_draft - spec)) > 1.0


class TestRoundTripIntervalIdentity:
    """det(sI + D_interval) = 0 <=> 1 in spec(E_{N-1}(s)) (Lemma 1),
    exercised at exact dense interval eigenvalues — this pins Dj = N - 1
    (reference nodes 0 and N-1; an off-by-one rephases E and fails)."""

    def test_round_trip_identity_at_interval_eigenvalues(
        self, e2_1_evaluator, e2_1_alpha_zero
    ):
        """E2_1 alpha = 0, (psi_L, psi_R) = (0.5, 1.0), N = 12 (the
        dense-stable cut-left + uniform-right configuration): at each
        propagating-band interval eigenvalue s* (on the imaginary axis,
        0.3 < Im < 0.95 — measured +/- 0.6235j and +/- 0.9010j), E_11(s*)
        has an eigenvalue equal to 1 to ~1e-14 (measured 2.4e-15 and
        2.2e-14)."""
        N = 12
        D = build_interval_operator(E2_1, e2_1_alpha_zero, 0.5, 1.0, N)
        eigs = np.linalg.eigvals(-D)
        cands = eigs[(np.abs(eigs.real) < 1e-8)
                     & (eigs.imag > 0.3) & (eigs.imag < 0.95)]
        assert len(cands) == 2  # measured: 0.6234898j and 0.9009689j

        interior = e2_1_evaluator.interior()
        rows_L, idx_L = e2_1_evaluator.rows(0.5, e2_1_alpha_zero)
        rows_R, idx_R = e2_1_evaluator.rows(1.0, e2_1_alpha_zero)
        mirrored = mirror_closure(rows_R, interior,
                                  from_physical_right=False,
                                  row_indices=[int(i) for i in idx_R])

        for s_star in cands:
            E = round_trip_matrix(
                interior[0], interior[1], rows_L, mirrored, s_star, N - 1,
                left_row_indices=[int(i) for i in idx_L],
            )
            ev = np.linalg.eigvals(E)
            assert np.min(np.abs(ev - 1.0)) < 1e-12, (
                f"round-trip identity fails at interval eigenvalue {s_star}"
            )

    def test_round_trip_identity_out_of_scope_at_per_end_det_zero(
        self, e2_1_evaluator, e2_1_alpha_zero
    ):
        """The same interval also has the real eigenvalue
        s = -(1 - psi_L)/(1 + psi_L) = -1/3 (the stable-half-plane mirror
        image of the item-3.2 outflow law, carried by the LEFT cut
        closure): there det M_L(-1/3) = 0 (measured sigma_min ~ 9e-17),
        Lemma 1's per-end-invertibility hypothesis is violated, and the
        round-trip criterion is out of scope by construction — the
        per-end determinant tier owns such points (module docstring)."""
        N = 12
        D = build_interval_operator(E2_1, e2_1_alpha_zero, 0.5, 1.0, N)
        eigs = np.linalg.eigvals(-D)
        assert np.min(np.abs(eigs - (-1.0 / 3.0))) < 1e-9

        interior = e2_1_evaluator.interior()
        rows_L, idx_L = e2_1_evaluator.rows(0.5, e2_1_alpha_zero)
        sigma, status = min_singular_value_ex(
            interior[0], interior[1], rows_L, -1.0 / 3.0 + 0j,
            row_indices=[int(i) for i in idx_L], equilibrate=True,
        )
        assert status == "ok"
        assert sigma < 1e-12


class TestRoundTripOrderingTripwire:
    """The plan-mandated p = 2 cross-validation: at p = 1 every factor is
    a scalar and ANY ordering gives the same rho, so only a p = 2
    configuration with an unstable dense interval eigenvalue can catch an
    ordering bug. The wall-eliminated Gaussian eps = 0.1 pair at n = 12
    (Dj = 10 = Dj_min) is that configuration: p = 2 (flat-limit classical
    E4 interior), dense max Re eig(-D_bc) ~ +0.101 (the same dense tier
    item 3.2 scans; the RBF closure is not a SchemeParams cut-cell, so
    the eigendecomposition of the assembled D_bc stands in for
    interval_stability_scan), and at Dj = 10 the evanescent modes still
    couple the 2x2 factors."""

    def test_round_trip_p2_identity_at_unstable_interval_eigenvalue(
        self, wall_eliminated_pair_n12
    ):
        """rho(E_Dj(s*)) ~ 1 at the matching Dj = N - 1 = 10, with an
        eigenvalue of E equal to 1 (measured min|eig - 1| ~ 5e-13). Both
        ends are separately nonsingular at s* (sigma_min ~ 1.4e-3 /
        4.9e-2 measured), so Lemma 1 applies cleanly."""
        D_bc, iw, io, left_rows, left_idx, mirrored = wall_eliminated_pair_n12
        eigs = np.linalg.eigvals(-D_bc)
        s_star = eigs[np.argmax(eigs.real)]
        assert s_star.real > 0.05  # genuinely unstable (measured +0.101)

        # Lemma 1 hypothesis check: neither end is singular at s*.
        sigma_L, status_L = min_singular_value_ex(
            iw, io, left_rows, s_star, row_indices=left_idx,
            equilibrate=True,
        )
        sigma_R, status_R = min_singular_value_ex(
            mirrored.interior_weights, mirrored.interior_offsets,
            mirrored.boundary_rows, s_star,
            row_indices=mirrored.row_indices, equilibrate=True,
        )
        assert status_L == status_R == "ok"
        assert sigma_L > 1e-6 and sigma_R > 1e-6

        E = round_trip_matrix(iw, io, left_rows, mirrored, s_star, 10,
                              left_row_indices=left_idx)
        ev = np.linalg.eigvals(E)
        assert np.min(np.abs(ev - 1.0)) < 1e-9
        assert np.max(np.abs(ev)) == pytest.approx(1.0, rel=1e-9)

    def test_round_trip_p2_wrong_ordering_fails(
        self, wall_eliminated_pair_n12
    ):
        """The original draft's ordering Mdiag^{-Dj} A_L K^{Dj} A_R has NO
        eigenvalue at 1 at the same s* (measured min|eig - 1| ~ 8.3e-5 vs
        ~5e-13 for the correct ordering — eight orders of magnitude): a
        wrong ordering fails this test, the correct one passes."""
        D_bc, iw, io, left_rows, left_idx, mirrored = wall_eliminated_pair_n12
        eigs = np.linalg.eigvals(-D_bc)
        s_star = eigs[np.argmax(eigs.real)]

        f = round_trip_factors(iw, io, left_rows, mirrored, s_star,
                               left_row_indices=left_idx)
        K = np.diag(f.kappa ** 10)
        Mi = np.diag(f.mu ** (-10.0))

        E_correct = Mi @ f.A_R @ K @ f.A_L
        E_draft = Mi @ f.A_L @ K @ f.A_R
        miss_correct = np.min(np.abs(np.linalg.eigvals(E_correct) - 1.0))
        miss_draft = np.min(np.abs(np.linalg.eigvals(E_draft) - 1.0))

        assert miss_correct < 1e-9
        assert miss_draft > 1e-5
        assert miss_draft > 1e3 * miss_correct

    def test_round_trip_p2_off_by_one_delta_j_fails(
        self, wall_eliminated_pair_n12
    ):
        """An off-by-one in Dj rephases E and the identity fails (plan 48
        item 3.3 convention pinning): measured min|eig - 1| ~ 0.55 at
        Dj = 11 and ~ 0.86 at Dj = 12 vs ~5e-13 at the true Dj = 10."""
        D_bc, iw, io, left_rows, left_idx, mirrored = wall_eliminated_pair_n12
        eigs = np.linalg.eigvals(-D_bc)
        s_star = eigs[np.argmax(eigs.real)]

        for dj_wrong in (11, 12):
            E = round_trip_matrix(iw, io, left_rows, mirrored, s_star,
                                  dj_wrong, left_row_indices=left_idx)
            ev = np.linalg.eigvals(E)
            assert np.min(np.abs(ev - 1.0)) > 0.3


class TestRoundTripWallEliminatedAcceptance:
    """Plan 48 item 3.3 acceptance, first half: the (Gaussian eps = 0.1,
    psi = 1 uniform ends) pair must be flagged — round_trip_radius > 1 OR
    reflection_product > 1 — at the Dj corresponding to the N = 20 case
    where L3 catches it.

    Mechanism refinement (established here; sharpens the item-2.5
    account): the operator L3 actually eigendecomposes is the
    wall-eliminated D_bc = D[1:, 1:], a 19-node interval (Dj = 18). Its
    unstable eigenvalue converges with N to a LEFT-half-line det-zero of
    the wall-eliminated 3-row closure (kreiss check below: unstable,
    witness s ~ 0.09834 + 1.37803j = the N -> infinity limit of the dense
    eigenvalue; N-drift measured 0.1014 -> 0.09834 over n = 12..40), i.e.
    the mechanism is a per-end eigensolution of the ELIMINATED block —
    the full 4-row block items 0.3/2.5 analyzed is genuinely half-line
    stable, so both were right about the operator they looked at. On the
    axis the round-trip radius therefore stays < 1 (measured 0.8706 —
    the per-end pole sits at Re(s) ~ 0.098, off the axis) and the flag
    comes from the product screen; AT the dense eigenvalue the round-trip
    identity rho(E_18(s*)) = 1 holds to ~4e-12 (Lemma 1, hypothesis
    intact at finite N since sigma_min(M_L(s*)) ~ 2.4e-5 > 0)."""

    def test_round_trip_wall_eliminated_pair_flagged_at_matched_dj(
        self, wall_eliminated_pair_n20
    ):
        D_bc, iw, io, left_rows, left_idx, mirrored = wall_eliminated_pair_n20
        assert D_bc.shape == (19, 19)

        res = round_trip_check(iw, io, left_rows, mirrored,
                               delta_j_values=[18],
                               left_row_indices=left_idx)
        assert isinstance(res, RoundTripResult)
        assert res.delta_j_values == (18,)
        assert res.finite is True

        # The plan's acceptance flag (radius > 1 OR product > 1).
        assert (res.round_trip_radius > 1.0
                or res.reflection_product > 1.0)
        # Sharpened: the flag comes from the product screen (measured
        # 2.58e11); the axis radius stays below 1 (measured 0.8706)
        # because the instability is the per-end pole at Re(s) ~ 0.098,
        # not an axis round-trip resonance.
        assert res.reflection_product > 1.0
        assert 0.5 < res.round_trip_radius < 1.0

    def test_round_trip_wall_eliminated_identity_at_dense_eigenvalue(
        self, wall_eliminated_pair_n20
    ):
        """rho(E_18(s*)) = 1 at the dense unstable eigenvalue of -D_bc
        (measured min|eig - 1| ~ 4.4e-12): the analytic tier reproduces
        the repo's documented L3-only catch."""
        D_bc, iw, io, left_rows, left_idx, mirrored = wall_eliminated_pair_n20
        eigs = np.linalg.eigvals(-D_bc)
        s_star = eigs[np.argmax(eigs.real)]
        assert s_star.real == pytest.approx(0.0983, abs=2e-3)
        assert abs(s_star.imag) == pytest.approx(1.3780, abs=2e-3)

        E = round_trip_matrix(iw, io, left_rows, mirrored, s_star, 18,
                              left_row_indices=left_idx)
        ev = np.linalg.eigvals(E)
        assert np.min(np.abs(ev - 1.0)) < 1e-8

    def test_round_trip_wall_eliminated_left_end_is_the_mechanism(
        self, wall_eliminated_pair_n20
    ):
        """The wall-eliminated LEFT closure alone is GKS-unstable with the
        half-line witness at the N -> infinity limit of the dense
        eigenvalue (measured witness 0.09834 + 1.37803j, sigma ~ 3e-17;
        dense s* at n = 20 is 6.5e-5 away) — the per-end tier, run on the
        operator L3 actually time-steps, owns this instability."""
        D_bc, iw, io, left_rows, left_idx, _ = wall_eliminated_pair_n20
        eigs = np.linalg.eigvals(-D_bc)
        s_star = eigs[np.argmax(eigs.real)]

        result = kreiss_stability_check(iw, io, left_rows,
                                        row_indices=left_idx)
        assert result.is_stable is False
        witness = result.witness_s
        assert min(abs(witness - s_star),
                   abs(witness - np.conj(s_star))) < 1e-3


class TestRoundTripE2_1Pairs:
    """Plan 48 item 3.3 acceptance, second half: E2_1 alpha = 0 at several
    (psi_L, psi_R) — radius < 1. Pairs are the dense-stable cut-left +
    uniform-right configurations; per the item-3.2 outflow-law finding,
    every psi_R < 1 cut-RIGHT pair is genuinely eigenvalue-unstable
    through a per-end det-zero (covered by the dedicated test below), so
    those cannot be radius-<-1 acceptance cases."""

    @pytest.mark.parametrize("pair", [(0.5, 1.0), (0.1, 1.0),
                                      (0.9999, 1.0), (1.0, 1.0)])
    def test_round_trip_e2_1_stable_pairs_radius_below_one(
        self, e2_1_alpha_zero, pair
    ):
        res = cutcell_round_trip(E2_1, e2_1_alpha_zero, *pair,
                                 s_grid=_RT_S_GRID)
        assert isinstance(res, RoundTripResult)
        assert res.finite is True
        assert res.defective_points == ()
        # E2_1: t = T - 1 = 4 both ends -> Dj_min = 4 + 4 - 1 = 7, and
        # the default window is the plan's {Dj_min, ..., Dj_min + 5}.
        assert res.delta_j_min == 7
        assert res.delta_j_values == (7, 8, 9, 10, 11, 12)
        # Marginally-stable hyperbolic pair: the radius approaches 1 from
        # below (near-origin consistency physics; measured 1 - 6.1e-6 on
        # the strip) but never crosses it.
        assert 0.99 < res.round_trip_radius < 1.0

        # Dense cross-check at N = 20 (same convention u_t = -D u).
        D = build_interval_operator(E2_1, e2_1_alpha_zero, *pair, 20)
        assert float(np.max(np.linalg.eigvals(-D).real)) < 1e-4

    def test_round_trip_e2_1_product_upper_bounds_radius(
        self, e2_1_alpha_zero
    ):
        """On the axis ||K^Dj|| = ||Mdiag^-Dj|| <= 1, so the decoupled
        reflection product pointwise upper-bounds rho(E_Dj) for every Dj
        — the documented screen property."""
        res = cutcell_round_trip(E2_1, e2_1_alpha_zero, 0.5, 1.0,
                                 s_grid=_RT_S_GRID)
        rho_max = np.nanmax(res.rho, axis=1)
        assert np.all(rho_max <= res.products * (1 + 1e-9) + 1e-12)
        assert res.reflection_product >= res.round_trip_radius

    def test_round_trip_e2_1_outflow_pair_per_end_mechanism(
        self, e2_1_evaluator, e2_1_alpha_zero
    ):
        """Cut-right pair (0.5, 0.5): genuinely unstable — dense max Re
        eig(-D) = (1 - psi_R)/(1 + psi_R) = 1/3 exactly (the item-3.2
        outflow law) — through a PER-END det-zero of the mirrored right
        closure at s = 1/3 (measured sigma_min ~ 9e-17), which violates
        Lemma 1's per-end-invertibility hypothesis: the round-trip radius
        on the axis stays < 1 and must NOT be read as a certificate —
        per-end screening is a mandatory companion (module docstring)."""
        # Dense instability (outflow law).
        D = build_interval_operator(E2_1, e2_1_alpha_zero, 0.5, 0.5, 12)
        assert float(np.max(np.linalg.eigvals(-D).real)) == pytest.approx(
            1.0 / 3.0, rel=1e-9
        )

        # Per-end det-zero at s = 1/3 on the mirrored right closure ...
        interior = e2_1_evaluator.interior()
        rows_R, idx_R = e2_1_evaluator.rows(0.5, e2_1_alpha_zero)
        mirrored = mirror_closure(rows_R, interior,
                                  from_physical_right=False,
                                  row_indices=[int(i) for i in idx_R])
        sigma_R, status_R = min_singular_value_ex(
            mirrored.interior_weights, mirrored.interior_offsets,
            mirrored.boundary_rows, 1.0 / 3.0 + 0j,
            row_indices=mirrored.row_indices, equilibrate=True,
        )
        assert status_R == "ok"
        assert sigma_R < 1e-12
        # ... while the left end is fine there (measured ~3.2e-2).
        rows_L, idx_L = e2_1_evaluator.rows(0.5, e2_1_alpha_zero)
        sigma_L, status_L = min_singular_value_ex(
            interior[0], interior[1], rows_L, 1.0 / 3.0 + 0j,
            row_indices=[int(i) for i in idx_L], equilibrate=True,
        )
        assert status_L == "ok"
        assert sigma_L > 1e-3

        # The axis round-trip radius does not flag it (correctly: the
        # mechanism is per-end, off-axis at real s = 1/3).
        res = cutcell_round_trip(E2_1, e2_1_alpha_zero, 0.5, 0.5,
                                 s_grid=_RT_S_GRID)
        assert res.round_trip_radius < 1.0

    def test_round_trip_e2_1_infeasible_marker(
        self, e2_1_evaluator, e2_1_alpha_zero, monkeypatch
    ):
        """The item-1.2 feasibility guard runs at BOTH end psi before any
        assembly; a vanishing structural denominator returns the distinct
        CutCellInfeasible marker (parameterization failure, never a
        stability verdict)."""
        monkeypatch.setattr(
            e2_1_evaluator, "denominators",
            frozenset({Symbol("psi") - Rational(1, 2)}),
        )
        for pair in ((0.5, 0.9), (0.9, 0.5)):
            marker = cutcell_round_trip(E2_1, e2_1_alpha_zero, *pair,
                                        s_grid=_RT_S_GRID)
            assert isinstance(marker, CutCellInfeasible)
            assert marker.infeasible is True
            assert marker.psi == 0.5
            assert not hasattr(marker, "round_trip_radius")


class TestRoundTripValidation:
    """Dj minimum (tied to item 3.1's IntervalTooShort), defective
    routing, and argument validation."""

    def test_round_trip_delta_j_minimum_values(
        self, e2_1_evaluator, e2_1_alpha_zero, wall_eliminated_pair_n20
    ):
        """Dj_min = max(t_L + t_R - 1, max_node_L + max_node_R + 1):
        supports may not overlap — STRICTER than 3.1's assembly minimum
        (N >= max(m_L + m_R, t_L, t_R) permits column overlap; E2_1's
        assembly minimum is N = 6 nodes but the round trip needs
        Dj >= 7, i.e. N >= 8)."""
        interior = e2_1_evaluator.interior()
        rows, idx = e2_1_evaluator.rows(0.5, e2_1_alpha_zero)
        mirrored = mirror_closure(rows, interior, from_physical_right=False,
                                  row_indices=[int(i) for i in idx])
        assert min_round_trip_delta_j(rows, mirrored) == 7

        # Wall-eliminated Gaussian pair: t_L = 5, t_R = 6 -> Dj_min = 10.
        _, _, _, left_rows, left_idx, mirrored_g = wall_eliminated_pair_n20
        assert min_round_trip_delta_j(
            left_rows, mirrored_g, left_row_indices=left_idx
        ) == 10

    def test_round_trip_below_minimum_raises_interval_too_short(
        self, e2_1_evaluator, e2_1_alpha_zero
    ):
        """Dj below the minimum raises the item-3.1 IntervalTooShort (a
        ValueError subclass) from every entry point."""
        interior = e2_1_evaluator.interior()
        rows, idx = e2_1_evaluator.rows(0.5, e2_1_alpha_zero)
        row_idx = [int(i) for i in idx]
        mirrored = mirror_closure(rows, interior, from_physical_right=False,
                                  row_indices=row_idx)

        assert issubclass(IntervalTooShort, ValueError)
        with pytest.raises(IntervalTooShort, match="supports"):
            round_trip_matrix(interior[0], interior[1], rows, mirrored,
                              1e-6 + 0.5j, 6, left_row_indices=row_idx)
        with pytest.raises(IntervalTooShort, match="Dj >= 7"):
            round_trip_check(interior[0], interior[1], rows, mirrored,
                             delta_j_values=[7, 6],
                             left_row_indices=row_idx)
        with pytest.raises(IntervalTooShort):
            cutcell_round_trip(E2_1, e2_1_alpha_zero, 0.5, 1.0,
                               delta_j_values=[6], s_grid=_RT_S_GRID)

    def test_round_trip_sonic_grid_point_flagged_not_clean(
        self, e2_1_alpha_zero
    ):
        """A grid point at the exact E2 sonic point s = i has a defective
        mode basis: it is recorded in defective_points with rho = nan and
        finite = False — never a clean value (items 0.1 / 2.2 routing) —
        while clean grid points still evaluate."""
        grid = np.array([1j, 1e-6 + 2.0j])
        res = cutcell_round_trip(E2_1, e2_1_alpha_zero, 0.5, 1.0,
                                 delta_j_values=[7], s_grid=grid)

        assert res.defective_points == (1j,)
        assert res.finite is False
        assert np.all(np.isnan(res.rho[0, :]))
        assert np.isnan(res.products[0])
        assert np.isfinite(res.round_trip_radius)
        assert res.argmax_s == grid[1]

    def test_round_trip_check_argument_validation(
        self, e2_1_evaluator, e2_1_alpha_zero
    ):
        interior = e2_1_evaluator.interior()
        rows, idx = e2_1_evaluator.rows(0.5, e2_1_alpha_zero)
        row_idx = [int(i) for i in idx]
        mirrored = mirror_closure(rows, interior, from_physical_right=False,
                                  row_indices=row_idx)

        with pytest.raises(ValueError, match="delta_j_values"):
            round_trip_check(interior[0], interior[1], rows, mirrored,
                             delta_j_values=[], left_row_indices=row_idx)
        with pytest.raises(ValueError, match="n_delta_j"):
            round_trip_check(interior[0], interior[1], rows, mirrored,
                             n_delta_j=0, left_row_indices=row_idx)
        with pytest.raises(ValueError, match="left_row_indices"):
            min_round_trip_delta_j(rows, mirrored, left_row_indices=[0])
        with pytest.raises(ValueError, match="at least one closure row"):
            min_round_trip_delta_j([], mirrored)
        for bad in (0.0, 1.5):
            with pytest.raises(ValueError, match="psi"):
                cutcell_round_trip(E2_1, e2_1_alpha_zero, bad, 0.5,
                                   s_grid=_RT_S_GRID)
            with pytest.raises(ValueError, match="psi"):
                cutcell_round_trip(E2_1, e2_1_alpha_zero, 0.5, bad,
                                   s_grid=_RT_S_GRID)
