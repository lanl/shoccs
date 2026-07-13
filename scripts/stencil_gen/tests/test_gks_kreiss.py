"""Tests for the rigorous GKS Kreiss determinant stability module."""

import numpy as np
import pytest

from stencil_gen.gks_kreiss import (
    KreissResult,
    DefectiveKappaError,
    _classify_imag_axis,
    _kappa_roots_from_poly,
    _refine_witness,
    _sweep_grid,
    kappa_roots,
    kreiss_matrix,
    kreiss_matrix_general,
    kreiss_stability_check,
    make_s_grid,
    min_singular_value,
    min_singular_value_ex,
)


class TestKappaRoots:
    """Tests for kappa_roots (41.3b)."""

    def test_first_order_upwind(self):
        """First-order upwind: u_t = -(u_n - u_{n-1}) has one admissible root.

        Interior stencil: weights=[-1, 1], offsets=[-1, 0].
        Characteristic eqn: s + (-1)*kappa^{-1} + 1*kappa^0 = 0
        Multiply by kappa (L_left=1): -1 + kappa + s*kappa = 0
        => kappa*(1 + s) = 1 => kappa = 1/(1+s)

        At s=1: kappa = 1/2 (admissible, |kappa| < 1).
        The polynomial is degree 1 so there's exactly one root total.
        """
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        s = 1.0

        all_roots, admissible, is_defective = kappa_roots(
            interior_weights, interior_offsets, s
        )

        # Degree-1 polynomial => exactly one root
        assert len(all_roots) == 1
        # The root should be 1/(1+s) = 0.5
        assert abs(all_roots[0] - 0.5) < 1e-12

        # One admissible root
        assert len(admissible) == 1
        assert abs(admissible[0] - 0.5) < 1e-12
        assert abs(admissible[0]) < 1.0

        # Not defective (only one root)
        assert is_defective is False

    def test_second_order_centered(self):
        """Second-order centered: u_t = -(u_{n+1} - u_{n-1})/2.

        Interior stencil: weights=[-0.5, 0.5], offsets=[-1, 1].
        Char eqn: s + (-0.5)*kappa^{-1} + 0.5*kappa = 0
        Multiply by kappa: -0.5 + s*kappa + 0.5*kappa^2 = 0

        Degree-2 polynomial => 2 roots total.
        """
        interior_weights = np.array([-0.5, 0.5])
        interior_offsets = np.array([-1, 1])
        s = 0.5 + 0.3j

        all_roots, admissible, is_defective = kappa_roots(
            interior_weights, interior_offsets, s
        )

        # Degree-2 polynomial
        assert len(all_roots) == 2

        # Verify all roots satisfy the polynomial
        # Q(kappa) = -0.5 + s*kappa + 0.5*kappa^2
        for k in all_roots:
            val = -0.5 + s * k + 0.5 * k**2
            assert abs(val) < 1e-10, f"Root {k} doesn't satisfy polynomial: Q={val}"

    def test_defective_kappa_detected_from_poly(self):
        """Deliberately-coalescing case: double root at kappa=0.5.

        Construct polynomial with known double root at 0.5 and a root at 2.0:
        Q(kappa) = (kappa - 0.5)^2 * (kappa - 2.0)
        """
        # np.poly([0.5, 0.5, 2.0]) gives coefficients [1, -3, 2.25, -0.5]
        poly_coeffs = np.poly([0.5, 0.5, 2.0])

        all_roots, admissible, is_defective = _kappa_roots_from_poly(poly_coeffs)

        # Should have 3 roots total
        assert len(all_roots) == 3

        # Two admissible roots (both near 0.5), one outside (near 2.0)
        assert len(admissible) == 2
        for k in admissible:
            assert abs(k - 0.5) < 1e-6

        # Defective because the two admissible roots are very close
        assert is_defective is True

    def test_no_admissible_roots(self):
        """All roots outside the unit disk."""
        # Q(kappa) = (kappa - 2)(kappa - 3) = kappa^2 - 5*kappa + 6
        poly_coeffs = np.poly([2.0, 3.0])

        all_roots, admissible, is_defective = _kappa_roots_from_poly(poly_coeffs)

        assert len(all_roots) == 2
        assert len(admissible) == 0
        assert is_defective is False

    def test_upwind_various_s_values(self):
        """For upwind, kappa = 1/(1+s); admissible iff Re(s) > 0 or |1+s|>1."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])

        # s=0.1: kappa = 1/1.1 ≈ 0.909, still admissible
        _, adm, _ = kappa_roots(interior_weights, interior_offsets, 0.1)
        assert len(adm) == 1
        assert abs(adm[0] - 1.0 / 1.1) < 1e-12

        # s with large Re: kappa small, definitely admissible
        _, adm, _ = kappa_roots(interior_weights, interior_offsets, 10.0)
        assert len(adm) == 1
        assert abs(adm[0]) < 0.1

    def test_purely_imaginary_s(self):
        """Upwind at s = 2j: kappa = 1/(1+2j) = (1-2j)/5."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        s = 2j

        all_roots, admissible, _ = kappa_roots(interior_weights, interior_offsets, s)
        expected = 1.0 / (1.0 + 2j)
        assert abs(all_roots[0] - expected) < 1e-12
        # |kappa| = 1/sqrt(5) ≈ 0.447 < 1
        assert len(admissible) == 1


class TestKreissMatrix:
    """Tests for kreiss_matrix and min_singular_value (41.3c)."""

    def test_1x1_upwind(self):
        """First-order upwind with one boundary row gives a 1x1 Kreiss matrix.

        Interior: offsets=[-1, 0], weights=[-1, 1].
        At s=1: kappa = 1/2.
        Boundary row 0: weights=[2, -1], offsets=[0, 1] (grid points 0 and 1).
        M[0,0] = s*kappa^0 + 2*kappa^0 + (-1)*kappa^1
               = 1 + 2 - 0.5 = 2.5
        """
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [(np.array([2.0, -1.0]), np.array([0, 1]))]
        s = 1.0

        M = kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s)

        assert M.shape == (1, 1)
        assert abs(M[0, 0] - 2.5) < 1e-12

    def test_2x2_hand_computed(self):
        """Explicit 2x2 case with hand-computed M(s=1).

        Interior: offsets=[-2, -1, 0], weights=[0.1, 0.5, -0.4].
        At s=1, polynomial 0.6*kappa^2 + 0.5*kappa + 0.1 = 0.
        Roots: kappa_a = -1/3, kappa_b = -1/2 (both admissible).

        Boundary row 0: weights=[3, -2], offsets=[0, 1].
        Boundary row 1: weights=[1, 0.5, -0.5], offsets=[0, 1, 2].

        For kappa in {-1/3, -1/2} (order from np.roots may vary):
          M[0, l] = s*kappa^0 + 3*kappa^0 + (-2)*kappa^1
                  = 1 + 3 - 2*kappa = 4 - 2*kappa
          M[1, l] = s*kappa^1 + 1*kappa^0 + 0.5*kappa^1 + (-0.5)*kappa^2
                  = kappa + 1 + 0.5*kappa - 0.5*kappa^2
                  = 1 + 1.5*kappa - 0.5*kappa^2
        """
        interior_weights = np.array([0.1, 0.5, -0.4])
        interior_offsets = np.array([-2, -1, 0])
        boundary_rows = [
            (np.array([3.0, -2.0]), np.array([0, 1])),
            (np.array([1.0, 0.5, -0.5]), np.array([0, 1, 2])),
        ]
        s = 1.0

        M = kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s)

        assert M.shape == (2, 2)

        # Get the admissible roots to know the column ordering
        _, admissible, _ = kappa_roots(interior_weights, interior_offsets, s)
        assert len(admissible) == 2

        # Verify each entry against the formula
        for ell, kappa in enumerate(admissible):
            # Row 0: M[0, l] = 4 - 2*kappa
            expected_0 = 4.0 - 2.0 * kappa
            assert abs(M[0, ell] - expected_0) < 1e-12, (
                f"M[0,{ell}]: got {M[0,ell]}, expected {expected_0}"
            )
            # Row 1: M[1, l] = 1 + 1.5*kappa - 0.5*kappa^2
            expected_1 = 1.0 + 1.5 * kappa - 0.5 * kappa**2
            assert abs(M[1, ell] - expected_1) < 1e-12, (
                f"M[1,{ell}]: got {M[1,ell]}, expected {expected_1}"
            )

    def test_2x2_exact_values(self):
        """Verify M entries for kappa = -1/3, -1/2 (same setup as above).

        For kappa = -1/3: M[0] = 4 + 2/3 = 14/3, M[1] = 1 - 1/2 - 1/18 = 4/9
        For kappa = -1/2: M[0] = 4 + 1 = 5,       M[1] = 1 - 3/4 - 1/8 = 1/8

        The exact matrix (up to column reordering) is:
            [[14/3, 5], [4/9, 1/8]]  or  [[5, 14/3], [1/8, 4/9]]
        """
        interior_weights = np.array([0.1, 0.5, -0.4])
        interior_offsets = np.array([-2, -1, 0])
        boundary_rows = [
            (np.array([3.0, -2.0]), np.array([0, 1])),
            (np.array([1.0, 0.5, -0.5]), np.array([0, 1, 2])),
        ]
        s = 1.0

        M = kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s)
        _, admissible, _ = kappa_roots(interior_weights, interior_offsets, s)

        # Map kappa values to expected M columns
        for ell, kappa in enumerate(admissible):
            if abs(kappa - (-1.0 / 3.0)) < 1e-10:
                assert abs(M[0, ell] - 14.0 / 3.0) < 1e-12
                assert abs(M[1, ell] - 4.0 / 9.0) < 1e-12
            elif abs(kappa - (-1.0 / 2.0)) < 1e-10:
                assert abs(M[0, ell] - 5.0) < 1e-12
                assert abs(M[1, ell] - 1.0 / 8.0) < 1e-12
            else:
                pytest.fail(f"Unexpected admissible root: {kappa}")

    def test_shape_mismatch_raises(self):
        """ValueError when admissible root count != boundary row count."""
        # Upwind has 1 admissible root at s=1, but we provide 2 boundary rows
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [
            (np.array([1.0]), np.array([0])),
            (np.array([1.0]), np.array([0])),
        ]
        with pytest.raises(ValueError, match="admissible roots"):
            kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s=1.0)

    def test_defective_raises(self):
        """DefectiveKappaError when admissible roots coalesce.

        Reverse-engineer a stencil whose characteristic polynomial is
        (kappa - 0.3)^2 * (kappa - 5) at s = 0.4, giving a double admissible
        root at kappa = 0.3.

        For offsets [-2, -1, 0, 1] with L_left=2, shifted=[0,1,2,3]:
          Q(kappa) = w_{-2} + w_{-1}*k + (w_0 + s)*k^2 + w_1*k^3
        Target:    -0.45   + 3.09*k    - 5.6*k^2       + k^3
        So w_{-2}=-0.45, w_{-1}=3.09, w_0=-6.0 (since w_0+s=-5.6), w_1=1.0.
        """
        interior_weights = np.array([-0.45, 3.09, -6.0, 1.0])
        interior_offsets = np.array([-2, -1, 0, 1])
        s = 0.4

        # Confirm kappa_roots detects defective roots
        _, admissible, is_defective = kappa_roots(
            interior_weights, interior_offsets, s
        )
        assert len(admissible) == 2
        assert is_defective is True

        # kreiss_matrix must raise DefectiveKappaError
        boundary_rows = [
            (np.array([1.0]), np.array([0])),
            (np.array([1.0]), np.array([0])),
        ]
        with pytest.raises(DefectiveKappaError):
            kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s)

    def test_min_singular_value_1x1(self):
        """min_singular_value returns |M[0,0]| for a 1x1 matrix."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [(np.array([2.0, -1.0]), np.array([0, 1]))]
        s = 1.0

        sv = min_singular_value(interior_weights, interior_offsets, boundary_rows, s)
        assert abs(sv - 2.5) < 1e-12

    def test_min_singular_value_2x2(self):
        """min_singular_value matches numpy SVD on the 2x2 hand-computed M."""
        interior_weights = np.array([0.1, 0.5, -0.4])
        interior_offsets = np.array([-2, -1, 0])
        boundary_rows = [
            (np.array([3.0, -2.0]), np.array([0, 1])),
            (np.array([1.0, 0.5, -0.5]), np.array([0, 1, 2])),
        ]
        s = 1.0

        sv = min_singular_value(interior_weights, interior_offsets, boundary_rows, s)

        # Cross-check: build M manually and compute SVD
        M = kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s)
        expected_sv = float(np.linalg.svd(M, compute_uv=False)[-1])
        assert abs(sv - expected_sv) < 1e-12

    def test_min_singular_value_m_gt_p_dispatches_general(self):
        """m > p boundary rows now dispatch to the augmented matrix (plan 48 0.2).

        Before item 0.2 this returned inf (the square kreiss_matrix raised
        ValueError and the extra rows were silently dropped). Now the
        augmented m x m matrix is built and a finite sigma_min is returned.
        """
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [
            (np.array([1.0]), np.array([0])),
            (np.array([1.0]), np.array([0])),
        ]
        sv = min_singular_value(interior_weights, interior_offsets, boundary_rows, s=1.0)
        assert np.isfinite(sv)
        M = kreiss_matrix_general(
            interior_weights, interior_offsets, boundary_rows, 1.0
        )
        expected_sv = float(np.linalg.svd(M, compute_uv=False)[-1])
        assert sv == pytest.approx(expected_sv, abs=1e-14)

    def test_min_singular_value_root_count_anomaly_returns_inf(self):
        """min_singular_value returns inf when the admissible-root count at
        this s differs from the structural count p = L_left.

        E2 centered interior at s=0: the characteristic roots are kappa = ±1,
        both exactly on the unit circle, so 0 admissible roots vs p = 1.
        """
        interior_weights = np.array([-0.5, 0.5])
        interior_offsets = np.array([-1, 1])
        boundary_rows = [(np.array([1.0]), np.array([0]))]
        sv = min_singular_value(interior_weights, interior_offsets, boundary_rows, s=0.0)
        assert sv == np.inf

    def test_min_singular_value_defective_returns_inf(self):
        """min_singular_value returns inf on the DefectiveKappaError path.

        Uses the same engineered stencil as test_defective_raises:
        Q(kappa) = (kappa - 0.3)^2 * (kappa - 5) at s = 0.4.
        """
        interior_weights = np.array([-0.45, 3.09, -6.0, 1.0])
        interior_offsets = np.array([-2, -1, 0, 1])
        boundary_rows = [
            (np.array([1.0]), np.array([0])),
            (np.array([1.0]), np.array([0])),
        ]
        sv = min_singular_value(
            interior_weights, interior_offsets, boundary_rows, s=0.4
        )
        assert sv == np.inf


class TestKreissMatrixGeneral:
    """Tests for the augmented Kreiss matrix (plan 48 item 0.2).

    Sign convention: u_t = -sum_k a_k u_{n+k}. Worked scalar example from
    the plan: upwind interior du_n/dt = -(u_n - u_{n-1}), i.e.
    offsets = [-1, 0], weights = [-1, 1], h = 1. Characteristic equation
    s + (-1)/kappa + 1 = 0 gives kappa(s) = 1/(1+s), so p = L_left = 1.
    With m = 2 closure rows there is one nodal unknown phi_0 and one
    amplitude alpha; M(s) is 2x2:

        M[0,0] = s + w_00           M[0,1] = sum_{j>=1} w_0j kappa^j
        M[1,0] = w_10               M[1,1] = s*kappa + sum_{j>=1} w_1j kappa^j
    """

    _UPWIND_W = np.array([-1.0, 1.0])
    _UPWIND_O = np.array([-1, 0])

    def test_upwind_m2_hand_computed(self):
        """Every entry of the 2x2 augmented matrix matches the hand formula.

        At s=1, kappa = 1/2. Rows with support {0, 1, 2}:
          row 0: w = [2, -1, 0.5]   row 1: w = [-0.5, 1.5, -1]
        M[0,0] = 1 + 2 = 3
        M[0,1] = -1*(1/2) + 0.5*(1/4) = -0.375
        M[1,0] = -0.5
        M[1,1] = 1*(1/2) + 1.5*(1/2) - 1*(1/4) = 1.0
        """
        boundary_rows = [
            (np.array([2.0, -1.0, 0.5]), np.array([0, 1, 2])),
            (np.array([-0.5, 1.5, -1.0]), np.array([0, 1, 2])),
        ]
        M = kreiss_matrix_general(
            self._UPWIND_W, self._UPWIND_O, boundary_rows, s=1.0
        )

        assert M.shape == (2, 2)
        assert M[0, 0] == pytest.approx(3.0, abs=1e-14)
        assert M[0, 1] == pytest.approx(-0.375, abs=1e-14)
        assert M[1, 0] == pytest.approx(-0.5, abs=1e-14)
        assert M[1, 1] == pytest.approx(1.0, abs=1e-14)

    def test_engineered_unstable_det_zero_at_s_1(self):
        """Engineered m=2 closure with det M(s) = 0 at s = 1 is flagged.

        Mirrors the existing engineered-unstable 1x1 pattern (boundary
        weights [0, -2]) augmented with the interior row at node 1:
          row 0: w = [0, -2] at {0, 1}    row 1: w = [-1, 1] at {0, 1}
        M(s) = [[s, -2*kappa], [-1, (s+1)*kappa]] with (s+1)*kappa = 1, so
        det M = s - 2*kappa = (s-1)(s+2)/(1+s): a zero at s = 1 (Re > 0),
        a genuine GKS eigensolution.
        """
        boundary_rows = [
            (np.array([0.0, -2.0]), np.array([0, 1])),
            (np.array([-1.0, 1.0]), np.array([0, 1])),
        ]

        # Hand-check the matrix at s=1, kappa=1/2 first
        M = kreiss_matrix_general(
            self._UPWIND_W, self._UPWIND_O, boundary_rows, s=1.0
        )
        expected = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=complex)
        np.testing.assert_allclose(M, expected, atol=1e-14)
        # Exactly singular at s=1 (sigma_min, not det, to avoid the LAPACK
        # divide-by-zero warning on singular input)
        assert np.linalg.svd(M, compute_uv=False)[-1] < 1e-14

        # The full check dispatches to the augmented matrix (m=2 != p=1)
        # and finds the violation.
        result = kreiss_stability_check(
            self._UPWIND_W, self._UPWIND_O, boundary_rows,
            s_grid_params={"s_max": 5.0, "n_radial": 30, "n_imag": 81,
                           "imag_max": 5.0},
        )

        assert result.is_stable is False
        assert result.witness_sigma_min < 1e-6
        assert abs(result.witness_s - 1.0) < 0.1, (
            f"Expected witness near s=1+0j, got {result.witness_s}"
        )

    def test_stable_closure_row1_interior(self):
        """Stable m=2 closure: Dirichlet-like row 0 + interior row at node 1.

          row 0: w = [1] at {0}          row 1: w = [-1, 1] at {0, 1}
        M(s) = [[s+1, 0], [-1, (s+1)*kappa]] = [[s+1, 0], [-1, 1]] since
        (s+1)*kappa = 1; det M = s + 1 has no zero for Re(s) >= 0.
        """
        boundary_rows = [
            (np.array([1.0]), np.array([0])),
            (np.array([-1.0, 1.0]), np.array([0, 1])),
        ]

        result = kreiss_stability_check(
            self._UPWIND_W, self._UPWIND_O, boundary_rows,
            s_grid_params={"s_max": 5.0, "n_radial": 10, "n_imag": 20,
                           "imag_max": 10.0},
        )

        assert result.is_stable is True
        assert result.defective_kappa_detected is False

    def test_reduces_to_kreiss_matrix_when_m_equals_p(self):
        """Regression identity: m = p and default row_indices reproduces
        kreiss_matrix exactly."""
        cases = [
            # p = 1: upwind with one boundary row
            (
                self._UPWIND_W,
                self._UPWIND_O,
                [(np.array([2.0, -1.0]), np.array([0, 1]))],
            ),
            # p = 2: the TestKreissMatrix hand-computed stencil, two rows
            (
                np.array([0.1, 0.5, -0.4]),
                np.array([-2, -1, 0]),
                [
                    (np.array([3.0, -2.0]), np.array([0, 1])),
                    (np.array([1.0, 0.5, -0.5]), np.array([0, 1, 2])),
                ],
            ),
        ]
        # s values chosen so the p=2 artifact stencil has exactly 2
        # admissible roots at each (it is an engineered test stencil, not a
        # genuine hyperbolic interior, so that does not hold at every s).
        for iw, io, brows in cases:
            for s in [1.0, 1.5 + 0.5j, 2.0 - 1.0j]:
                M_sq = kreiss_matrix(iw, io, brows, s)
                M_gen = kreiss_matrix_general(iw, io, brows, s)
                np.testing.assert_allclose(
                    M_gen, M_sq, rtol=1e-14, atol=1e-14,
                    err_msg=f"reduction identity failed at s={s}",
                )

    def test_row_indices_temporal_delta_on_grid_node(self):
        """The temporal delta follows the row's grid node n_i, not row index i.

        Upwind, m=2, row_indices=[0, 2]: row 1 sits at grid node 2, so its
        temporal term contributes s*kappa^2 to the root column, and its
        nodal-column entry gets no s (n_1 = 2 != 0):
          row 0: w = [1] at {0}          row 1: w = [-1, 1] at {0, 1}
        At s=1, kappa=1/2:
          M[0,0] = 1 + 1 = 2      M[0,1] = 0
          M[1,0] = -1             M[1,1] = 1*(1/4) + 1*(1/2) = 0.75
        With default row_indices, M[1,1] = (s+1)*kappa = 1 instead.
        """
        boundary_rows = [
            (np.array([1.0]), np.array([0])),
            (np.array([-1.0, 1.0]), np.array([0, 1])),
        ]
        M = kreiss_matrix_general(
            self._UPWIND_W, self._UPWIND_O, boundary_rows, s=1.0,
            row_indices=[0, 2],
        )
        expected = np.array([[2.0, 0.0], [-1.0, 0.75]], dtype=complex)
        np.testing.assert_allclose(M, expected, atol=1e-14)

        M_default = kreiss_matrix_general(
            self._UPWIND_W, self._UPWIND_O, boundary_rows, s=1.0
        )
        assert M_default[1, 1] == pytest.approx(1.0, abs=1e-14)

    def test_m_less_than_p_raises(self):
        """m < p is rejected: the augmented construction needs m >= p rows."""
        interior_weights = np.array([0.1, 0.5, -0.4])
        interior_offsets = np.array([-2, -1, 0])  # p = L_left = 2
        boundary_rows = [(np.array([1.0]), np.array([0]))]
        with pytest.raises(ValueError, match="m >= p"):
            kreiss_matrix_general(
                interior_weights, interior_offsets, boundary_rows, s=1.0
            )

    def test_negative_grid_node_raises(self):
        """Closure rows referencing j < 0 are invalid on the left half-line."""
        boundary_rows = [
            (np.array([1.0, -1.0]), np.array([-1, 0])),
            (np.array([-1.0, 1.0]), np.array([0, 1])),
        ]
        with pytest.raises(ValueError, match="negative grid node"):
            kreiss_matrix_general(
                self._UPWIND_W, self._UPWIND_O, boundary_rows, s=1.0
            )

    def test_row_indices_length_mismatch_raises(self):
        """row_indices must have one grid node per boundary row."""
        boundary_rows = [
            (np.array([1.0]), np.array([0])),
            (np.array([-1.0, 1.0]), np.array([0, 1])),
        ]
        with pytest.raises(ValueError, match="row_indices"):
            kreiss_matrix_general(
                self._UPWIND_W, self._UPWIND_O, boundary_rows, s=1.0,
                row_indices=[0],
            )


class TestSGridSweep:
    """Tests for make_s_grid and _sweep_grid (41.3d)."""

    def test_grid_shape_default(self):
        """Default grid shape is (n_imag, n_radial + 1) = (120, 41)."""
        grid = make_s_grid()
        assert grid.shape == (120, 41)
        assert grid.dtype == complex

    def test_grid_shape_custom(self):
        """Custom parameters produce the expected shape."""
        grid = make_s_grid(n_radial=10, n_imag=30)
        assert grid.shape == (30, 11)

    def test_grid_real_parts_positive(self):
        """All grid points have Re(s) > 0 (right half-plane)."""
        grid = make_s_grid()
        assert np.all(grid.real > 0)

    def test_grid_imaginary_axis_strip(self):
        """Column 0 is the imaginary-axis strip at Re(s) = eps_imag."""
        eps = 1e-6
        grid = make_s_grid(eps_imag=eps, n_radial=10, n_imag=20)
        np.testing.assert_allclose(grid[:, 0].real, eps)

    def test_grid_imaginary_range(self):
        """Imaginary parts span [-imag_max, imag_max]."""
        grid = make_s_grid(imag_max=15.0, n_imag=50, n_radial=5)
        assert grid.imag.min() == pytest.approx(-15.0)
        assert grid.imag.max() == pytest.approx(15.0)

    def test_grid_real_log_spacing(self):
        """Re(s) values in columns 1..n_radial are logarithmically spaced."""
        grid = make_s_grid(s_max=10.0, n_radial=20, n_imag=5)
        # All rows share the same Re values; check row 0
        re_vals = grid[0, 1:].real  # skip imaginary-axis strip column
        assert re_vals[0] == pytest.approx(1e-4)
        assert re_vals[-1] == pytest.approx(10.0)
        # Log-spacing means ratios between consecutive values are approximately constant
        ratios = re_vals[1:] / re_vals[:-1]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-10)

    def test_sweep_grid_upwind_stable(self):
        """First-order upwind with natural boundary is stable: min(sigma) > 0.01.

        Interior: weights=[-1, 1], offsets=[-1, 0] (first-order upwind).
        Boundary row 0: weights=[1], offsets=[0] (identity = Dirichlet at left).
        This is GKS-stable, so sigma_min should be bounded away from zero.
        """
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        # Dirichlet boundary: u_0 = prescribed → the boundary equation
        # is simply u_0(t) = g(t), which in the homogeneous stability problem
        # becomes u_0 = 0. Row: [1]*[kappa^0] = kappa^0 = 1.
        boundary_rows = [(np.array([1.0]), np.array([0]))]

        # Small grid for speed
        grid = make_s_grid(s_max=5.0, n_radial=10, n_imag=20, imag_max=10.0)
        sigma_field, argmin_idx = _sweep_grid(
            interior_weights, interior_offsets, boundary_rows, grid
        )

        assert sigma_field.shape == grid.shape
        assert np.all(np.isfinite(sigma_field))
        assert sigma_field.min() > 0.01, (
            f"Expected stable scheme, got min(sigma) = {sigma_field.min()}"
        )

    def test_sweep_grid_argmin_consistent(self):
        """argmin_idx points to the actual minimum of sigma_field."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [(np.array([1.0]), np.array([0]))]

        grid = make_s_grid(s_max=3.0, n_radial=5, n_imag=10, imag_max=5.0)
        sigma_field, argmin_idx = _sweep_grid(
            interior_weights, interior_offsets, boundary_rows, grid
        )

        flat = sigma_field.ravel()
        assert flat[argmin_idx] == flat.min()


class TestRefineWitness:
    """Tests for _refine_witness (41.3e)."""

    def test_converges_to_known_zero(self):
        """Perturbed start near s=1+0j on a known-unstable scheme converges to sigma_min < 1e-6.

        Construct an unstable scheme: first-order upwind interior with boundary
        weights=[0, -2], offsets=[0, 1]. The Kreiss matrix is 1x1:
            M(s) = s + 0*kappa^0 + (-2)*kappa^1 = s - 2/(1+s) = (s^2 + s - 2)/(1+s)
        which has a zero at s=1 (Re>0), making this scheme GKS-unstable.
        """
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [(np.array([0.0, -2.0]), np.array([0, 1]))]

        # Start from a perturbed point
        s_start = 1.5 + 0.5j
        s_refined = _refine_witness(
            interior_weights, interior_offsets, boundary_rows, s_start
        )

        # Verify convergence: sigma_min at refined s should be < 1e-6
        sv = min_singular_value(
            interior_weights, interior_offsets, boundary_rows, s_refined
        )
        assert sv < 1e-6, f"Expected sigma_min < 1e-6, got {sv} at s={s_refined}"

        # The refined s should be near s=1+0j
        assert abs(s_refined - 1.0) < 0.1, (
            f"Expected refined s near 1+0j, got {s_refined}"
        )

    def test_stays_in_right_half_plane(self):
        """Refined witness has Re(s) >= 0 even when starting near the imaginary axis.

        Same unstable scheme as above, starting from s_start = 0.1 + 0.1j.
        """
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [(np.array([0.0, -2.0]), np.array([0, 1]))]

        s_start = 0.1 + 0.1j
        s_refined = _refine_witness(
            interior_weights, interior_offsets, boundary_rows, s_start
        )

        assert s_refined.real >= 0, f"Re(s) = {s_refined.real} < 0"

    def test_stable_scheme_stays_bounded(self):
        """On a stable scheme, _refine_witness still returns a valid s with sigma_min > 0."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        # Dirichlet boundary: identity at grid point 0
        boundary_rows = [(np.array([1.0]), np.array([0]))]

        s_start = 1.5 + 0.5j
        s_refined = _refine_witness(
            interior_weights, interior_offsets, boundary_rows, s_start
        )

        sv = min_singular_value(
            interior_weights, interior_offsets, boundary_rows, s_refined
        )
        # Stable scheme: sigma_min should be bounded away from zero
        assert sv > 0.01, f"Expected sigma_min > 0.01 on stable scheme, got {sv}"


class TestClassifyImagAxis:
    """Tests for _classify_imag_axis (41.3f)."""

    def test_inward_perturbation_violation(self):
        """Case A: root at kappa=1 moves inward under Re(s) perturbation → GKS violation.

        Construct Q(kappa; s=0) = (kappa - 1)(kappa - 0.5) via stencil with
        offsets [-1, 0, 1], weights [0.5, -1.5, 1.0]. At s=0, roots are
        kappa=1 and kappa=0.5. For small positive delta, d(kappa)/ds = -2
        at kappa=1 (computed from implicit differentiation), so the root
        moves inward → outgoing_mode_detected (a GKS violation).
        """
        interior_weights = np.array([0.5, -1.5, 1.0])
        interior_offsets = np.array([-1, 0, 1])

        result = _classify_imag_axis(
            interior_weights, interior_offsets, s_candidate=0.0 + 0j, delta=1e-4
        )
        assert result == "outgoing_mode_detected"

    def test_outward_perturbation_non_violation(self):
        """Case B: root at kappa=1 moves outward under Re(s) perturbation → non-violation.

        Construct Q(kappa; s=0) = (kappa - 1)(kappa - 2) via stencil with
        offsets [-1, 0, 1], weights [2.0, -3.0, 1.0]. At s=0, roots are
        kappa=1 and kappa=2. For small positive delta, d(kappa)/ds = +1
        at kappa=1, so the root moves outward → all_incoming (non-violation).
        """
        interior_weights = np.array([2.0, -3.0, 1.0])
        interior_offsets = np.array([-1, 0, 1])

        result = _classify_imag_axis(
            interior_weights, interior_offsets, s_candidate=0.0 + 0j, delta=1e-4
        )
        assert result == "all_incoming"

    def test_no_near_unit_roots_returns_no_candidates(self):
        """When no roots are near the unit circle, return 'no_candidates'."""
        # Upwind at s=5: kappa = 1/6 ≈ 0.167, far from unit circle
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])

        result = _classify_imag_axis(
            interior_weights, interior_offsets, s_candidate=5.0 + 0j
        )
        assert result == "no_candidates"

    def test_defective_roots_far_from_unit_circle_returns_no_candidates(self):
        """When defective roots are far from unit circle, return 'no_candidates'.

        The engineered defective stencil has a double root at |kappa|=0.3,
        which is far from the unit circle. The defective check should only
        fire for near-unit-circle roots, so this returns 'no_candidates'.
        """
        # Same engineered defective stencil from TestKreissMatrix:
        # double root at kappa=0.3 (|kappa|=0.3, far from unit circle)
        interior_weights = np.array([-0.45, 3.09, -6.0, 1.0])
        interior_offsets = np.array([-2, -1, 0, 1])

        result = _classify_imag_axis(
            interior_weights, interior_offsets, s_candidate=0.4 + 0j
        )
        assert result == "no_candidates"

    def test_defective_roots_near_unit_circle_returns_defective(self):
        """When defective roots are near the unit circle, return 'defective'.

        Engineer a stencil whose characteristic polynomial has a double root
        at kappa=0.99995 (|kappa| - 1 = -5e-5, within unit_tol=1e-4) and a
        third root at kappa=5 (outside the unit disk).

        Target polynomial: (kappa - 0.99995)^2 (kappa - 5.0)
        With offsets [-2, -1, 0, 1], L_left=2, the polynomial is:
            Q(kappa) = w_0 + w_1*kappa + (w_2 + s)*kappa^2 + w_3*kappa^3
        Set s=0 and match coefficients.
        """
        r = 0.99995
        a = 5.0
        # (kappa - r)^2 (kappa - a) = kappa^3 - (2r+a)*kappa^2
        #                            + (r^2 + 2*r*a)*kappa - r^2*a
        w_3 = 1.0
        s_val = 0.0 + 0j
        w_2 = -(2 * r + a) - s_val  # coeff of kappa^2 minus s
        w_1 = r**2 + 2 * r * a
        w_0 = -(r**2 * a)

        interior_weights = np.array([w_0, w_1, w_2, w_3])
        interior_offsets = np.array([-2, -1, 0, 1])

        result = _classify_imag_axis(
            interior_weights, interior_offsets, s_candidate=s_val
        )
        assert result == "defective"


class TestKreissStabilityCheck:
    """Tests for kreiss_stability_check orchestrator (41.3g)."""

    def test_stable_upwind_dirichlet(self):
        """First-order upwind with Dirichlet BC is GKS-stable."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [(np.array([1.0]), np.array([0]))]

        result = kreiss_stability_check(
            interior_weights, interior_offsets, boundary_rows,
            s_grid_params={"s_max": 5.0, "n_radial": 10, "n_imag": 20,
                           "imag_max": 10.0},
        )

        assert isinstance(result, KreissResult)
        assert result.is_stable is True
        assert result.compute_time > 0
        assert result.s_grid_shape == (20, 11)

    def test_unstable_scheme_detected(self):
        """Known-unstable scheme: upwind with boundary weights [0, -2] has zero at s=1."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [(np.array([0.0, -2.0]), np.array([0, 1]))]

        # Dense grid near s=1+0j: need fine imaginary spacing so a grid point
        # lands close enough for the sweep to trigger refinement.
        result = kreiss_stability_check(
            interior_weights, interior_offsets, boundary_rows,
            s_grid_params={"s_max": 5.0, "n_radial": 30, "n_imag": 81,
                           "imag_max": 5.0},
        )

        assert result.is_stable is False
        assert result.witness_sigma_min < 1e-6
        assert result.witness_s is not None

    def test_compute_time_positive(self):
        """compute_time is always positive."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [(np.array([1.0]), np.array([0]))]

        result = kreiss_stability_check(
            interior_weights, interior_offsets, boundary_rows,
            s_grid_params={"s_max": 3.0, "n_radial": 5, "n_imag": 10,
                           "imag_max": 5.0},
            refine=False,
        )

        assert result.compute_time > 0

    def test_no_refine_option(self):
        """With refine=False, result depends on grid resolution only."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [(np.array([1.0]), np.array([0]))]

        result = kreiss_stability_check(
            interior_weights, interior_offsets, boundary_rows,
            s_grid_params={"s_max": 3.0, "n_radial": 5, "n_imag": 10,
                           "imag_max": 5.0},
            refine=False,
        )

        # Stable scheme stays stable even without refinement
        assert result.is_stable is True
        assert result.witness_s is not None
        assert result.witness_sigma_min > 0


class TestMinSingularValueEx:
    """Tests for min_singular_value_ex (plan 48 item 0.1)."""

    def test_ok_status_matches_wrapper(self):
        """On a regular point, status is 'ok' and sigma matches the wrapper."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [(np.array([2.0, -1.0]), np.array([0, 1]))]
        s = 1.0

        sigma, status = min_singular_value_ex(
            interior_weights, interior_offsets, boundary_rows, s
        )
        assert status == "ok"
        assert sigma == pytest.approx(2.5)
        assert sigma == min_singular_value(
            interior_weights, interior_offsets, boundary_rows, s
        )

    def test_defective_status(self):
        """Defective admissible roots return (inf, 'defective'), never sigma=0.

        Engineered stencil with Q(kappa; s=0.4) = (kappa - 0.3)^2 (kappa - 5):
        double admissible root at kappa = 0.3 (see test_defective_raises).
        """
        interior_weights = np.array([-0.45, 3.09, -6.0, 1.0])
        interior_offsets = np.array([-2, -1, 0, 1])
        boundary_rows = [
            (np.array([1.0]), np.array([0])),
            (np.array([1.0]), np.array([0])),
        ]

        sigma, status = min_singular_value_ex(
            interior_weights, interior_offsets, boundary_rows, s=0.4
        )
        assert status == "defective"
        assert sigma == np.inf  # flagged, NOT mapped to 0 (no false witness)

    def test_count_mismatch_status(self):
        """Admissible-root count != p = L_left returns (inf, 'count_mismatch').

        E2 centered interior at s=0: characteristic roots kappa = ±1 sit
        exactly on the unit circle, so 0 admissible roots vs p = 1.
        (Before plan 48 item 0.2 the m != p case also landed here; it now
        dispatches to the augmented matrix instead — see
        TestKreissMatrixGeneral.)
        """
        interior_weights = np.array([-0.5, 0.5])
        interior_offsets = np.array([-1, 1])
        boundary_rows = [(np.array([1.0]), np.array([0]))]

        sigma, status = min_singular_value_ex(
            interior_weights, interior_offsets, boundary_rows, s=0.0
        )
        assert status == "count_mismatch"
        assert sigma == np.inf

    def test_m_less_than_p_is_count_mismatch(self):
        """m < p boundary rows is unresolvable: (inf, 'count_mismatch')."""
        # p = L_left = 2, but only one boundary row
        interior_weights = np.array([0.1, 0.5, -0.4])
        interior_offsets = np.array([-2, -1, 0])
        boundary_rows = [(np.array([1.0]), np.array([0]))]

        sigma, status = min_singular_value_ex(
            interior_weights, interior_offsets, boundary_rows, s=1.0
        )
        assert status == "count_mismatch"
        assert sigma == np.inf


class TestDefectiveKappaObservability:
    """Defective/anomalous kappa points are observable, not 'maximally stable'.

    Plan 48 item 0.1: previously min_singular_value returned inf on
    DefectiveKappaError/ValueError, so anomalous s-points could never be
    selected as witnesses and an engineered double-root closure came back
    is_stable=True, defective_kappa_detected=False. Now anomalies are
    recorded in a side list, near-axis ones are classified, and the check
    fails conservatively (verdict 'defective') while any remain unclassified.
    """

    # Engineered stencil: Q(kappa; s) = w_{-2} + w_{-1} k + (w_0 + s) k^2 + w_1 k^3
    # equals (kappa - 0.3)^2 (kappa - 5) at s = 0.4 — a double ADMISSIBLE root
    # at kappa = 0.3 (same construction as TestKreissMatrix.test_defective_raises).
    _IW = np.array([-0.45, 3.09, -6.0, 1.0])
    _IO = np.array([-2, -1, 0, 1])
    _BROWS = [
        (np.array([1.0]), np.array([0])),
        (np.array([1.0]), np.array([0])),
    ]
    # Grid engineered to contain s = 0.4 + 0j exactly: the logspace radial
    # sweep ends at 10**log10(0.4) == 0.4, and odd n_imag puts Im = 0 on
    # the grid.
    _GRID_PARAMS = {"s_max": 0.4, "n_radial": 10, "n_imag": 21, "imag_max": 5.0}

    def test_sweep_grid_records_anomalies(self):
        """_sweep_grid appends anomalous grid points to the side list."""
        grid = make_s_grid(**self._GRID_PARAMS)
        anomalies = []
        sigma_field, _ = _sweep_grid(
            self._IW, self._IO, self._BROWS, grid, anomalies=anomalies
        )

        assert len(anomalies) == 1
        s_a, status = anomalies[0]
        assert abs(s_a - 0.4) < 1e-9
        assert status == "defective"
        # The anomalous point carries sigma = inf in the field (not 0)
        flat = grid.ravel()
        idx = int(np.argmin(np.abs(flat - 0.4)))
        assert sigma_field.ravel()[idx] == np.inf

    def test_engineered_double_root_flagged_unstable(self):
        """Double admissible root at kappa=0.3, s=0.4: conservative fail.

        This is the empirical reproduction from the plan-48 gap analysis:
        before item 0.1 this returned is_stable=True,
        defective_kappa_detected=False.
        """
        result = kreiss_stability_check(
            self._IW, self._IO, self._BROWS,
            s_grid_params=self._GRID_PARAMS,
        )

        assert result.defective_kappa_detected is True
        assert result.is_stable is False
        assert result.imaginary_axis_perturbation_verdict == "defective"
        # The off-axis defective point is recorded and marked unclassifiable
        assert any(
            abs(s_a - 0.4) < 1e-9
            and status == "defective"
            and classification == "not_classified"
            for s_a, status, classification in result.anomalous_points
        )

    def test_engineered_double_root_no_refine(self):
        """Same verdict with refine=False (sweep-only path)."""
        result = kreiss_stability_check(
            self._IW, self._IO, self._BROWS,
            s_grid_params=self._GRID_PARAMS,
            refine=False,
        )

        assert result.defective_kappa_detected is True
        assert result.is_stable is False
        assert result.imaginary_axis_perturbation_verdict == "defective"

    def test_stable_scheme_reports_no_anomalies(self):
        """Upwind + Dirichlet: no anomalies recorded, flags stay False/empty."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [(np.array([1.0]), np.array([0]))]

        result = kreiss_stability_check(
            interior_weights, interior_offsets, boundary_rows,
            s_grid_params={"s_max": 5.0, "n_radial": 10, "n_imag": 20,
                           "imag_max": 10.0},
        )

        assert result.is_stable is True
        assert result.defective_kappa_detected is False
        assert result.anomalous_points == ()


def _extract_stencil_data_from_D(D, p, tol=1e-10):
    """Extract interior weights/offsets and ALL closure rows from a D matrix.

    Test-local duplicate of brady2d_stability._extract_stencil_data.
    Plan 48 item 0.3: extracts every left-end row of D that differs from
    the (shifted) interior row — compared against the middle row within a
    tolerance — instead of the first p rows, so the analyzed operator
    matches the real closure. Contiguity: the returned block is rows
    0..m-1 where m-1 is the last differing row (interior-matching rows
    inside the span are included verbatim).

    Parameters
    ----------
    D : np.ndarray
        Full N×N differentiation matrix.
    p : int
        Interior half-bandwidth.

    Returns
    -------
    interior_weights, interior_offsets, boundary_rows, row_indices
    """
    n = D.shape[0]
    mid = n // 2
    io = np.arange(-p, p + 1)
    interior_weights = D[mid, mid + io]
    threshold = tol * max(1.0, float(np.max(np.abs(interior_weights))))

    differing = []
    for i in range(mid):
        expected = np.zeros(n)
        cols = i + io
        valid = (cols >= 0) & (cols < n)
        expected[cols[valid]] = interior_weights[valid]
        if not np.all(valid) or np.max(np.abs(D[i, :] - expected)) > threshold:
            differing.append(i)

    assert differing, "no closure rows found at the left end of D"
    m = differing[-1] + 1

    boundary_rows = []
    for i in range(m):
        brow = D[i, :]
        bcols = np.nonzero(np.abs(brow) > 1e-15)[0]
        boundary_rows.append((brow[bcols], bcols))

    return interior_weights, io, boundary_rows, list(range(m))


class TestKreissIntegration:
    """Integration tests using real E4 stencils (41.3h).

    Note on GKS vs eigenvalue instability: the Kreiss determinant test
    assesses boundary-closure stability on a SEMI-INFINITE domain (one
    boundary only). A scheme can be GKS-stable at each boundary
    individually but eigenvalue-unstable on the finite domain (due to
    left-right boundary interaction). The Gaussian eps=0.1 is one such
    case: it passes the GKS Kreiss test but has max Re(eigenvalue(-D_bc)) > 0.
    Its instability is caught at Layer 3 (eigenvalue check), not Layer 2.
    """

    def test_classical_e4_passes(self):
        """E4 classical closure is GKS-stable (Kreiss test passes).

        Plan 48 item 0.3 recalibration: all m=4 differing closure rows are
        now fed to the augmented Kreiss matrix (previously only the first
        p=2, i.e. the wrong operator). The verdict did NOT flip — still
        GKS-stable (witness sigma_min ~2e-5 at |s| ~ 0.1), consistent with
        L3 dense eigenvalues max Re(eig(-D_bc)) < 0 at N in {20, 40, 80}.
        """
        from stencil_gen.phs import build_diff_matrix_rbf

        # Use tension sigma=3.0 as the stable reference (RBF path, known stable)
        D = build_diff_matrix_rbf(n=20, p=2, q=3, epsilon=3.0,
                                   kernel="tension", nu=1, nextra=0)
        interior_weights, interior_offsets, boundary_rows, row_indices = (
            _extract_stencil_data_from_D(D, p=2)
        )

        # All 4 boundary rows differ from the interior for the RBF closure
        assert len(boundary_rows) == 4
        assert row_indices == [0, 1, 2, 3]

        result = kreiss_stability_check(
            interior_weights, interior_offsets, boundary_rows,
            s_grid_params={"s_max": 10.0, "n_radial": 30, "n_imag": 80,
                           "imag_max": 15.0},
            row_indices=row_indices,
        )

        assert result.is_stable is True

    def test_gaussian_known_unstable_passes_gks(self):
        """E4 Gaussian epsilon=0.1 is eigenvalue-unstable but GKS-stable.

        The Gaussian eps=0.1 passes the Kreiss test because its instability
        arises from left-right boundary interaction (caught at Layer 3),
        not from a single-boundary GKS violation. This test documents this
        important distinction.

        Plan 48 item 0.3 recalibration: with all m=4 differing rows fed to
        the augmented Kreiss matrix the verdict did NOT flip — still
        GKS-stable (witness sigma_min ~4e-7 at |s| ~ 0.1), while L3 dense
        eigenvalues confirm the finite-domain instability
        (max Re(eig(-D_bc)) ~ +9.8e-2 at N in {20, 40, 80}).
        """
        from stencil_gen.phs import build_diff_matrix_rbf

        D = build_diff_matrix_rbf(n=20, p=2, q=3, epsilon=0.1,
                                   kernel="gaussian", nu=1, nextra=0)
        interior_weights, interior_offsets, boundary_rows, row_indices = (
            _extract_stencil_data_from_D(D, p=2)
        )
        assert len(boundary_rows) == 4
        assert row_indices == [0, 1, 2, 3]

        result = kreiss_stability_check(
            interior_weights, interior_offsets, boundary_rows,
            s_grid_params={"s_max": 10.0, "n_radial": 30, "n_imag": 80,
                           "imag_max": 15.0},
            row_indices=row_indices,
        )

        # GKS-stable — the instability is NOT a boundary-closure issue
        assert result.is_stable is True

        # But eigenvalue check catches the actual instability
        from stencil_gen.phs import stability_eigenvalue_from_matrix
        max_re = stability_eigenvalue_from_matrix(D)
        assert max_re > 0, "Expected eigenvalue instability for Gaussian eps=0.1"

    def test_consistency_with_heuristic(self):
        """GKS heuristic and Kreiss test are complementary diagnostics.

        The heuristic detects eigenmodes with outgoing group velocity (which
        may have negative real part — damped but outgoing). The Kreiss test
        checks for modes with Re(s) >= 0 that violate the boundary condition.
        They test different aspects and need not be strictly consistent.

        This test verifies that both complete without error on the Gaussian
        eps=0.1 case and documents their relationship.
        """
        from stencil_gen.phs import build_diff_matrix_rbf
        from stencil_gen.group_velocity import gks_group_velocity_check

        n = 20
        D = build_diff_matrix_rbf(n=n, p=2, q=3, epsilon=0.1,
                                   kernel="gaussian", nu=1, nextra=0)
        xi_array = np.linspace(0, np.pi, 1000)
        modes = gks_group_velocity_check(D, xi_array)
        outgoing = [m for m in modes if m.is_outgoing]

        # Heuristic detects an outgoing mode (with negative Re(eigenvalue))
        assert len(outgoing) > 0, "Expected heuristic to find outgoing modes"
        # The outgoing mode has Re(eigenvalue) < 0 (damped), so not a GKS violation
        for m in outgoing:
            assert m.eigenvalue.real < 0, (
                f"Outgoing mode has Re(eigenvalue)={m.eigenvalue.real} >= 0"
            )

    def test_s_equals_zero_godunov_ryabenkii_reduction(self):
        """At s=0, min_singular_value returns inf (admissible-root count != p).

        For the E4 centered stencil at s=0, one root is at kappa=1 (exactly
        on the unit circle, not strictly inside), giving only 1 admissible
        root against the structural count p = L_left = 2 — a root-count
        anomaly, so min_singular_value returns inf (plan 48 items 0.1/0.2;
        after item 0.3 all m=4 differing rows are fed with row_indices, and
        the augmented matrix reports the same count mismatch). This is the
        Godunov-Ryabenkii condition: the boundary closure does not support
        the trivial (constant) mode, which is correct.
        """
        from stencil_gen.phs import build_diff_matrix_rbf

        D = build_diff_matrix_rbf(n=20, p=2, q=3, epsilon=3.0,
                                   kernel="tension", nu=1, nextra=0)
        interior_weights, interior_offsets, boundary_rows, row_indices = (
            _extract_stencil_data_from_D(D, p=2)
        )

        sv = min_singular_value(
            interior_weights, interior_offsets, boundary_rows, s=0.0,
            row_indices=row_indices,
        )
        # At s=0, only 1 admissible root vs p=2 → inf (count mismatch)
        assert sv == np.inf, f"sigma_min(M(0)) = {sv}, expected inf"
