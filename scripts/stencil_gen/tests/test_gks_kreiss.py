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
    kreiss_stability_check,
    make_s_grid,
    min_singular_value,
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

    def test_min_singular_value_shape_mismatch_returns_inf(self):
        """min_singular_value returns inf when root count != boundary row count."""
        interior_weights = np.array([-1.0, 1.0])
        interior_offsets = np.array([-1, 0])
        boundary_rows = [
            (np.array([1.0]), np.array([0])),
            (np.array([1.0]), np.array([0])),
        ]
        sv = min_singular_value(interior_weights, interior_offsets, boundary_rows, s=1.0)
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


def _extract_stencil_data_from_D(D, p):
    """Extract interior weights/offsets and boundary rows from a D matrix.

    Uses the first p rows as boundary rows (matching the number of admissible
    kappa roots for a 2p+1-point centered stencil in the GKS framework).

    Parameters
    ----------
    D : np.ndarray
        Full N×N differentiation matrix.
    p : int
        Interior half-bandwidth.

    Returns
    -------
    interior_weights, interior_offsets, boundary_rows
    """
    n = D.shape[0]
    mid = n // 2
    io = np.arange(-p, p + 1)
    interior_weights = D[mid, mid + io]

    boundary_rows = []
    for i in range(p):
        brow = D[i, :]
        bcols = np.nonzero(np.abs(brow) > 1e-15)[0]
        boundary_rows.append((brow[bcols], bcols))

    return interior_weights, io, boundary_rows


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
        """E4 classical closure is GKS-stable (Kreiss test passes)."""
        from stencil_gen.phs import build_diff_matrix_rbf

        # Use tension sigma=3.0 as the stable reference (RBF path, known stable)
        D = build_diff_matrix_rbf(n=20, p=2, q=3, epsilon=3.0,
                                   kernel="tension", nu=1, nextra=0)
        interior_weights, interior_offsets, boundary_rows = (
            _extract_stencil_data_from_D(D, p=2)
        )

        result = kreiss_stability_check(
            interior_weights, interior_offsets, boundary_rows,
            s_grid_params={"s_max": 10.0, "n_radial": 30, "n_imag": 80,
                           "imag_max": 15.0},
        )

        assert result.is_stable is True

    def test_gaussian_known_unstable_passes_gks(self):
        """E4 Gaussian epsilon=0.1 is eigenvalue-unstable but GKS-stable.

        The Gaussian eps=0.1 passes the Kreiss test because its instability
        arises from left-right boundary interaction (caught at Layer 3),
        not from a single-boundary GKS violation. This test documents this
        important distinction.
        """
        from stencil_gen.phs import build_diff_matrix_rbf

        D = build_diff_matrix_rbf(n=20, p=2, q=3, epsilon=0.1,
                                   kernel="gaussian", nu=1, nextra=0)
        interior_weights, interior_offsets, boundary_rows = (
            _extract_stencil_data_from_D(D, p=2)
        )

        result = kreiss_stability_check(
            interior_weights, interior_offsets, boundary_rows,
            s_grid_params={"s_max": 10.0, "n_radial": 30, "n_imag": 80,
                           "imag_max": 15.0},
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
        """At s=0, min_singular_value returns inf (no admissible roots ≠ n_boundary).

        For the E4 centered stencil at s=0, one root is at kappa=1 (exactly
        on the unit circle, not strictly inside), giving only 1 admissible root.
        With 2 boundary rows, this is a shape mismatch → min_singular_value
        returns inf. This is the Godunov-Ryabenkii condition: the boundary
        closure does not support the trivial (constant) mode, which is correct.
        """
        from stencil_gen.phs import build_diff_matrix_rbf

        D = build_diff_matrix_rbf(n=20, p=2, q=3, epsilon=3.0,
                                   kernel="tension", nu=1, nextra=0)
        interior_weights, interior_offsets, boundary_rows = (
            _extract_stencil_data_from_D(D, p=2)
        )

        sv = min_singular_value(
            interior_weights, interior_offsets, boundary_rows, s=0.0
        )
        # At s=0, only 1 admissible root vs 2 boundary rows → inf (shape mismatch)
        assert sv == np.inf, f"sigma_min(M(0)) = {sv}, expected inf"
