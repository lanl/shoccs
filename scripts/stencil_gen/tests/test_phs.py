"""Tests for PHS+poly stencil derivation (Phase 29)."""

import json
from pathlib import Path

import pytest
from sympy import Rational, S, cancel

from stencil_gen.phs import (
    _tension_kernel_eval,
    _tension_kernel_deriv,
    build_diff_matrix_rbf_penalty,
    cut_cell_weights,
    phs_stencil_weights,
    uniform_boundary_weights,
    uniform_boundary_weights_rbf,
    uniform_boundary_weights_tension,
    uniform_interior_weights,
    uniform_interior_weights_rbf,
    uniform_interior_weights_tension,
)


# ---------------------------------------------------------------------------
# 29.1a: Core PHS solver
# ---------------------------------------------------------------------------


class TestPHSCore:
    """Basic tests for phs_stencil_weights."""

    def test_3pt_first_deriv_centered(self):
        """3-point centered first derivative should give [-1/2, 0, 1/2]."""
        # 3 points, centered at 0, degree q=2, any k >= 1
        points = [Rational(-1), Rational(0), Rational(1)]
        w = phs_stencil_weights(points, Rational(0), nu=1, q=2, k=2)
        assert w == [Rational(-1, 2), S.Zero, Rational(1, 2)]

    def test_3pt_second_deriv_centered(self):
        """3-point centered second derivative should give [1, -2, 1]."""
        points = [Rational(-1), Rational(0), Rational(1)]
        w = phs_stencil_weights(points, Rational(0), nu=2, q=2, k=2)
        assert w == [S.One, Rational(-2), S.One]

    def test_2pt_first_deriv(self):
        """2-point first derivative: [-1, 1] (forward difference)."""
        points = [Rational(0), Rational(1)]
        w = phs_stencil_weights(points, Rational(0), nu=1, q=1, k=1)
        assert w == [Rational(-1), Rational(1)]

    def test_polynomial_exactness(self):
        """Stencil should be exact for polynomials up to degree q."""
        points = [Rational(j) for j in range(6)]
        for q in [1, 2, 3, 4]:
            w = phs_stencil_weights(points, Rational(0), nu=1, q=q, k=max(2, q))
            # Check: sum_j w_j * x_j^d = d * 0^(d-1) for d = 0..q
            for d in range(q + 1):
                actual = sum(wj * xj**d for wj, xj in zip(w, points))
                expected = d * Rational(0) ** max(0, d - 1) if d >= 1 else S.Zero
                assert cancel(actual - expected) == 0, (
                    f"q={q}, d={d}: got {actual}, expected {expected}"
                )

    def test_weights_sum_to_zero_for_first_deriv(self):
        """First derivative weights should sum to 0 (exact for constants)."""
        points = [Rational(j) for j in range(5)]
        for k in [2, 3, 4]:
            w = phs_stencil_weights(points, Rational(1), nu=1, q=3, k=k)
            assert cancel(sum(w)) == 0, f"k={k}: weights don't sum to 0"


# ---------------------------------------------------------------------------
# 29.1b + 29.2c: Uniform grid wrappers + interior verification
# ---------------------------------------------------------------------------


class TestPHSInterior:
    """Verify PHS interior stencils match classical FD."""

    def test_e2_interior_high_k(self):
        """E2 interior (p=1, nu=1): [-1/2, 0, 1/2] for any k >= 2."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        classical = full_gamma_array(derive_interior(0, 1, 1))
        for k in [2, 3, 5]:
            phs = uniform_interior_weights(p=1, nu=1, k=k, q=2)
            for j in range(len(classical)):
                assert cancel(phs[j] - classical[j]) == 0, (
                    f"k={k}, j={j}: PHS={phs[j]}, classical={classical[j]}"
                )

    def test_e4_interior_high_k(self):
        """E4 interior (p=2, nu=1): [1/12, -2/3, 0, 2/3, -1/12] for high k."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        classical = full_gamma_array(derive_interior(0, 2, 1))
        for k in [3, 5]:
            phs = uniform_interior_weights(p=2, nu=1, k=k, q=4)
            for j in range(len(classical)):
                assert cancel(phs[j] - classical[j]) == 0, (
                    f"k={k}, j={j}: PHS={phs[j]}, classical={classical[j]}"
                )

    def test_e2_interior_nu2(self):
        """E2 second derivative interior (p=1, nu=2): [1, -2, 1]."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        classical = full_gamma_array(derive_interior(0, 1, 2))
        phs = uniform_interior_weights(p=1, nu=2, k=2, q=2)
        for j in range(len(classical)):
            assert cancel(phs[j] - classical[j]) == 0


# ---------------------------------------------------------------------------
# 29.2a: E2_1 boundary comparison
# ---------------------------------------------------------------------------


class TestPHSvsE2Boundary:
    """Compare PHS boundary stencils against E2_1."""

    def test_taylor_accuracy(self):
        """PHS E2_1 boundary rows should have order q=1 accuracy."""
        # E2_1: p=1, q=1, t=4, r=3
        for i in range(3):
            for k in [2, 3]:
                w = uniform_boundary_weights(i, t=4, nu=1, k=k, q=1)
                # Check: exact for f(x) = 1 and f(x) = x
                pts = [Rational(j) for j in range(4)]
                # d/dx(1) = 0
                assert cancel(sum(w)) == 0, f"row {i}, k={k}: not exact for constants"
                # d/dx(x) = 1 at x=i
                actual = sum(wj * xj for wj, xj in zip(w, pts))
                assert cancel(actual - 1) == 0, f"row {i}, k={k}: not exact for x"

    def test_extract_implied_alpha(self):
        """Extract the implied alpha from PHS E2_1 boundary stencils."""
        from stencil_gen.temo import E2_1, derive_uniform_boundary_for_temo

        uniform = derive_uniform_boundary_for_temo(E2_1)
        B_u = uniform.B_u
        alpha_sym = uniform.alpha_symbols

        print("\n=== E2_1 PHS vs Symbolic Boundary Stencils ===")
        for i in range(B_u.rows):
            for k in [2, 3, 4]:
                w_phs = uniform_boundary_weights(i, t=4, nu=1, k=k, q=1)
                # The symbolic stencil has form B_u[i,j] = a_j + b_j * alpha
                # The PHS stencil has specific numeric values.
                # Extract alpha by solving: B_u[i, j_free] = w_phs[j_free]
                # where j_free is the free column (last column in our convention)
                if len(alpha_sym) > 0 and i < B_u.rows:
                    row_syms = B_u[i, :].free_symbols
                    if row_syms:
                        # Find the alpha that appears in this row
                        alpha_in_row = sorted(row_syms, key=str)
                        alpha = alpha_in_row[0]
                        # Solve for alpha from the last column
                        from sympy import solve

                        for j in range(B_u.cols):
                            expr = B_u[i, j]
                            if alpha in expr.free_symbols:
                                sol = solve(expr - w_phs[j], alpha)
                                if sol:
                                    print(f"  Row {i}, k={k}: alpha = {sol[0]}")
                                    break


# ---------------------------------------------------------------------------
# 29.2b: E4_1 boundary comparison
# ---------------------------------------------------------------------------


class TestPHSvsE4Boundary:
    """Compare PHS boundary stencils against E4_1."""

    def test_taylor_accuracy(self):
        """PHS E4_1 boundary rows should have order q=3 accuracy."""
        # E4_1: p=2, q=3, t=6, r=4
        for i in range(4):
            w = uniform_boundary_weights(i, t=6, nu=1, k=3, q=3)
            pts = [Rational(j) for j in range(6)]
            # Exact for polynomials up to degree 3
            for d in range(4):
                actual = sum(wj * xj**d for wj, xj in zip(w, pts))
                if d == 0:
                    expected = S.Zero
                elif d == 1:
                    expected = S.One
                else:
                    expected = d * Rational(i) ** (d - 1)
                assert cancel(actual - expected) == 0, (
                    f"row {i}, d={d}: got {actual}, expected {expected}"
                )

    def test_extract_implied_alphas(self):
        """Extract implied alpha values from PHS E4_1 boundary stencils."""
        from stencil_gen.temo import E4_1, build_uniform_for_mathematica

        uniform = build_uniform_for_mathematica(E4_1)
        B_u = uniform.B_u

        print("\n=== E4_1 PHS vs Symbolic Boundary Stencils ===")
        for k in [2, 3, 4, 5]:
            print(f"\n  k={k}:")
            for i in range(B_u.rows):
                w_phs = uniform_boundary_weights(i, t=6, nu=1, k=k, q=3)
                # Print the PHS weights
                print(f"    Row {i}: {[float(cancel(w)) for w in w_phs]}")


# ---------------------------------------------------------------------------
# 29.5b + 29.5c: Gaussian/Multiquadric RBF convenience wrappers & tests
# ---------------------------------------------------------------------------

import numpy as np

from stencil_gen.phs import (
    build_diff_matrix_mixed_epsilon,
    build_diff_matrix_rbf,
    max_real_eigenvalue,
    stability_eigenvalue,
    stability_eigenvalue_from_matrix,
)

# Floating-point eigenvalue solvers return tiny positive real parts (~1e-14)
# for genuinely stable operators.  Use this threshold to distinguish true
# instability from numerical noise.
STABILITY_TOL = 1e-10


# ---------------------------------------------------------------------------
# 29.6a: Differentiation matrix builder
# ---------------------------------------------------------------------------


class TestBuildDiffMatrixRBF:
    """Tests for build_diff_matrix_rbf."""

    def test_matrix_shape(self):
        """Matrix should be n×n."""
        for n in [20, 40]:
            D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=1.0, nextra=1)
            assert D.shape == (n, n)

    def test_interior_column_sums_zero(self):
        """Interior rows should have column sums of 0 (first derivative)."""
        n = 30
        # E2: p=1, q=1, nextra=1 → r=3 boundary rows
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=1.0, nextra=1)
        r = 3
        # Each interior row should sum to 0
        for i in range(r, n - r):
            row_sum = np.sum(D[i, :])
            assert abs(row_sum) < 1e-14, f"Interior row {i} sum = {row_sum}"

    def test_boundary_rows_nonzero(self):
        """Boundary rows should have nonzero entries."""
        n = 20
        D = build_diff_matrix_rbf(n, p=2, q=3, epsilon=1.0)
        # Left boundary row 0 should have nonzero entries in first t=6 columns
        assert np.any(D[0, :6] != 0)
        # Right boundary row n-1 should have nonzero entries in last t=6 columns
        assert np.any(D[-1, -6:] != 0)

    def test_antisymmetry_first_deriv(self):
        """Right boundary should be antisymmetric reflection of left for nu=1."""
        n = 20
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=2.0, nextra=1)
        r = 3
        t = 4
        for i in range(r):
            left_row = D[i, :t]
            right_row = D[n - 1 - i, n - t:][::-1]
            np.testing.assert_allclose(right_row, -left_row, atol=1e-14)

    def test_polynomial_reproduction(self):
        """D applied to x should give all 1s (exact for linear)."""
        n = 30
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=1.0, nextra=1)
        x = np.arange(n, dtype=float)
        result = D @ x
        np.testing.assert_allclose(result, 1.0, atol=1e-12)

    def test_nu2_dimensions_match_temo(self):
        """build_diff_matrix_rbf nu=2 dimensions should match temo.compute_dimensions."""
        from stencil_gen.temo import compute_dimensions

        # E2_2: p=1, q=1, nextra=0, nu=2 → t=3, r=2
        dims = compute_dimensions(p=1, q=1, s=0, nextra=0, nu=2)
        n = 20
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=1.0, nu=2, nextra=0)
        # Boundary stencil width = t: row 0 should have nonzero entries in cols 0..t-1
        nonzero_cols_row0 = np.where(np.abs(D[0, :]) > 1e-15)[0]
        assert nonzero_cols_row0[-1] <= dims.t - 1, (
            f"Row 0 extends to col {nonzero_cols_row0[-1]}, expected max {dims.t - 1}"
        )
        # Number of boundary rows per side = r
        # Interior rows should use the centered stencil, not boundary
        # Row r should be an interior row (centered stencil)
        r = dims.r
        center_col = r  # interior row at index r is centered at column r
        assert D[r, center_col] != 0, f"Interior row {r} should have centered stencil"
        # Row r-1 should be a boundary row (one-sided stencil)
        assert D[r - 1, 0] != 0, f"Boundary row {r-1} should use left-boundary stencil"

    def test_nu2_polynomial_reproduction(self):
        """D (nu=2) applied to x should give all 0s (exact for linear, q=1)."""
        n = 30
        # E2_2 params: p=1, q=1, nextra=0 → boundary exact for poly deg ≤ 1
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=1.0, nu=2, nextra=0)
        x = np.arange(n, dtype=float)
        # D^2(x) = 0 everywhere for q >= 1
        result = D @ x
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_nu2_polynomial_reproduction_higher_q(self):
        """D (nu=2) applied to x^2 should give 2s with q=3 (E4_2 params)."""
        n = 30
        # E4_2: p=2, q=3, nextra=0 → boundary exact for poly deg ≤ 3
        D = build_diff_matrix_rbf(n, p=2, q=3, epsilon=1.0, nu=2, nextra=0)
        x = np.arange(n, dtype=float)
        result = D @ (x**2)
        np.testing.assert_allclose(result, 2.0, atol=1e-10)

    def test_nu2_symmetry_second_deriv(self):
        """Right boundary should be symmetric reflection of left for nu=2."""
        n = 20
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=2.0, nu=2, nextra=0)
        from stencil_gen.temo import compute_dimensions

        dims = compute_dimensions(p=1, q=1, s=0, nextra=0, nu=2)
        r, t = dims.r, dims.t
        for i in range(r):
            left_row = D[i, :t]
            right_row = D[n - 1 - i, n - t:][::-1]
            # nu=2 (even) → symmetric reflection: sign = (-1)^2 = +1
            np.testing.assert_allclose(right_row, left_row, atol=1e-14)


# ---------------------------------------------------------------------------
# 30.2a: Diff matrix builder with tension kernel
# ---------------------------------------------------------------------------


class TestBuildDiffMatrixTension:
    """Tests for build_diff_matrix_rbf with kernel='tension'."""

    def test_matrix_shape(self):
        """Tension diff matrix should be n×n."""
        for n in [20, 40]:
            D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=2.0,
                                      kernel="tension", nextra=1)
            assert D.shape == (n, n)

    def test_interior_column_sums_zero(self):
        """Interior rows should sum to 0 (first derivative)."""
        n = 30
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=3.0,
                                  kernel="tension", nextra=1)
        r = 3  # q + 1 + nextra = 1 + 1 + 1
        for i in range(r, n - r):
            row_sum = np.sum(D[i, :])
            assert abs(row_sum) < 1e-14, f"Interior row {i} sum = {row_sum}"

    def test_sigma_zero_matches_phs(self):
        """At σ=0, tension diff matrix should match PHS k=2."""
        n = 20
        D_tension = build_diff_matrix_rbf(n, p=1, q=1, epsilon=0.0,
                                          kernel="tension", nextra=1)
        D_phs = build_diff_matrix_rbf(n, p=1, q=1, epsilon=1.0,
                                      kernel="gaussian", nextra=1)
        # Build reference PHS k=2 matrix manually
        from stencil_gen.interior import derive_interior, full_gamma_array

        r = 3  # q + 1 + nextra
        t = 4  # p + q + 1 + nextra
        D_ref = np.zeros((n, n))
        for i in range(r):
            w = [float(x) for x in phs_stencil_weights(
                [Rational(j) for j in range(t)], Rational(i), 1, 1, k=2)]
            for j in range(t):
                D_ref[i, j] = w[j]
        interior_w = [float(c) for c in full_gamma_array(derive_interior(0, 1, 1))]
        for i in range(r, n - r):
            for k_idx, j in enumerate(range(i - 1, i + 2)):
                D_ref[i, j] = interior_w[k_idx]
        for i in range(r):
            w = [float(x) for x in phs_stencil_weights(
                [Rational(j) for j in range(t)], Rational(i), 1, 1, k=2)]
            row = n - 1 - i
            for j in range(t):
                D_ref[row, n - 1 - j] = -w[j]

        np.testing.assert_allclose(D_tension, D_ref, atol=1e-13,
                                   err_msg="Tension at σ=0 matrix ≠ PHS k=2 matrix")

    def test_polynomial_reproduction(self):
        """D applied to x should give all 1s (exact for linear)."""
        n = 30
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=5.0,
                                  kernel="tension", nextra=1)
        x = np.arange(n, dtype=float)
        result = D @ x
        np.testing.assert_allclose(result, 1.0, atol=1e-12)

    def test_antisymmetry_first_deriv(self):
        """Right boundary should be antisymmetric reflection of left for nu=1."""
        n = 20
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=3.0,
                                  kernel="tension", nextra=1)
        r = 3
        t = 4
        for i in range(r):
            left_row = D[i, :t]
            right_row = D[n - 1 - i, n - t:][::-1]
            np.testing.assert_allclose(right_row, -left_row, atol=1e-14)

    def test_nu2_polynomial_reproduction(self):
        """D (nu=2) applied to x should give all 0s (exact for linear)."""
        n = 30
        D = build_diff_matrix_rbf(n, p=1, q=1, epsilon=3.0,
                                  kernel="tension", nu=2, nextra=0)
        x = np.arange(n, dtype=float)
        result = D @ x
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_eigenvalue_finite(self):
        """max_real_eigenvalue should return a finite float for tension kernel."""
        result = max_real_eigenvalue(20, p=1, q=1, epsilon=2.0,
                                    kernel="tension", nextra=1)
        assert np.isfinite(result), f"Non-finite max Re(λ) = {result}"

    def test_mixed_epsilon_tension(self):
        """build_diff_matrix_mixed_epsilon should work with tension kernel."""
        n = 20
        r = 3  # q + 1 + nextra for p=1, q=1, nextra=1
        sigmas = [1.0, 2.0, 3.0]
        D = build_diff_matrix_mixed_epsilon(n, p=1, q=1, epsilons=sigmas,
                                            kernel="tension", nextra=1)
        assert D.shape == (n, n)
        # Polynomial reproduction: D @ x = 1
        x = np.arange(n, dtype=float)
        result = D @ x
        np.testing.assert_allclose(result, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 29.6b: Max real eigenvalue diagnostic
# ---------------------------------------------------------------------------


class TestMaxRealEigenvalue:
    """Tests for max_real_eigenvalue."""

    def test_interior_only_pure_imaginary(self):
        """Interior-only (periodic) FD matrix should have max Re(λ) ≈ 0.

        Build a matrix with ALL rows using classical interior stencils
        (wrapping around periodically).  This is equivalent to the circulant
        interior matrix which has purely imaginary eigenvalues.
        """
        n = 40
        p = 2
        from stencil_gen.interior import derive_interior, full_gamma_array

        interior_coeffs = derive_interior(0, p, 1)
        interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

        D = np.zeros((n, n))
        for i in range(n):
            for k_idx, offset in enumerate(range(-p, p + 1)):
                j = (i + offset) % n  # periodic wrapping
                D[i, j] = interior_w[k_idx]

        eigvals = np.linalg.eigvals(D)
        max_re = float(np.max(np.real(eigvals)))
        assert abs(max_re) < 1e-12, f"Periodic interior max Re(λ) = {max_re}"

    def test_returns_float(self):
        """max_real_eigenvalue should return a float."""
        result = max_real_eigenvalue(20, p=1, q=1, epsilon=1.0, nextra=1)
        assert isinstance(result, float)


class TestGaussianRBF:
    """Tests for Gaussian and Multiquadric RBF kernels."""

    def test_polynomial_exactness(self):
        """Gaussian RBF stencils should be exact for polynomials up to degree q."""
        for epsilon in [0.5, 1.0, 3.0]:
            for q in [1, 2, 3]:
                # Use t = q + 3 points (enough for the augmented system)
                t = q + 3
                w = uniform_boundary_weights_rbf(i=0, t=t, nu=1, q=q, epsilon=epsilon)
                pts = list(range(t))
                for d in range(q + 1):
                    actual = sum(wj * xj**d for wj, xj in zip(w, pts))
                    expected = d * 0 ** max(0, d - 1) if d >= 1 else 0.0
                    assert abs(actual - expected) < 1e-12, (
                        f"eps={epsilon}, q={q}, d={d}: got {actual}, expected {expected}"
                    )

    def test_interior_matches_classical(self):
        """Gaussian interior weights should match classical FD for all epsilon.

        The polynomial augmentation forces polynomial reproduction, so the
        interior (centered) weights must equal the classical FD coefficients
        regardless of the RBF shape parameter.
        """
        from stencil_gen.interior import derive_interior, full_gamma_array

        # E2: p=1, q=2
        classical_e2 = full_gamma_array(derive_interior(0, 1, 1))
        for epsilon in [0.1, 1.0, 5.0]:
            w = uniform_interior_weights_rbf(p=1, nu=1, q=2, epsilon=epsilon)
            for j in range(len(classical_e2)):
                assert abs(w[j] - float(classical_e2[j])) < 1e-12, (
                    f"E2 eps={epsilon}, j={j}: RBF={w[j]}, classical={classical_e2[j]}"
                )

        # E4: p=2, q=4
        classical_e4 = full_gamma_array(derive_interior(0, 2, 1))
        for epsilon in [0.1, 1.0, 5.0]:
            w = uniform_interior_weights_rbf(p=2, nu=1, q=4, epsilon=epsilon)
            for j in range(len(classical_e4)):
                assert abs(w[j] - float(classical_e4[j])) < 1e-12, (
                    f"E4 eps={epsilon}, j={j}: RBF={w[j]}, classical={classical_e4[j]}"
                )

    def test_weights_sum_to_zero(self):
        """First derivative weights should sum to 0 (exact for constants)."""
        for kernel in ["gaussian", "multiquadric"]:
            for epsilon in [0.5, 1.0, 3.0]:
                # Interior
                w = uniform_interior_weights_rbf(
                    p=2, nu=1, q=3, epsilon=epsilon, kernel=kernel
                )
                assert abs(sum(w)) < 1e-12, (
                    f"{kernel} eps={epsilon} interior: sum={sum(w)}"
                )
                # Boundary
                w = uniform_boundary_weights_rbf(
                    i=0, t=6, nu=1, q=3, epsilon=epsilon, kernel=kernel
                )
                assert abs(sum(w)) < 1e-12, (
                    f"{kernel} eps={epsilon} boundary: sum={sum(w)}"
                )

    def test_small_epsilon_interior_matches_polynomial(self):
        """As epsilon -> 0, interior Gaussian RBF weights approach polynomial FD.

        For centered interior stencils where 2p+1 = q+1, the polynomial
        augmentation fully determines the weights, so the flat limit is
        well-defined and equals classical FD.  For over-determined boundary
        stencils (t > q+1) the flat limit is ill-conditioned, so we only
        test interior convergence here.
        """
        from stencil_gen.interior import derive_interior, full_gamma_array

        # E4: p=2, 5 points, q=4 (5 poly terms = n, so system is determined)
        classical = full_gamma_array(derive_interior(0, 2, 1))
        ref = [float(c) for c in classical]

        # As epsilon decreases, should approach classical
        for epsilon in [2.0, 1.0, 0.5]:
            w = uniform_interior_weights_rbf(p=2, nu=1, q=4, epsilon=epsilon)
            err = max(abs(w[j] - ref[j]) for j in range(len(ref)))
            assert err < 1e-10, f"eps={epsilon}: error {err} too large"

    def test_boundary_weights_bounded(self):
        """Boundary weights remain bounded across a range of epsilon values."""
        for epsilon in [0.5, 1.0, 2.0, 5.0]:
            for kernel in ["gaussian", "multiquadric"]:
                w = uniform_boundary_weights_rbf(
                    i=0, t=6, nu=1, q=3, epsilon=epsilon, kernel=kernel
                )
                assert all(abs(wj) < 100 for wj in w), (
                    f"{kernel} eps={epsilon}: weights unbounded: {w}"
                )

    def test_multiquadric_polynomial_exactness(self):
        """Multiquadric RBF stencils should be exact for polynomials up to degree q."""
        for epsilon in [0.5, 1.0, 3.0]:
            q = 3
            t = q + 3
            w = uniform_boundary_weights_rbf(
                i=1, t=t, nu=1, q=q, epsilon=epsilon, kernel="multiquadric"
            )
            pts = list(range(t))
            for d in range(q + 1):
                actual = sum(wj * xj**d for wj, xj in zip(w, pts))
                expected = d * 1 ** max(0, d - 1) if d >= 1 else 0.0
                assert abs(actual - expected) < 1e-12, (
                    f"MQ eps={epsilon}, d={d}: got {actual}, expected {expected}"
                )



# ---------------------------------------------------------------------------
# 30.1d: Tension spline kernel tests
# ---------------------------------------------------------------------------


class TestTensionSpline:
    """Tests for the tension spline kernel φ(r;σ) = σ|r| - 1 + exp(-σ|r|)."""

    def test_sigma_zero_matches_phs_k2(self):
        """At very small σ, tension boundary weights ≈ PHS k=2 weights."""
        # E2 boundary: p=1, q=1, t=3, row i=0
        phs_w = uniform_boundary_weights(0, 3, nu=1, k=2, q=1)
        phs_w_float = [float(w) for w in phs_w]

        # Tension with very small sigma should approach PHS k=2
        tension_w = uniform_boundary_weights_tension(0, 3, nu=1, q=1, sigma=1e-6)

        np.testing.assert_allclose(tension_w, phs_w_float, atol=1e-6,
                                   err_msg="Tension at σ≈0 should match PHS k=2")

    def test_polynomial_exactness(self):
        """Tension stencil should be exact for polynomials up to degree q."""
        for q in [1, 2, 3]:
            t = q + 3  # enough points
            sigma = 2.0
            for i in range(min(2, t)):
                w = uniform_boundary_weights_tension(i, t, nu=1, q=q, sigma=sigma)
                pts = list(range(t))
                for d in range(q + 1):
                    # sum_j w_j * x_j^d should equal d * i^(d-1) for d >= 1, 0 for d=0
                    actual = sum(wj * xj**d for wj, xj in zip(w, pts))
                    expected = d * i ** max(0, d - 1) if d >= 1 else 0.0
                    np.testing.assert_allclose(
                        actual, expected, atol=1e-10,
                        err_msg=f"q={q}, i={i}, d={d}: poly exactness failed"
                    )

    def test_weights_sum_to_zero(self):
        """First derivative weights should sum to 0 (exact for constants)."""
        for sigma in [0.5, 2.0, 10.0]:
            w = uniform_boundary_weights_tension(0, 4, nu=1, q=1, sigma=sigma)
            np.testing.assert_allclose(
                sum(w), 0.0, atol=1e-12,
                err_msg=f"σ={sigma}: weights don't sum to 0"
            )

    def test_kernel_symmetry(self):
        """φ(r;σ) = φ(-r;σ) — kernel is an even function."""
        for sigma in [0.1, 1.0, 5.0, 20.0]:
            for r in [0.5, 1.0, 2.5, 7.0]:
                val_pos = _tension_kernel_eval(r, sigma)
                val_neg = _tension_kernel_eval(-r, sigma)
                assert abs(val_pos - val_neg) < 1e-14, (
                    f"σ={sigma}, r={r}: φ(r)={val_pos} ≠ φ(-r)={val_neg}"
                )

    def test_interior_matches_classical(self):
        """Interior tension weights match classical FD for all σ."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        classical = [float(c) for c in full_gamma_array(derive_interior(0, 1, 1))]

        for sigma in [0.01, 1.0, 5.0, 20.0]:
            tension_w = uniform_interior_weights_tension(p=1, nu=1, q=2, sigma=sigma)
            np.testing.assert_allclose(
                tension_w, classical, atol=1e-10,
                err_msg=f"σ={sigma}: interior weights differ from classical"
            )

    def test_numerical_stability_large_sigma(self):
        """No overflow for σ up to 50 on unit grid."""
        for sigma in [10.0, 25.0, 50.0]:
            w = uniform_boundary_weights_tension(0, 4, nu=1, q=1, sigma=sigma)
            assert all(np.isfinite(w)), (
                f"σ={sigma}: non-finite weights {w}"
            )

    def test_kernel_positive_for_nonzero_r(self):
        """φ(r;σ) > 0 for r ≠ 0 and σ > 0."""
        for sigma in [0.1, 1.0, 5.0, 20.0]:
            for r in [0.1, 0.5, 1.0, 3.0, 10.0]:
                val = _tension_kernel_eval(r, sigma)
                assert val > 0, f"σ={sigma}, r={r}: φ={val} should be positive"

    def test_kernel_zero_at_origin(self):
        """φ(0;σ) = 0 for all σ."""
        for sigma in [0.0, 0.1, 1.0, 10.0]:
            assert _tension_kernel_eval(0.0, sigma) == 0.0

    def test_d1_antisymmetric(self):
        """D¹φ is an odd function: D¹φ(-r) = -D¹φ(r)."""
        for sigma in [0.5, 2.0, 10.0]:
            for r in [0.5, 1.0, 3.0]:
                dp = _tension_kernel_deriv(r, 1, sigma)
                dm = _tension_kernel_deriv(-r, 1, sigma)
                assert abs(dp + dm) < 1e-13, (
                    f"σ={sigma}, r={r}: D¹φ(r)+D¹φ(-r) = {dp+dm}"
                )

    def test_d2_symmetric(self):
        """D²φ is an even function: D²φ(-r) = D²φ(r)."""
        for sigma in [0.5, 2.0, 10.0]:
            for r in [0.5, 1.0, 3.0]:
                dp = _tension_kernel_deriv(r, 2, sigma)
                dm = _tension_kernel_deriv(-r, 2, sigma)
                assert abs(dp - dm) < 1e-13, (
                    f"σ={sigma}, r={r}: D²φ(r)-D²φ(-r) = {dp-dm}"
                )

    def test_taylor_matches_direct(self):
        """Taylor branch (z<2) matches direct evaluation at the boundary z≈2."""
        # Test at z = 1.99 (Taylor) vs z = 2.01 (direct) — should be close
        sigma = 2.0
        r = 1.0  # z = sigma*r = 2.0, right at boundary
        # Evaluate slightly on each side
        r_lo = 0.99  # z = 1.98, Taylor path
        r_hi = 1.01  # z = 2.02, direct path
        phi_lo = _tension_kernel_eval(r_lo, sigma)
        phi_hi = _tension_kernel_eval(r_hi, sigma)
        # They should be close (continuous function)
        expected_diff = abs(phi_hi - phi_lo)
        assert expected_diff < 0.1, (
            f"φ discontinuous near Taylor/direct boundary: {phi_lo} vs {phi_hi}"
        )

        # Also check derivative continuity
        for nu in [1, 2]:
            d_lo = _tension_kernel_deriv(r_lo, nu, sigma)
            d_hi = _tension_kernel_deriv(r_hi, nu, sigma)
            assert abs(d_hi - d_lo) < 0.2, (
                f"D{nu}φ discontinuous near boundary: {d_lo} vs {d_hi}"
            )

    def test_sigma_exactly_zero_dispatches_to_phs(self):
        """At σ=0.0, tension wrappers must not crash and must match PHS k=2."""
        # Boundary weights: E2 layout (p=1, q=1, t=3, row i=0)
        phs_w = uniform_boundary_weights(0, 3, nu=1, k=2, q=1)
        phs_w_float = [float(w) for w in phs_w]

        tension_w = uniform_boundary_weights_tension(0, 3, nu=1, q=1, sigma=0.0)
        np.testing.assert_allclose(
            tension_w, phs_w_float, atol=1e-14,
            err_msg="Tension at σ=0 should exactly match PHS k=2",
        )

        # Interior weights: p=1, q=1
        phs_int = uniform_interior_weights(1, nu=1, k=2, q=1)
        phs_int_float = [float(w) for w in phs_int]

        tension_int = uniform_interior_weights_tension(1, nu=1, q=1, sigma=0.0)
        np.testing.assert_allclose(
            tension_int, phs_int_float, atol=1e-14,
            err_msg="Interior tension at σ=0 should exactly match PHS k=2",
        )

    def test_nu2_polynomial_exactness(self):
        """Second-derivative weights reproduce D² x^d exactly for d ≤ q."""
        for q in [2, 3]:
            t = q + 4  # enough points for a well-determined system
            sigma = 3.0
            for i in range(t):
                w = uniform_boundary_weights_tension(i, t, nu=2, q=q, sigma=sigma)
                pts = np.arange(t, dtype=float)
                for d in range(q + 1):
                    # D² x^d at x=i should be d*(d-1)*i^(d-2) for d>=2, else 0
                    got = sum(wj * xj**d for wj, xj in zip(w, pts))
                    if d >= 2:
                        expected = d * (d - 1) * float(i) ** (d - 2)
                    else:
                        expected = 0.0
                    assert abs(got - expected) < 1e-10, (
                        f"nu=2 poly exactness failed: q={q}, i={i}, d={d}, "
                        f"got={got}, expected={expected}"
                    )


# ---------------------------------------------------------------------------
# 30.3a: Soft conservation penalty
# ---------------------------------------------------------------------------


class TestConservationPenalty:
    """Tests for build_diff_matrix_rbf_penalty (Phase 30.3a).

    Verifies that the penalty-augmented RBF-FD system:
    1. At γ=0, recovers the standard RBF weights exactly.
    2. As γ→∞, approaches conservation-enforced weights (zero column sums).
    3. Preserves polynomial exactness at all γ values.
    """

    # E2_1 parameters
    E2_P, E2_Q, E2_NEXTRA, E2_NU = 1, 1, 1, 1
    # E4_1 parameters
    E4_P, E4_Q, E4_NEXTRA, E4_NU = 2, 3, 0, 1

    def _conservation_deficit(self, D):
        """Max absolute column sum of D."""
        return float(np.max(np.abs(np.sum(D, axis=0))))

    def _polynomial_reproduction_error(self, D, n, nu, q):
        """Max error in D applied to polynomials x^d for d=0..q.

        The differentiation matrix D is built for unit-spacing grid
        {0, 1, ..., n-1}, so f_j = j^d and (Df)_i should equal
        d!/(d-nu)! * i^{d-nu}.
        """
        x = np.arange(n, dtype=float)
        max_err = 0.0
        for d in range(q + 1):
            f = x**d
            Df = D @ f
            if d >= nu:
                coeff = 1.0
                for j in range(nu):
                    coeff *= (d - j)
                exact = coeff * x ** (d - nu)
            else:
                exact = np.zeros(n)
            err = np.max(np.abs(Df - exact))
            max_err = max(max_err, err)
        return max_err

    def test_gamma_zero_matches_standard_e2(self):
        """γ=0 penalty matrix is identical to standard RBF matrix (E2)."""
        n, sigma = 40, 6.0
        p, q, nextra, nu = self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU

        D_std = build_diff_matrix_rbf(n, p, q, sigma, "tension", nu, nextra)
        D_pen = build_diff_matrix_rbf_penalty(
            n, p, q, sigma, "tension", nu, nextra, gamma=0.0
        )

        np.testing.assert_allclose(D_pen, D_std, atol=1e-15)

    def test_gamma_zero_matches_standard_e4(self):
        """γ=0 penalty matrix is identical to standard RBF matrix (E4)."""
        n, sigma = 40, 37.0
        p, q, nextra, nu = self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU

        D_std = build_diff_matrix_rbf(n, p, q, sigma, "tension", nu, nextra)
        D_pen = build_diff_matrix_rbf_penalty(
            n, p, q, sigma, "tension", nu, nextra, gamma=0.0
        )

        np.testing.assert_allclose(D_pen, D_std, atol=1e-15)

    def test_conservation_improves_with_gamma_e2(self):
        """Conservation deficit decreases as γ increases (E2).

        Full conservation is NOT achievable while maintaining polynomial
        exactness: the null space of P has dimension t-(q+1) per row,
        but all rows share the same null space, so the effective column-sum
        freedom is only t-(q+1) dimensions vs t conservation equations.
        The penalty reduces the deficit to a fundamental limit set by the
        polynomial-exactness / conservation trade-off.
        """
        n, sigma = 40, 6.0
        p, q, nextra, nu = self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU

        gammas = [0, 1, 10, 100, 1000, 1e6]
        deficits = []
        for g in gammas:
            D = build_diff_matrix_rbf_penalty(
                n, p, q, sigma, "tension", nu, nextra, gamma=g
            )
            deficits.append(self._conservation_deficit(D))

        print("\n  E2 conservation deficit vs γ:")
        for g, d in zip(gammas, deficits):
            print(f"    γ={g:>10.0f}  deficit={d:.6e}")

        # Deficit should decrease overall
        assert deficits[-1] < deficits[0], (
            f"Large γ ({deficits[-1]:.6e}) should reduce deficit vs γ=0 ({deficits[0]:.6e})"
        )
        # At large γ, deficit converges to a fundamental limit (rank-limited)
        assert abs(deficits[-1] - deficits[-2]) / deficits[-2] < 0.01, (
            "Deficit should converge at large γ"
        )

    def test_conservation_improves_with_gamma_e4(self):
        """Conservation deficit decreases as γ increases (E4)."""
        n, sigma = 40, 37.0
        p, q, nextra, nu = self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU

        gammas = [0, 1, 10, 100, 1000, 1e6]
        deficits = []
        for g in gammas:
            D = build_diff_matrix_rbf_penalty(
                n, p, q, sigma, "tension", nu, nextra, gamma=g
            )
            deficits.append(self._conservation_deficit(D))

        print("\n  E4 conservation deficit vs γ:")
        for g, d in zip(gammas, deficits):
            print(f"    γ={g:>10.0f}  deficit={d:.6e}")

        assert deficits[-1] < deficits[0], (
            f"Large γ ({deficits[-1]:.6e}) should reduce deficit vs γ=0 ({deficits[0]:.6e})"
        )
        # Converges at large γ
        assert abs(deficits[-1] - deficits[-2]) / deficits[-2] < 0.01, (
            "Deficit should converge at large γ"
        )

    def test_polynomial_exactness_preserved_e2(self):
        """Polynomial exactness is maintained at all γ values (E2)."""
        n, sigma = 40, 6.0
        p, q, nextra, nu = self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU

        for g in [0, 10, 1000, 1e6]:
            D = build_diff_matrix_rbf_penalty(
                n, p, q, sigma, "tension", nu, nextra, gamma=g
            )
            err = self._polynomial_reproduction_error(D, n, nu, q)
            assert err < 1e-8, (
                f"Polynomial exactness lost at γ={g}: error={err:.6e}"
            )

    def test_polynomial_exactness_preserved_e4(self):
        """Polynomial exactness is maintained at all γ values (E4)."""
        n, sigma = 40, 37.0
        p, q, nextra, nu = self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU

        for g in [0, 10, 1000, 1e6]:
            D = build_diff_matrix_rbf_penalty(
                n, p, q, sigma, "tension", nu, nextra, gamma=g
            )
            err = self._polynomial_reproduction_error(D, n, nu, q)
            assert err < 1e-8, (
                f"Polynomial exactness lost at γ={g}: error={err:.6e}"
            )



# ---------------------------------------------------------------------------
# 30.4b: Modified wavenumber analysis for tension spline stencils
# ---------------------------------------------------------------------------


class TestModifiedWavenumber:
    """Modified wavenumber analysis for boundary vs interior stencils (Phase 30.4b).

    For a stencil w_j applied at node i_eval using nodes {j_0, ..., j_{t-1}},
    the modified wavenumber is:

        κ*(ξ) = Σ_j w_j · exp(i·(j - i_eval)·ξ)

    For D¹: exact κ* = iξ  →  Re(κ*)=0, Im(κ*)=ξ.
      - Re(κ*) < 0 is dissipative (good for stability)
      - Re(κ*) > 0 is amplifying (bad — causes eigenvalue instability)

    We check:
    1. Interior stencil Re(κ*) = 0 (centered, antisymmetric → pure imaginary)
    2. At optimal tension σ*, boundary Re(κ*_bdy) ≤ 0 for all ξ (E2)
    3. For E4, boundary Re(κ*_bdy) may have small positive region (unstable)
    4. Compare boundary vs interior dispersion Im(κ*)
    """

    # E2_1 parameters
    E2_P, E2_Q, E2_NEXTRA, E2_NU = 1, 1, 1, 1
    # E4_1 parameters
    E4_P, E4_Q, E4_NEXTRA, E4_NU = 2, 3, 0, 1

    N_XI = 500  # wavenumber resolution

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _modified_wavenumber(weights, i_eval, node_indices, xi_array):
        """Compute modified wavenumber κ*(ξ) for given stencil weights.

        Parameters
        ----------
        weights : array-like
            Stencil coefficients w_j.
        i_eval : int
            Grid index where derivative is evaluated.
        node_indices : array-like of int
            Grid indices used by the stencil (e.g. [0,1,...,t-1] for boundary).
        xi_array : np.ndarray
            Wavenumber values ξ ∈ [0, π].

        Returns
        -------
        np.ndarray (complex)
            κ*(ξ) = Σ_j w_j exp(i (j - i_eval) ξ)
        """
        w = np.asarray(weights, dtype=complex)
        offsets = np.asarray(node_indices) - i_eval  # j - i_eval
        # κ*(ξ) = Σ w_j exp(i·offset_j·ξ)  (vectorized over ξ)
        # shape: (len(xi), len(offsets))
        phase = np.exp(1j * np.outer(xi_array, offsets))
        return phase @ w  # shape (len(xi),)

    def _interior_mod_wavenumber(self, p, nu, xi_array):
        """Compute modified wavenumber for the classical interior stencil."""
        from stencil_gen.interior import derive_interior, full_gamma_array

        coeffs = derive_interior(0, p, nu)
        w = [float(c) for c in full_gamma_array(coeffs)]
        nodes = list(range(-p, p + 1))
        return self._modified_wavenumber(w, 0, nodes, xi_array)

    def _boundary_mod_wavenumbers(self, p, q, nextra, nu, sigma, kernel="tension"):
        """Compute modified wavenumber for all boundary rows.

        Returns dict: {row_index: κ*(ξ)} for rows 0..r-1.
        """
        t = p + q + 1 + nextra  # boundary stencil width (nu=1)
        r = q + 1 + nextra       # number of boundary rows

        xi = np.linspace(0, np.pi, self.N_XI)
        nodes = list(range(t))
        result = {}
        for i in range(r):
            w = uniform_boundary_weights_rbf(i, t, nu, q, sigma, kernel=kernel)
            result[i] = self._modified_wavenumber(w, i, nodes, xi)
        return result

    def _find_best_sigma(self, n, p, q, nu, nextra):
        """Coarse + fine sweep for best tension σ using corrected stability metric."""
        sigmas_coarse = np.concatenate([[0.0], np.linspace(1.0, 55.0, 100)])
        best_sigma, best_se = None, np.inf
        for s in sigmas_coarse:
            se = stability_eigenvalue(n, p, q, s, "tension", nu, nextra)
            if se < best_se:
                best_se = se
                best_sigma = s

        lo = max(0.0, best_sigma - 5.0)
        hi = min(60.0, best_sigma + 5.0)
        for s in np.linspace(lo, hi, 200):
            se = stability_eigenvalue(n, p, q, s, "tension", nu, nextra)
            if se < best_se:
                best_se = se
                best_sigma = s

        return best_sigma, best_se

    # ---------------------------------------------- interior sanity check

    def test_interior_pure_imaginary(self):
        """Interior centered D¹ stencil has Re(κ*)=0 (antisymmetric weights)."""
        xi = np.linspace(0, np.pi, self.N_XI)
        for p in [1, 2]:
            kappa = self._interior_mod_wavenumber(p, nu=1, xi_array=xi)
            max_real = float(np.max(np.abs(np.real(kappa))))
            assert max_real < 1e-14, (
                f"Interior p={p} Re(κ*) should be 0, got max |Re|={max_real:.2e}"
            )

    # ---------------------------------------------- E2 boundary analysis

    @pytest.mark.slow
    def test_e2_boundary_at_optimal_sigma(self):
        """Modified wavenumber profile of E2 boundary rows at optimal tension σ*.

        Key finding: the full operator is stable under corrected metric.
        Individual boundary stencils can have small positive Re(κ*), but
        stability is a global property of the coupled operator, not a
        per-stencil property.
        """
        p, q, nextra, nu = self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU
        sigma_star, best_se = self._find_best_sigma(40, p, q, nu, nextra)

        xi = np.linspace(0, np.pi, self.N_XI)
        bdy_kappas = self._boundary_mod_wavenumbers(p, q, nextra, nu, sigma_star)

        r = q + 1 + nextra
        print(f"\n  E2 Modified Wavenumber Analysis at σ*={sigma_star:.3f}")
        print(f"  (stab_eig = {best_se:.2e})")
        print(f"  {'row':>4s}  {'max Re(κ*)':>14s}  {'min Re(κ*)':>14s}"
              f"  {'max |Im(κ*)-ξ|':>16s}")
        print(f"  {'-'*4}  {'-'*14}  {'-'*14}  {'-'*16}")

        kappa_int = self._interior_mod_wavenumber(p, nu, xi)

        max_re_all = -np.inf
        for i in range(r):
            kappa = bdy_kappas[i]
            max_re = float(np.max(np.real(kappa)))
            min_re = float(np.min(np.real(kappa)))
            disp_err = float(np.max(np.abs(np.imag(kappa) - np.imag(kappa_int))))
            print(f"  {i:4d}  {max_re:14.6e}  {min_re:14.6e}  {disp_err:16.6e}")
            max_re_all = max(max_re_all, max_re)

        # Row 0 (boundary point itself) should be dissipative
        assert float(np.max(np.real(bdy_kappas[0]))) < STABILITY_TOL, (
            f"E2 boundary row 0 should be dissipative at σ*={sigma_star:.3f}"
        )

        # Per-stencil amplification is bounded (O(0.1-0.3), much less than 1).
        # At the corrected-metric optimal σ*, per-stencil amplification can be
        # larger than under the old metric because the optimal σ* is different.
        assert max_re_all < 0.5, (
            f"E2 boundary max Re(κ*) too large at σ*={sigma_star:.3f}: {max_re_all:.6e}"
        )

        # Full matrix is stable under corrected metric
        assert best_se < 0, (
            f"E2 full matrix should be stable at σ*: stab_eig = {best_se:.6e}"
        )

    def test_e2_boundary_amplifying_at_sigma_zero(self):
        """At σ=0 (PHS k=2), E2 boundary rows have Re(κ*) > 0 (some amplifying).

        Per-stencil amplification (Re(κ*) > 0 for some wavenumbers) is a local
        property that does NOT imply full-operator instability.  PHS k=2 IS
        stable under the corrected full-matrix test, but individual boundary
        stencils can still have amplifying modes at certain wavenumbers.
        """
        p, q, nextra, nu = self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU
        # Use σ → 0 (dispatches to PHS k=2)
        sigma_zero = 1e-15

        bdy_kappas = self._boundary_mod_wavenumbers(p, q, nextra, nu, sigma_zero)

        r = q + 1 + nextra
        any_amplifying = False
        for i in range(r):
            kappa = bdy_kappas[i]
            max_re = float(np.max(np.real(kappa)))
            if max_re > STABILITY_TOL:
                any_amplifying = True

        assert any_amplifying, (
            "At σ=0 (PHS k=2), at least one E2 boundary row should be amplifying"
        )

    # ---------------------------------------------- E4 boundary analysis

    @pytest.mark.slow
    def test_e4_boundary_at_optimal_sigma(self):
        """Modified wavenumber profile of E4 boundary rows at optimal tension σ*.

        Under corrected metric, E4 IS stable (full operator has stab_eig < 0).
        Per-stencil modified wavenumber can still show positive Re(κ*) regions
        at some wavenumbers — this is a local property that does NOT imply
        full-operator instability.  Per-stencil analysis overpredicts
        instability vs the coupled operator.
        """
        p, q, nextra, nu = self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU
        sigma_star, best_se = self._find_best_sigma(40, p, q, nu, nextra)

        xi = np.linspace(0, np.pi, self.N_XI)
        bdy_kappas = self._boundary_mod_wavenumbers(p, q, nextra, nu, sigma_star)

        r = q + 1 + nextra
        kappa_int = self._interior_mod_wavenumber(p, nu, xi)

        print(f"\n  E4 Modified Wavenumber Analysis at σ*={sigma_star:.3f}")
        print(f"  (stab_eig = {best_se:.2e})")
        print(f"  {'row':>4s}  {'max Re(κ*)':>14s}  {'min Re(κ*)':>14s}"
              f"  {'max |Im(κ*)-ξ|':>16s}")
        print(f"  {'-'*4}  {'-'*14}  {'-'*14}  {'-'*16}")

        overall_max_re = -np.inf
        for i in range(r):
            kappa = bdy_kappas[i]
            max_re = float(np.max(np.real(kappa)))
            min_re = float(np.min(np.real(kappa)))
            disp_err = float(np.max(np.abs(np.imag(kappa) - np.imag(kappa_int))))
            print(f"  {i:4d}  {max_re:14.6e}  {min_re:14.6e}  {disp_err:16.6e}")
            overall_max_re = max(overall_max_re, max_re)

        # Per-stencil amplification is bounded (O(0.1), much less than 1)
        assert overall_max_re < 0.5, (
            f"E4 boundary max Re(κ*) too large at σ*={sigma_star:.3f}: {overall_max_re:.6e}"
        )

        # Full operator is stable under corrected metric, even though
        # individual boundary stencils may have per-stencil amplification
        assert best_se < 0, (
            f"E4 full matrix should be stable at σ*: stab_eig = {best_se:.6e}"
        )

    def test_e4_phs_boundary_vs_tension_per_stencil(self):
        """Compare per-stencil max Re(κ*) between PHS k=2 and tension σ=3.0.

        Under corrected metric, PHS k=2 (σ=0) is the full-operator optimal.
        Tension at σ>0 can reduce per-stencil amplification, but this is a
        local property that doesn't affect full-operator stability.
        Both configurations are stable under the correct full-matrix test.
        """
        p, q, nextra, nu = self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU

        # PHS k=2 boundary max Re
        bdy_phs = self._boundary_mod_wavenumbers(p, q, nextra, nu, 1e-15)
        r = q + 1 + nextra
        phs_max_re = max(
            float(np.max(np.real(bdy_phs[i]))) for i in range(r)
        )

        # Tension σ=3.0 (production value) boundary max Re
        sigma_prod = 3.0
        bdy_tension = self._boundary_mod_wavenumbers(p, q, nextra, nu, sigma_prod)
        tension_max_re = max(
            float(np.max(np.real(bdy_tension[i]))) for i in range(r)
        )

        print(f"\n  E4 max Re(κ*) across boundary rows:")
        print(f"    PHS k=2 (σ=0):      {phs_max_re:.6e}")
        print(f"    Tension σ={sigma_prod:.1f}:    {tension_max_re:.6e}")

        # Both should have bounded per-stencil amplification
        assert phs_max_re < 0.5, (
            f"PHS k=2 per-stencil amplification too large: {phs_max_re:.6e}"
        )
        assert tension_max_re < 0.5, (
            f"Tension σ={sigma_prod} per-stencil amplification too large: "
            f"{tension_max_re:.6e}"
        )

    # ---------------------------------------------- dispersion comparison

    @pytest.mark.slow
    def test_dispersion_comparison(self):
        """Compare boundary vs interior dispersion for E2 and E4.

        Interior Im(κ*) approximates ξ to order 2p.  Boundary rows may have
        different dispersion.  We verify boundary dispersion error is bounded
        and report the comparison.
        """
        xi = np.linspace(0, np.pi, self.N_XI)

        configs = [
            ("E2", self.E2_P, self.E2_Q, self.E2_NEXTRA, self.E2_NU),
            ("E4", self.E4_P, self.E4_Q, self.E4_NEXTRA, self.E4_NU),
        ]

        print(f"\n  {'='*80}")
        print(f"  Dispersion Comparison: boundary vs interior Im(κ*)")
        print(f"  {'='*80}")

        for label, p, q, nextra, nu in configs:
            sigma_star, _ = self._find_best_sigma(40, p, q, nu, nextra)
            kappa_int = self._interior_mod_wavenumber(p, nu, xi)

            bdy_kappas = self._boundary_mod_wavenumbers(
                p, q, nextra, nu, sigma_star,
            )
            r = q + 1 + nextra

            print(f"\n  {label} — σ*={sigma_star:.3f}")
            print(f"  {'row':>4s}  {'max |ΔIm|':>14s}  {'mean |ΔIm|':>14s}")
            print(f"  {'-'*4}  {'-'*14}  {'-'*14}")

            # Interior dispersion error vs exact (iξ → Im = ξ)
            int_disp_err = float(np.max(np.abs(np.imag(kappa_int) - xi)))
            print(f"  {'int':>4s}  {int_disp_err:14.6e}  "
                  f"{float(np.mean(np.abs(np.imag(kappa_int) - xi))):14.6e}")

            for i in range(r):
                kappa = bdy_kappas[i]
                disp_vs_exact = np.abs(np.imag(kappa) - xi)
                max_err = float(np.max(disp_vs_exact))
                mean_err = float(np.mean(disp_vs_exact))
                print(f"  {i:4d}  {max_err:14.6e}  {mean_err:14.6e}")

                # Boundary dispersion should be finite (not blow up)
                assert max_err < 10.0, (
                    f"{label} row {i} dispersion error too large: {max_err:.2e}"
                )




# ---------------------------------------------------------------------------
# 32.1c: Validate corrected stability infrastructure
# ---------------------------------------------------------------------------


class TestStabilityInfrastructure:
    """Validate stability_eigenvalue with correct BC and sign convention.

    The corrected test removes the inflow row/column (Dirichlet at left)
    and checks eigenvalues of -D (the semi-discrete advection operator).
    """

    def test_production_e4_tension_stable(self):
        """E4 with tension spline at σ=3.0 is stable under correct test.

        With the corrected stability check (inflow-Dirichlet BC, eigenvalues
        of -D), the E4 tension spline configuration is stable at all grid
        sizes.
        """
        for n in [20, 40, 80, 160]:
            se = stability_eigenvalue(n, p=2, q=3, epsilon=3.0,
                                      kernel="tension", nu=1, nextra=0)
            assert se < STABILITY_TOL, (
                f"E4 tension σ=3.0, n={n}: expected stable, "
                f"got stability_eigenvalue={se:.6e}"
            )

    def test_interior_only_neutrally_stable(self):
        """Periodic interior-only matrix should be neutrally stable.

        A pure circulant interior matrix has purely imaginary eigenvalues,
        so eigenvalues of -D also have zero real parts (neutrally stable).
        """
        from stencil_gen.interior import derive_interior, full_gamma_array

        n = 40
        p = 2
        interior_coeffs = derive_interior(0, p, 1)
        interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

        D = np.zeros((n, n))
        for i in range(n):
            for k_idx, offset in enumerate(range(-p, p + 1)):
                j = (i + offset) % n
                D[i, j] = interior_w[k_idx]

        se = stability_eigenvalue_from_matrix(D)
        assert abs(se) < 1e-12, (
            f"Periodic interior stability_eigenvalue should be ≈0, got {se:.6e}"
        )

    def test_unstable_detected(self):
        """A known-unstable configuration should have positive stability_eigenvalue.

        E4 (p=2) with Gaussian RBF at epsilon=0.1 produces an unstable
        advection operator, confirming the test can detect instability.
        """
        se = stability_eigenvalue(20, p=2, q=3, epsilon=0.1,
                                  kernel="gaussian", nu=1, nextra=0)
        assert se > STABILITY_TOL, (
            f"Expected unstable configuration, got stability_eigenvalue={se:.6e}"
        )

    def test_stability_eigenvalue_from_matrix_consistent(self):
        """stability_eigenvalue and stability_eigenvalue_from_matrix agree."""
        n, p, q, eps = 20, 1, 2, 1.0
        D = build_diff_matrix_rbf(n, p, q, eps, kernel="gaussian", nu=1, nextra=0)
        se_direct = stability_eigenvalue(n, p, q, eps, kernel="gaussian",
                                         nu=1, nextra=0)
        se_from_mat = stability_eigenvalue_from_matrix(D)
        assert abs(se_direct - se_from_mat) < 1e-14, (
            f"Mismatch: direct={se_direct:.6e}, from_matrix={se_from_mat:.6e}"
        )

    def test_e2_stable(self):
        """E2 (p=1) with standard parameters should be stable.

        E2 is known to be unconditionally stable with reasonable boundary
        closures.
        """
        for n in [20, 40, 80]:
            se = stability_eigenvalue(n, p=1, q=2, epsilon=1.0,
                                      kernel="gaussian", nu=1, nextra=0)
            assert se < STABILITY_TOL, (
                f"E2 n={n}: expected stable, got stability_eigenvalue={se:.6e}"
            )


# ---------------------------------------------------------------------------
# Regression test helpers — load known-good values from sweeps/known_values.json
# ---------------------------------------------------------------------------

_KNOWN_VALUES_FILE = (
    Path(__file__).resolve().parent.parent / "sweeps" / "known_values.json"
)


def _load_known_values() -> dict | None:
    """Load known values, returning None if the file is absent."""
    try:
        with open(_KNOWN_VALUES_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


_KNOWN = _load_known_values()


# ---------------------------------------------------------------------------
# 37.3: Fast regression tests — require sweeps/known_values.json
# ---------------------------------------------------------------------------

if _KNOWN is not None:

    class TestRegressionE2Stability:
        """Fast regression spot-checks for E2 stability with known-good parameters.

        Values loaded from sweeps/known_values.json (E2_1 entry).
        """

        _kv = _KNOWN["E2_1"]
        P = _kv["params"]["p"]
        Q = _kv["params"]["q"]
        NEXTRA = _kv["params"]["nextra"]
        NU = _kv["params"]["nu"]

        def test_e2_tension_optimal_sigma(self):
            """E2_1 tension at known σ is stable."""
            sigma = _KNOWN["E2_1"]["tension"]["sigma"]
            se = stability_eigenvalue(
                40, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            assert se < STABILITY_TOL, (
                f"E2_1 tension σ={sigma} n=40: expected stable, got {se:.6e}"
            )

        def test_e2_gaussian_optimal_epsilon(self):
            """E2_1 Gaussian at known ε is stable."""
            eps = _KNOWN["E2_1"]["gaussian"]["epsilon"]
            se = stability_eigenvalue(
                40, p=self.P, q=self.Q, epsilon=eps,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            assert se < STABILITY_TOL, (
                f"E2_1 Gaussian ε={eps} n=40: expected stable, got {se:.6e}"
            )

        def test_e2_gaussian_epsilon_strictly_above_floor(self):
            """E2_1 gaussian epsilon must be strictly above the CLI default eps_floor.

            Plan 46.3b.1.2: the persisted ``E2_1.gaussian.epsilon`` must be a
            genuine fine-sweep optimum from the upper stable basin, not the
            value at the constraint boundary or the eps -> 0
            polynomial-reproduction limit. At floor=0.0 the fine sweep
            extrapolates below the coarse range to a degenerate-kernel
            value (eps ≈ 0.0025); asserting strictly above the floor catches
            a re-collapse if the floor is ever lowered.
            """
            from sweeps.epsilon_sweep import CLI_DEFAULT_EPS_FLOOR

            eps = _KNOWN["E2_1"]["gaussian"]["epsilon"]
            assert eps > CLI_DEFAULT_EPS_FLOOR + 1e-9, (
                f"E2_1 gaussian epsilon={eps} must be strictly greater than "
                f"the CLI default eps_floor={CLI_DEFAULT_EPS_FLOOR}; "
                f"floor-snap or eps -> 0 collapse means the regression "
                f"entry sits in a degenerate-kernel limit."
            )

        def test_e2_multiquadric_epsilon_strictly_above_floor(self):
            """E2_1 multiquadric epsilon must be strictly above the CLI default eps_floor.

            Plan 46.3c: the persisted ``E2_1.multiquadric.epsilon`` must be a
            genuine fine-sweep optimum (~2.68), not the historical hardcoded
            ``1.0`` (which was below floor=1.5 and never a refined optimum)
            or a constraint-boundary snap. Asserting strictly above the floor
            catches a re-collapse if the floor is ever lowered.
            """
            from sweeps.epsilon_sweep import CLI_DEFAULT_EPS_FLOOR

            eps = _KNOWN["E2_1"]["multiquadric"]["epsilon"]
            assert eps > CLI_DEFAULT_EPS_FLOOR + 1e-9, (
                f"E2_1 multiquadric epsilon={eps} must be strictly greater "
                f"than the CLI default eps_floor={CLI_DEFAULT_EPS_FLOOR}; "
                f"floor-snap means the regression entry lies at the "
                f"constraint boundary, not at a real local minimum."
            )

        def test_e2_stable_at_multiple_grid_sizes(self):
            """E2_1 tension and PHS k=2 are stable at known grid sizes."""
            kv = _KNOWN["E2_1"]
            sigma = kv["tension"]["sigma"]
            for s, label, grid_key in [
                (0.0, "PHS k=2", "phs_k2"),
                (sigma, f"tension σ={sigma}", "tension"),
            ]:
                for n in kv[grid_key]["stable_at"]:
                    se = stability_eigenvalue(
                        n, p=self.P, q=self.Q, epsilon=s,
                        kernel="tension", nu=self.NU, nextra=self.NEXTRA,
                    )
                    assert se < STABILITY_TOL, (
                        f"E2_1 {label} n={n}: expected stable, got {se:.6e}"
                    )

    class TestRegressionE4Stability:
        """Fast regression spot-checks for E4 stability with known-good parameters.

        Values loaded from sweeps/known_values.json (E4_1 entry).
        """

        _kv = _KNOWN["E4_1"]
        P = _kv["params"]["p"]
        Q = _kv["params"]["q"]
        NEXTRA = _kv["params"]["nextra"]
        NU = _kv["params"]["nu"]

        def test_e4_tension_known_sigma(self):
            """E4_1 tension at known σ is stable."""
            sigma = _KNOWN["E4_1"]["tension"]["sigma"]
            se = stability_eigenvalue(
                40, p=self.P, q=self.Q, epsilon=sigma,
                kernel="tension", nu=self.NU, nextra=self.NEXTRA,
            )
            assert se < STABILITY_TOL, (
                f"E4_1 tension σ={sigma} n=40: expected stable, got {se:.6e}"
            )

        def test_e4_gaussian_known_epsilon(self):
            """E4_1 Gaussian at known ε is stable."""
            eps = _KNOWN["E4_1"]["gaussian"]["epsilon"]
            se = stability_eigenvalue(
                40, p=self.P, q=self.Q, epsilon=eps,
                kernel="gaussian", nu=self.NU, nextra=self.NEXTRA,
            )
            assert se < STABILITY_TOL, (
                f"E4_1 Gaussian ε={eps} n=40: expected stable, got {se:.6e}"
            )

        def test_e4_gaussian_epsilon_strictly_above_floor(self):
            """E4_1 gaussian epsilon must be strictly above the CLI default eps_floor.

            Plan 46.3b.1.2: the persisted ``E4_1.gaussian.epsilon`` must be a
            genuine fine-sweep optimum from the upper stable basin (~2.10),
            not the lower 0.681-basin (silently excluded by the floor) or
            the eps -> 0 degenerate-kernel limit. Asserting strictly above
            the floor catches a re-collapse if the floor is ever lowered.
            """
            from sweeps.epsilon_sweep import CLI_DEFAULT_EPS_FLOOR

            eps = _KNOWN["E4_1"]["gaussian"]["epsilon"]
            assert eps > CLI_DEFAULT_EPS_FLOOR + 1e-9, (
                f"E4_1 gaussian epsilon={eps} must be strictly greater than "
                f"the CLI default eps_floor={CLI_DEFAULT_EPS_FLOOR}; "
                f"floor-snap or basin-collapse means the regression entry "
                f"lies outside the upper stable basin."
            )

        def test_e4_multiquadric_known_epsilon(self):
            """E4_1 multiquadric at known ε is stable."""
            eps = _KNOWN["E4_1"]["multiquadric"]["epsilon"]
            se = stability_eigenvalue(
                40, p=self.P, q=self.Q, epsilon=eps,
                kernel="multiquadric", nu=self.NU, nextra=self.NEXTRA,
            )
            assert se < STABILITY_TOL, (
                f"E4_1 multiquadric ε={eps} n=40: expected stable, got {se:.6e}"
            )

        def test_e4_multiquadric_epsilon_strictly_above_floor(self):
            """E4_1 multiquadric epsilon must be strictly above the CLI default eps_floor.

            Plan 46.3c: the persisted ``E4_1.multiquadric.epsilon`` must be a
            genuine fine-sweep optimum from the upper stable basin (~1.57),
            not the lower basin near eps≈1.13 (silently excluded by the
            floor) or the historical hardcoded ``1.0``. The narrow margin
            (~0.07 above floor) is acceptable: the basin curvature is well
            behaved, and any genuine floor-snap is also caught by the
            ``snap_tol`` warning emitted from ``fine_sweep``.
            """
            from sweeps.epsilon_sweep import CLI_DEFAULT_EPS_FLOOR

            eps = _KNOWN["E4_1"]["multiquadric"]["epsilon"]
            assert eps > CLI_DEFAULT_EPS_FLOOR + 1e-9, (
                f"E4_1 multiquadric epsilon={eps} must be strictly greater "
                f"than the CLI default eps_floor={CLI_DEFAULT_EPS_FLOOR}; "
                f"floor-snap or basin-collapse means the regression entry "
                f"lies outside the upper stable basin."
            )

        def test_e4_stable_at_multiple_grid_sizes(self):
            """E4_1 tension and PHS k=2 are stable at known grid sizes."""
            kv = _KNOWN["E4_1"]
            sigma = kv["tension"]["sigma"]
            for s, label, grid_key in [
                (0.0, "PHS k=2", "phs_k2"),
                (sigma, f"tension σ={sigma}", "tension"),
            ]:
                for n in kv[grid_key]["stable_at"]:
                    se = stability_eigenvalue(
                        n, p=self.P, q=self.Q, epsilon=s,
                        kernel="tension", nu=self.NU, nextra=self.NEXTRA,
                    )
                    assert se < STABILITY_TOL, (
                        f"E4_1 {label} n={n}: expected stable, got {se:.6e}"
                    )

        def test_e4_unstable_detected(self):
            """Known-unstable configurations should be detected."""
            for entry in _KNOWN["E4_1"]["known_unstable"]:
                se = stability_eigenvalue(
                    entry["n"], p=self.P, q=self.Q, epsilon=entry["epsilon"],
                    kernel=entry["kernel"], nu=self.NU, nextra=self.NEXTRA,
                )
                assert se > STABILITY_TOL, (
                    f"E4_1 {entry['kernel']} ε={entry['epsilon']} n={entry['n']}: "
                    f"expected unstable, got {se:.6e}"
                )

        def test_e4_tension_sigma_strictly_above_floor(self):
            """E4_1 tension sigma must be strictly above the CLI default sigma_floor.

            Plan 46.3a.2: the persisted ``E4_1.tension.sigma`` must be a
            genuine fine-sweep optimum, not the value at the constraint
            boundary. At floors below 1.0 the constrained E4 optimum
            snaps exactly to the floor, making the tension regression
            entry numerically indistinguishable from the PHS k=2 limit.
            Asserting strictly above the floor catches a re-collapse if
            the floor is ever lowered.
            """
            from sweeps.tension_sweep import CLI_DEFAULT_SIGMA_FLOOR

            sigma = _KNOWN["E4_1"]["tension"]["sigma"]
            assert sigma > CLI_DEFAULT_SIGMA_FLOOR + 1e-9, (
                f"E4_1 tension sigma={sigma} must be strictly greater than "
                f"the CLI default sigma_floor={CLI_DEFAULT_SIGMA_FLOOR}; "
                f"floor-snap means the regression entry is structurally "
                f"identical to phs_k2."
            )

    class TestRegressionFootprint:
        """Fast regression spot-checks for E4 nextra stability.

        Values loaded from sweeps/known_values.json (footprint entry).
        Uses E4_1 base parameters (p=2, q=3, nu=1).
        """

        _e4 = _KNOWN["E4_1"]["params"]
        P = _e4["p"]
        Q = _e4["q"]
        NU = _e4["nu"]
        _fp = _KNOWN["footprint"]

        def test_nextra0_phs_k2_grid_independence(self):
            """E4 nextra=0, PHS k=2 is stable at known grid sizes."""
            entry = self._fp["E4_nextra0_phs"]
            for n in entry["stable_at"]:
                se = stability_eigenvalue(
                    n, p=self.P, q=self.Q, epsilon=0.0,
                    kernel="tension", nu=self.NU, nextra=entry["nextra"],
                )
                assert se < STABILITY_TOL, (
                    f"E4 PHS k=2 nextra=0 n={n}: expected stable, got {se:.6e}"
                )

        def test_nextra0_tension(self):
            """E4 nextra=0, tension at known σ is stable."""
            entry = self._fp["E4_nextra0_tension_3"]
            sigma = entry["sigma"]
            for n in entry["stable_at"]:
                se = stability_eigenvalue(
                    n, p=self.P, q=self.Q, epsilon=sigma,
                    kernel="tension", nu=self.NU, nextra=entry["nextra"],
                )
                assert se < STABILITY_TOL, (
                    f"E4 nextra=0 tension σ={sigma} n={n}: expected stable, got {se:.6e}"
                )

        def test_nextra1_has_stable_sigma(self):
            """E4 nextra=1 has a stable PHS k=2."""
            entry = self._fp["E4_nextra1_phs"]
            for n in entry["stable_at"]:
                se = stability_eigenvalue(
                    n, p=self.P, q=self.Q, epsilon=0.0,
                    kernel="tension", nu=self.NU, nextra=entry["nextra"],
                )
                assert se < STABILITY_TOL, (
                    f"E4 nextra=1 PHS k=2 n={n}: expected stable, got {se:.6e}"
                )

        def test_nextra2_has_stable_sigma(self):
            """E4 nextra=2 has a stable PHS k=2."""
            entry = self._fp["E4_nextra2_phs"]
            for n in entry["stable_at"]:
                se = stability_eigenvalue(
                    n, p=self.P, q=self.Q, epsilon=0.0,
                    kernel="tension", nu=self.NU, nextra=entry["nextra"],
                )
                assert se < STABILITY_TOL, (
                    f"E4 nextra=2 PHS k=2 n={n}: expected stable, got {se:.6e}"
                )

    class TestRegressionComparison:
        """Fast regression spot-checks for the comprehensive comparison.

        Values loaded from sweeps/known_values.json.
        Tests all methods for both E2_1 and E4_1 at known-good parameters.
        """

        @staticmethod
        def _configs_for(scheme_key):
            """Build (label, kernel, eps) configs from known values."""
            kv = _KNOWN[scheme_key]
            configs = [("PHS k=2", "tension", 0.0)]
            for method in ("gaussian", "tension", "multiquadric"):
                if method in kv:
                    param_key = "sigma" if method == "tension" else "epsilon"
                    configs.append((method, method, kv[method][param_key]))
            return configs

        def test_e2_all_methods_stable(self):
            """E2_1: all methods stable at n=40."""
            kv = _KNOWN["E2_1"]
            p = kv["params"]
            for label, kernel, eps in self._configs_for("E2_1"):
                se = stability_eigenvalue(
                    40, p=p["p"], q=p["q"], epsilon=eps,
                    kernel=kernel, nu=p["nu"], nextra=p["nextra"],
                )
                assert se < STABILITY_TOL, (
                    f"E2_1 {label} n=40: expected stable, got {se:.6e}"
                )

        def test_e4_all_methods_stable(self):
            """E4_1: all methods stable at n=40."""
            kv = _KNOWN["E4_1"]
            p = kv["params"]
            for label, kernel, eps in self._configs_for("E4_1"):
                se = stability_eigenvalue(
                    40, p=p["p"], q=p["q"], epsilon=eps,
                    kernel=kernel, nu=p["nu"], nextra=p["nextra"],
                )
                assert se < STABILITY_TOL, (
                    f"E4_1 {label} n=40: expected stable, got {se:.6e}"
                )

        def test_phs_k2_grid_convergence(self):
            """PHS k=2 is stable at known grid sizes for both E2 and E4."""
            for scheme_key in ("E2_1", "E4_1"):
                kv = _KNOWN[scheme_key]
                p = kv["params"]
                for n in kv["phs_k2"]["stable_at"]:
                    se = stability_eigenvalue(
                        n, p=p["p"], q=p["q"], epsilon=0.0,
                        kernel="tension", nu=p["nu"], nextra=p["nextra"],
                    )
                    assert se < STABILITY_TOL, (
                        f"{scheme_key} PHS k=2 n={n}: expected stable, got {se:.6e}"
                    )

    class TestRegressionGV:
        """Fast regression spot-checks for group-velocity-optimal known values.

        For each ``*_gv`` entry in ``sweeps/known_values.json`` (populated by
        running each sweep with ``--include-gv --update-known-values``), rebuild
        the boundary D matrix at the stored optimum and assert that
        ``gv_score_from_matrix(D)["max_gv_error"]`` matches the stored
        ``gv_error`` within 10% (allows floating-point / ``n_xi`` slack).

        Gracefully skips individual tests when the corresponding ``*_gv`` keys
        are absent — the 40.8a contract is "activate once sweeps populate the
        keys", not "fail when they haven't yet".
        """

        GV_TOLERANCE = 1.1
        GV_TOLERANCE_STRICT = 1.001
        GRID_N = 40

        @staticmethod
        def _iter_scheme_gv_entries():
            """Yield (scheme_key, kernel, param_name, gv_entry, params)."""
            for scheme_key in ("E2_1", "E4_1"):
                if scheme_key not in _KNOWN:
                    continue
                kv = _KNOWN[scheme_key]
                params = kv.get("params")
                if params is None:
                    continue
                for gv_key, kernel, param_name in (
                    ("tension_gv", "tension", "sigma"),
                    ("gaussian_gv", "gaussian", "epsilon"),
                    ("multiquadric_gv", "multiquadric", "epsilon"),
                ):
                    entry = kv.get(gv_key)
                    if entry is None or "gv_error" not in entry or param_name not in entry:
                        continue
                    yield scheme_key, kernel, param_name, entry, params

        def test_scheme_gv_entries_match_stored_error(self):
            """E2_1/E4_1 ``*_gv`` entries: measured GV error ≤ stored × 1.1."""
            from sweeps.gv_objectives import gv_score_from_matrix

            entries = list(self._iter_scheme_gv_entries())
            if not entries:
                pytest.skip("no *_gv entries in known_values.json")
            for scheme_key, kernel, param_name, gv_entry, params in entries:
                eps = gv_entry[param_name]
                stored = gv_entry["gv_error"]
                D = build_diff_matrix_rbf(
                    self.GRID_N,
                    p=params["p"], q=params["q"], epsilon=eps,
                    kernel=kernel, nu=params["nu"], nextra=params["nextra"],
                )
                measured = gv_score_from_matrix(D)["max_gv_error"]
                assert measured <= stored * self.GV_TOLERANCE, (
                    f"{scheme_key} {kernel} {param_name}={eps}: measured "
                    f"gv_error {measured:.6e} > stored {stored:.6e} × "
                    f"{self.GV_TOLERANCE}"
                )

        def test_footprint_gv_entries_match_stored_error(self):
            """``footprint.E4_nextra{nx}_tension_gv`` entries: rebuild D and verify."""
            from sweeps.gv_objectives import gv_score_from_matrix

            if "footprint" not in _KNOWN or "E4_1" not in _KNOWN:
                pytest.skip("footprint or E4_1 missing from known_values.json")
            fp = _KNOWN["footprint"]
            params = _KNOWN["E4_1"]["params"]
            gv_entries = [
                (key, entry)
                for key, entry in fp.items()
                if key.endswith("_tension_gv")
                and isinstance(entry, dict)
                and "gv_error" in entry
                and "sigma" in entry
                and "nextra" in entry
            ]
            if not gv_entries:
                pytest.skip("no footprint *_tension_gv entries in known_values.json")
            for key, entry in gv_entries:
                sigma = entry["sigma"]
                nx = entry["nextra"]
                stored = entry["gv_error"]
                D = build_diff_matrix_rbf(
                    self.GRID_N,
                    p=params["p"], q=params["q"], epsilon=sigma,
                    kernel="tension", nu=params["nu"], nextra=nx,
                )
                measured = gv_score_from_matrix(D)["max_gv_error"]
                assert measured <= stored * self.GV_TOLERANCE, (
                    f"footprint {key} sigma={sigma} nextra={nx}: measured "
                    f"gv_error {measured:.6e} > stored {stored:.6e} × "
                    f"{self.GV_TOLERANCE}"
                )

        def test_scheme_primary_gv_error_match(self):
            """``E{2,4}_1.{kernel}.gv_error`` (additive field on the primary entry)."""
            from sweeps.gv_objectives import gv_score_from_matrix

            checked = 0
            for scheme_key in ("E2_1", "E4_1"):
                if scheme_key not in _KNOWN:
                    continue
                kv = _KNOWN[scheme_key]
                params = kv.get("params")
                if params is None:
                    continue
                for kernel, param_name in (
                    ("tension", "sigma"),
                    ("gaussian", "epsilon"),
                    ("multiquadric", "epsilon"),
                ):
                    entry = kv.get(kernel)
                    if not isinstance(entry, dict):
                        continue
                    if "gv_error" not in entry or param_name not in entry:
                        continue
                    eps = entry[param_name]
                    stored = entry["gv_error"]
                    D = build_diff_matrix_rbf(
                        self.GRID_N,
                        p=params["p"], q=params["q"], epsilon=eps,
                        kernel=kernel, nu=params["nu"], nextra=params["nextra"],
                    )
                    measured = gv_score_from_matrix(D)["max_gv_error"]
                    assert measured <= stored * self.GV_TOLERANCE_STRICT, (
                        f"{scheme_key} {kernel} (stability-optimum) "
                        f"{param_name}={eps}: measured gv_error "
                        f"{measured:.6e} > stored {stored:.6e} × "
                        f"{self.GV_TOLERANCE_STRICT}"
                    )
                    checked += 1
            if checked == 0:
                pytest.skip("no scheme.*.gv_error fields in known_values.json")

        def test_footprint_primary_tension_gv_error_match(self):
            """``footprint.E4_nextra{nx}_tension_{N}.gv_error`` (primary entry, 4dp rounding)."""
            from sweeps.gv_objectives import gv_score_from_matrix

            if "footprint" not in _KNOWN or "E4_1" not in _KNOWN:
                pytest.skip("footprint or E4_1 missing from known_values.json")
            fp = _KNOWN["footprint"]
            params = _KNOWN["E4_1"]["params"]
            primary_entries = [
                (key, entry)
                for key, entry in fp.items()
                if "_tension_" in key
                and not key.endswith("_tension_gv")
                and isinstance(entry, dict)
                and "gv_error" in entry
                and "sigma" in entry
                and "nextra" in entry
            ]
            if not primary_entries:
                pytest.skip("no footprint primary *_tension_{N}.gv_error entries in known_values.json")
            for key, entry in primary_entries:
                sigma = entry["sigma"]
                nx = entry["nextra"]
                stored = entry["gv_error"]
                D = build_diff_matrix_rbf(
                    self.GRID_N,
                    p=params["p"], q=params["q"], epsilon=sigma,
                    kernel="tension", nu=params["nu"], nextra=nx,
                )
                measured = gv_score_from_matrix(D)["max_gv_error"]
                assert measured <= stored * self.GV_TOLERANCE_STRICT, (
                    f"footprint {key} (stability-optimum) sigma={sigma} "
                    f"nextra={nx}: measured gv_error {measured:.6e} > "
                    f"stored {stored:.6e} × {self.GV_TOLERANCE_STRICT}"
                )

        def test_tension_penalty_gv_entries_match_stored_error(self):
            """``E{2,4}_1.tension_penalty_gv`` entries: measured ≤ stored × 1.1.

            Secondary GV-optimum entries rebuild D via the penalty-augmented
            constructor at the persisted (sigma, gamma).  Gated at the loose
            10% tolerance matching the other secondary GV tests — the GV-
            optimum is a stationary point where honest floating-point /
            ``n_xi`` drift can hit a few percent.
            """
            from sweeps.gv_objectives import gv_score_from_matrix

            checked = 0
            for scheme_key in ("E2_1", "E4_1"):
                if scheme_key not in _KNOWN:
                    continue
                kv = _KNOWN[scheme_key]
                params = kv.get("params")
                if params is None:
                    continue
                entry = kv.get("tension_penalty_gv")
                if not isinstance(entry, dict):
                    continue
                if not {"sigma", "gamma", "gv_error"} <= set(entry):
                    continue
                sigma = entry["sigma"]
                gamma = entry["gamma"]
                stored = entry["gv_error"]
                D = build_diff_matrix_rbf_penalty(
                    self.GRID_N, params["p"], params["q"], sigma,
                    "tension", params["nu"], params["nextra"],
                    gamma=gamma,
                )
                measured = gv_score_from_matrix(D)["max_gv_error"]
                assert measured <= stored * self.GV_TOLERANCE, (
                    f"{scheme_key} tension_penalty_gv sigma={sigma} "
                    f"gamma={gamma}: measured gv_error {measured:.6e} > "
                    f"stored {stored:.6e} × {self.GV_TOLERANCE}"
                )
                checked += 1
            if checked == 0:
                pytest.skip("no tension_penalty_gv entries in known_values.json")

        def test_tension_penalty_primary_gv_error_match(self):
            """``E{2,4}_1.tension_penalty.gv_error`` (primary, 40.8g strict gate).

            The primary entry's ``(sigma, gamma, gv_error)`` triple is written
            at the stability-optimum by 40.8g; the pair must be bit-exact
            self-consistent (modulo ~1e-12 numerical noise), so 0.1% strict
            tolerance is appropriate.
            """
            from sweeps.gv_objectives import gv_score_from_matrix

            checked = 0
            for scheme_key in ("E2_1", "E4_1"):
                if scheme_key not in _KNOWN:
                    continue
                kv = _KNOWN[scheme_key]
                params = kv.get("params")
                if params is None:
                    continue
                entry = kv.get("tension_penalty")
                if not isinstance(entry, dict):
                    continue
                if not {"sigma", "gamma", "gv_error"} <= set(entry):
                    continue
                sigma = entry["sigma"]
                gamma = entry["gamma"]
                stored = entry["gv_error"]
                D = build_diff_matrix_rbf_penalty(
                    self.GRID_N, params["p"], params["q"], sigma,
                    "tension", params["nu"], params["nextra"],
                    gamma=gamma,
                )
                measured = gv_score_from_matrix(D)["max_gv_error"]
                assert measured <= stored * self.GV_TOLERANCE_STRICT, (
                    f"{scheme_key} tension_penalty (stability-optimum) "
                    f"sigma={sigma} gamma={gamma}: measured gv_error "
                    f"{measured:.6e} > stored {stored:.6e} × "
                    f"{self.GV_TOLERANCE_STRICT}"
                )
                checked += 1
            if checked == 0:
                pytest.skip("no tension_penalty.gv_error fields in known_values.json")

    class TestRegressionBrady2DCalibration:
        """Regression tests for Brady-Livescu 2D calibration results.

        Loads ``brady2d_calibration`` from ``known_values.json``, iterates
        each family, re-runs ``brady2d_stability_score`` at ``max_layer=3``
        (fast subset — keeps total runtime under 10 s), and asserts the
        overall verdict matches the stored value.
        """

        _CAL = _KNOWN.get("brady2d_calibration")

        @pytest.fixture(autouse=True)
        def _skip_if_absent(self):
            if self._CAL is None:
                pytest.skip("brady2d_calibration key absent from known_values.json")

        def test_all_families_verdict_matches(self):
            """Each family's max_layer=3 verdict matches stored calibration."""
            from stencil_gen.benchmarks.brady2d_calibration import FAMILIES
            from stencil_gen.brady2d_stability import brady2d_stability_score

            for scheme, kernel, params, label in FAMILIES:
                if label not in self._CAL:
                    continue
                stored = self._CAL[label]
                report = brady2d_stability_score(
                    scheme, kernel, params,
                    max_layer=3, short_circuit=True,
                )
                stored_verdict = stored["overall_verdict"]
                stored_failed = stored.get("failed_layer")
                # A family that failed at layer > 3 would pass at max_layer=3.
                # Only families that failed at layer <= 3 should fail here.
                if stored_failed is not None and stored_failed <= 3:
                    assert report.overall_verdict == "fail", (
                        f"{label}: expected fail (stored failed_layer="
                        f"{stored_failed}), got {report.overall_verdict}"
                    )
                else:
                    assert report.overall_verdict == "pass", (
                        f"{label}: expected pass at max_layer=3, "
                        f"got {report.overall_verdict} "
                        f"(failed_layer={report.failed_layer}, "
                        f"reason={report.failed_reason})"
                    )

        def test_bl42_spectral_abscissa_matches(self):
            """layer_bl42.max_spectral_abscissa reproduces within 1%."""
            from stencil_gen.benchmarks.brady2d_calibration import FAMILIES
            from stencil_gen.brady2d_stability import brady2d_stability_score

            for scheme, kernel, params, label in FAMILIES:
                if label not in self._CAL:
                    continue
                stored = self._CAL[label]
                stored_bl42 = stored.get("layer_bl42")
                if stored_bl42 is None:
                    continue
                report = brady2d_stability_score(
                    scheme, kernel, params,
                    max_layer=3, short_circuit=True,
                )
                assert report.layer_bl42 is not None, (
                    f"{label}: expected layer_bl42 populated"
                )
                got = report.layer_bl42["max_spectral_abscissa"]
                expected = stored_bl42["max_spectral_abscissa"]
                if expected < 1e-12:
                    assert got < 1e-10, (
                        f"{label}: expected near-zero spectral abscissa, "
                        f"got {got:.4e}"
                    )
                else:
                    rel_err = abs(got - expected) / abs(expected)
                    assert rel_err < 0.01, (
                        f"{label}: BL42 max_spectral_abscissa {got:.6e} "
                        f"vs stored {expected:.6e} (rel_err={rel_err:.4e})"
                    )

    class TestRegressionBrady2DSweep:
        """Regression tests for Brady-Livescu 2D sweep results.

        Loads ``brady2d_sweep`` from ``known_values.json``, iterates each
        stored (scheme, kernel) entry and each swept point, re-runs
        ``brady2d_stability_score`` at ``max_layer=3`` (fast subset), and
        asserts that the stored overall verdict matches the recomputed one.
        Graceful skip when the key is absent.
        """

        _SWEEP = _KNOWN.get("brady2d_sweep")

        @pytest.fixture(autouse=True)
        def _skip_if_absent(self):
            if self._SWEEP is None:
                pytest.skip("brady2d_sweep key absent from known_values.json")

        def test_all_sweep_points_verdict_matches(self):
            """Each stored sweep point's max_layer=3 verdict matches recompute."""
            from stencil_gen.brady2d_stability import brady2d_stability_score

            checked = 0
            for scheme, kernels in self._SWEEP.items():
                for kernel, bucket in kernels.items():
                    for point in bucket.get("points", []):
                        params = point["params_dict"]
                        stored = point["report"]
                        stored_verdict = stored["overall_verdict"]
                        stored_failed = stored.get("failed_layer")
                        report = brady2d_stability_score(
                            scheme, kernel, params,
                            max_layer=3, short_circuit=True,
                        )
                        # If the stored run failed at layer > 3, the fast
                        # max_layer=3 recompute may pass — only assert fail
                        # when the stored failure is reachable at layer <= 3.
                        if (stored_verdict == "fail"
                                and stored_failed is not None
                                and stored_failed <= 3):
                            assert report.overall_verdict == "fail", (
                                f"{scheme}/{kernel} param={point['param']}: "
                                f"expected fail (stored failed_layer="
                                f"{stored_failed}), got "
                                f"{report.overall_verdict}"
                            )
                        else:
                            assert report.overall_verdict == "pass", (
                                f"{scheme}/{kernel} param={point['param']}: "
                                f"expected pass at max_layer=3, got "
                                f"{report.overall_verdict} "
                                f"(failed_layer={report.failed_layer}, "
                                f"reason={report.failed_reason})"
                            )
                        checked += 1
            if checked == 0:
                pytest.skip("brady2d_sweep had no stored points")

    class TestRegressionBrady2DOptima:
        """Regression tests for Brady-Livescu 2D optimizer results.

        Loads ``brady2d_optima`` from ``known_values.json`` (written by
        ``python -m sweeps optimize --update-known-values``), iterates each
        stored ``[scheme][kernel][objective]`` entry, rebuilds the objective
        via :func:`stencil_gen.optimizer.make_objective`, evaluates it at
        the stored ``best_x``, and asserts the result matches the stored
        ``best_objective`` within 1% relative tolerance.  Also verifies
        ``converged is True`` at the stored result.  Graceful skip when the
        key is absent (first run before any optimizer persistence).
        """

        _OPTIMA = _KNOWN.get("brady2d_optima")

        @pytest.fixture(autouse=True)
        def _skip_if_absent(self):
            if self._OPTIMA is None:
                pytest.skip("brady2d_optima key absent from known_values.json")

        def test_all_optima_objective_matches(self):
            """Each stored optimum reproduces its recorded best_objective."""
            import numpy as np

            from stencil_gen.optimizer import make_objective

            checked = 0
            for scheme, kernels in self._OPTIMA.items():
                for kernel, objectives in kernels.items():
                    for objective, entry in objectives.items():
                        assert entry.get("converged") is True, (
                            f"{scheme}/{kernel}/{objective}: stored "
                            f"converged={entry.get('converged')!r}, expected True"
                        )
                        best_x = np.asarray(entry["best_x"], dtype=float)
                        stored_obj = float(entry["best_objective"])
                        # Plan 43.8c: rebuild the objective at the persisted
                        # gate/max layers so older defaults can't drift.  Fall
                        # back to make_objective's own defaults if the entry
                        # predates 43.8c (no gate_layer / max_layer keys).  For
                        # staged entries, use validator_max_layer since that is
                        # the depth at which best_objective was evaluated.
                        if "gate_layer" not in entry or "max_layer" not in entry:
                            pytest.skip(
                                f"{scheme}/{kernel}/{objective}: entry predates "
                                "plan 43.8c (no gate_layer/max_layer); "
                                "re-run optimizer to refresh"
                            )
                        gate_layer = int(entry["gate_layer"])
                        if entry.get("method") == "staged":
                            assert "validator_max_layer" in entry, (
                                f"{scheme}/{kernel}/{objective}: staged entry "
                                "missing validator_max_layer"
                            )
                            eval_max_layer = int(entry["validator_max_layer"])
                        else:
                            eval_max_layer = int(entry["max_layer"])
                        f = make_objective(
                            scheme=scheme,
                            kernel=kernel,
                            report_field=objective,
                            gate_layer=gate_layer,
                            max_layer=eval_max_layer,
                        )
                        recomputed = f(best_x)
                        assert np.isfinite(recomputed), (
                            f"{scheme}/{kernel}/{objective}: recomputed "
                            f"objective is non-finite ({recomputed}) at "
                            f"best_x={best_x}"
                        )
                        denom = max(abs(stored_obj), 1e-12)
                        rel = abs(recomputed - stored_obj) / denom
                        assert rel < 1e-2, (
                            f"{scheme}/{kernel}/{objective}: recomputed "
                            f"{recomputed:.6e} differs from stored "
                            f"{stored_obj:.6e} by {rel:.2%} (>1%)"
                        )
                        checked += 1
            if checked == 0:
                pytest.skip("brady2d_optima had no stored entries")


# ---------------------------------------------------------------------------
# 45.6b.3: Regression tests for per-run Pareto fronts under
# sweeps/pareto_fronts/. Each JSON written by ``python -m sweeps pareto
# --persist`` records stored objective values; this class recomputes them
# via :func:`stencil_gen.pareto.make_multi_objective` and asserts a tight
# match. Gated by 45.6b.1 (deterministic ARPACK in
# ``spectral_abscissa_sparse``) and 45.6b.2 (regenerated fronts under that
# regime) — without both, BL42 recompute is process-seeded and drifts.
# ---------------------------------------------------------------------------

_PARETO_FRONTS_DIR = (
    Path(__file__).resolve().parent.parent / "sweeps" / "pareto_fronts"
)


def _load_pareto_fronts() -> list[tuple[Path, dict]]:
    """Return ``(path, parsed_json)`` for every readable front, sorted by name."""
    if not _PARETO_FRONTS_DIR.is_dir():
        return []
    loaded: list[tuple[Path, dict]] = []
    for path in sorted(_PARETO_FRONTS_DIR.glob("*.json")):
        try:
            with open(path) as fh:
                loaded.append((path, json.load(fh)))
        except (OSError, ValueError):
            continue
    return loaded


_PARETO_FRONTS = _load_pareto_fronts()


@pytest.mark.slow
class TestRegressionBrady2DPareto:
    """Regression tests for Brady-Livescu 2D Pareto fronts.

    For each JSON under ``sweeps/pareto_fronts/``, iterate the ``front``,
    rebuild a multi-objective closure via
    :func:`stencil_gen.pareto.make_multi_objective`, evaluate at the stored
    ``x``, and assert the recomputed vector matches the stored ``objectives``
    within 1% relative tolerance.  Additionally verify no stored front
    contains a dominated member (guards against corrupt persistence).

    Skipped entirely when ``sweeps/pareto_fronts/`` is empty or absent.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_absent(self):
        if not _PARETO_FRONTS:
            pytest.skip("sweeps/pareto_fronts/ empty or absent")

    def test_each_front_member_objectives_match(self):
        """Each stored front member reproduces its recorded objectives."""
        import numpy as np

        from stencil_gen.pareto import _PARETO_SENTINEL, make_multi_objective

        checked = 0
        for path, data in _PARETO_FRONTS:
            scheme = data["scheme"]
            kernel = data["kernel"]
            objective_fields = tuple(data["objective_fields"])
            f = make_multi_objective(scheme, kernel, objective_fields)
            for idx, member in enumerate(data["front"]):
                stored = np.asarray(member["objectives"], dtype=float)
                if np.any(stored >= _PARETO_SENTINEL):
                    continue
                x = np.asarray(member["x"], dtype=float)
                recomputed = np.asarray(f(x), dtype=float)
                assert np.all(np.isfinite(recomputed)), (
                    f"{path.name} front[{idx}]: recomputed non-finite "
                    f"({recomputed}) at x={x}"
                )
                assert np.allclose(
                    recomputed, stored, rtol=1e-2, atol=1e-8
                ), (
                    f"{path.name} front[{idx}]: recomputed {recomputed} "
                    f"differs from stored {stored} by more than rtol=1e-2"
                )
                checked += 1
        if checked == 0:
            pytest.skip("no non-sentinel Pareto front members to check")

    def test_each_front_is_non_dominated(self):
        """No stored front contains a member dominated by another member."""
        import numpy as np

        from stencil_gen.pareto import _PARETO_SENTINEL

        for path, data in _PARETO_FRONTS:
            rows: list[np.ndarray] = []
            for member in data["front"]:
                obj = np.asarray(member["objectives"], dtype=float)
                if np.any(obj >= _PARETO_SENTINEL):
                    continue
                rows.append(obj)
            if len(rows) < 2:
                continue
            F = np.vstack(rows)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i == j:
                        continue
                    # j dominates i iff F[j] <= F[i] component-wise and
                    # strictly less in at least one.
                    leq = np.all(F[j] <= F[i])
                    strict = np.any(F[j] < F[i])
                    assert not (leq and strict), (
                        f"{path.name}: front member {i} ({F[i]}) is "
                        f"dominated by member {j} ({F[j]})"
                    )


# ---------------------------------------------------------------------------
# 47.7b: Regression tests for per-run MF-BO benchmarks under
# sweeps/bo_runs/. Each JSON written by ``python -m sweeps bo --persist``
# captures a ``BOResult`` (per the schema in ``sweeps/_bo_io.py``); this
# class rebuilds :func:`stencil_gen.bo.make_multi_fidelity_objective` from
# the stored ``report_fields_by_layer`` and re-evaluates ``best_x`` at the
# HF level, asserting the recomputed scalar matches the stored
# ``best_objective`` to within 1% relative tolerance.  ``_BO_SENTINEL``
# (``1e12``) is a finite float and round-trips bytewise, so sentinel runs
# (those that exhausted the budget without finding a feasible region — see
# 47.7a's Done note for the calibration entry) compare cleanly.
# ---------------------------------------------------------------------------

_BO_RUNS_DIR = (
    Path(__file__).resolve().parent.parent / "sweeps" / "bo_runs"
)


def _load_bo_runs() -> list[tuple[Path, dict]]:
    """Return ``(path, parsed_json)`` for every readable BO run, sorted by name.

    Uses :func:`sweeps._bo_io.load_bo_run` so the four whitelisted
    int-keyed top-level dicts (``report_fields_by_layer``, ``cost_model``,
    ``n_evals_per_fidelity``, ``wall_time_per_fidelity``) come back with
    int keys, per plan item 47.4c.1.  Without this, piping
    ``report_fields_by_layer`` straight into
    :func:`stencil_gen.bo.make_multi_fidelity_objective` raises
    ``TypeError`` at the factory's field-vs-layer validation step.
    """
    if not _BO_RUNS_DIR.is_dir():
        return []
    from sweeps._bo_io import load_bo_run

    loaded: list[tuple[Path, dict]] = []
    for path in sorted(_BO_RUNS_DIR.glob("*.json")):
        try:
            loaded.append((path, load_bo_run(path)))
        except (OSError, ValueError):
            continue
    return loaded


_BO_RUNS = _load_bo_runs()


@pytest.mark.slow
class TestRegressionBOBenchmark:
    """Regression tests for per-run MF-BO benchmarks.

    For each JSON under ``sweeps/bo_runs/``, rebuild a multi-fidelity
    objective via :func:`stencil_gen.bo.make_multi_fidelity_objective`
    from the stored ``report_fields_by_layer``, evaluate at the stored
    ``best_x`` at the HF layer, and assert the recomputed scalar matches
    the stored ``best_objective`` within 1% relative tolerance.  Also
    sanity-checks the ``extras.baseline`` schema when a ``--baseline
    staged`` run is recorded.

    Skipped entirely when ``sweeps/bo_runs/`` is empty or absent.

    The MF-BO closure returns ``_BO_SENTINEL`` (``1e12``, a finite float)
    on infeasible / failed evaluations; sentinel runs from 47.7a's
    calibration benchmark (where the vanilla MF-BO composition exhausted
    its 60-eval budget without finding a feasible L7 region) compare
    cleanly under :func:`numpy.isclose` because the sentinel is a stable
    finite value and the recompute at the same ``best_x`` returns the
    same sentinel deterministically.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_absent(self):
        if not _BO_RUNS:
            pytest.skip("sweeps/bo_runs/ empty or absent")

    def test_each_run_best_x_recomputes_within_tolerance(self):
        """Each stored run's ``best_x`` reproduces its recorded objective."""
        import numpy as np

        from stencil_gen.bo import make_multi_fidelity_objective

        checked = 0
        for path, data in _BO_RUNS:
            scheme = data["scheme"]
            kernel = data["kernel"]
            report_fields = data["report_fields_by_layer"]
            hf_level = int(data["hf_level"])
            stored = float(data["best_objective"])
            best_x = np.asarray(data["best_x"], dtype=float)

            f = make_multi_fidelity_objective(scheme, kernel, report_fields)
            value, _wall, _report = f(best_x, hf_level)
            recomputed = float(value)

            assert np.isfinite(recomputed), (
                f"{path.name}: recomputed objective non-finite "
                f"({recomputed}) at best_x={best_x.tolist()} m={hf_level}"
            )
            assert np.isclose(recomputed, stored, rtol=1e-2, atol=1e-8), (
                f"{path.name}: recomputed {recomputed!r} differs from "
                f"stored {stored!r} by more than rtol=1e-2"
            )
            checked += 1
        if checked == 0:
            pytest.skip("no BO run files to check")

    def test_each_run_baseline_present_when_recorded(self):
        """When ``extras.baseline`` is recorded, sanity-check its schema.

        Per ``sweeps/bo.py::_run_staged_baseline`` the baseline payload is
        a dict produced by ``sweeps.optimize._result_to_persist_dict`` plus
        a ``method`` and ``n_evals_at_hf`` key.  On the failure path
        (47.5b.2's try/except), the dict instead carries an ``error`` key
        plus ``best_objective=None`` and ``best_x=None``.  This test
        accepts either shape and rejects only schemas missing both.
        """
        checked = 0
        for path, data in _BO_RUNS:
            extras = data.get("extras") or {}
            baseline = extras.get("baseline")
            if baseline is None:
                continue
            assert isinstance(baseline, dict), (
                f"{path.name}: extras.baseline is not a dict "
                f"({type(baseline).__name__})"
            )
            assert "method" in baseline, (
                f"{path.name}: extras.baseline missing 'method' key "
                f"(keys={sorted(baseline.keys())})"
            )
            if "error" in baseline:
                # 47.5b.2 failure-record shape.
                assert baseline["best_objective"] is None, (
                    f"{path.name}: error baseline has non-None "
                    f"best_objective={baseline['best_objective']!r}"
                )
                assert baseline["best_x"] is None, (
                    f"{path.name}: error baseline has non-None best_x"
                )
            else:
                # Success-path shape from
                # ``sweeps.optimize._result_to_persist_dict`` plus
                # ``n_evals_at_hf`` (47.5b).
                for key in ("best_x", "best_objective", "compute_time", "n_evals_at_hf"):
                    assert key in baseline, (
                        f"{path.name}: extras.baseline missing required "
                        f"key {key!r} (keys={sorted(baseline.keys())})"
                    )
            checked += 1
        if checked == 0:
            pytest.skip("no BO runs with extras.baseline recorded")
