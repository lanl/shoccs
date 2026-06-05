"""Tests for stencil_gen.non_normality module."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from stencil_gen.non_normality import (
    spectral_abscissa_sparse,
    numerical_abscissa_sparse,
    henrici_departure,
    eigenvector_condition,
    _sigma_field,
    pseudospectral_abscissa_estimate,
    kreiss_constant_estimate,
    compute_non_normality,
    NonNormalityReport,
)


# ---------------------------------------------------------------------------
# TestSpectralAbscissa (41.8b)
# ---------------------------------------------------------------------------


class TestSpectralAbscissa:
    """Tests for spectral_abscissa_sparse."""

    def test_diagonal_negative(self):
        """Diagonal -diag(1..50) has spectral abscissa ≈ -1."""
        diag_vals = -np.arange(1, 51, dtype=float)
        L = sp.diags(diag_vals, format="csr")
        max_re, evals = spectral_abscissa_sparse(L, k=10)
        assert max_re == pytest.approx(-1.0, abs=1e-10)
        # All eigenvalues should have negative real part
        assert np.all(evals.real < 0)

    def test_random_sparse_returns_finite(self):
        """Random sparse matrix returns a finite spectral abscissa."""
        rng = np.random.default_rng(42)
        n = 100
        # Random sparse with density ~5%
        data = rng.standard_normal(500)
        rows = rng.integers(0, n, 500)
        cols = rng.integers(0, n, 500)
        L = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        max_re, evals = spectral_abscissa_sparse(L, k=10)
        assert np.isfinite(max_re)
        assert len(evals) > 0

    def test_dense_fallback_small_n(self):
        """Dense fallback path is exercised at N=20."""
        rng = np.random.default_rng(123)
        n = 20
        A = rng.standard_normal((n, n))
        # Make it stable: shift eigenvalues to left half-plane
        A = A - 3.0 * np.eye(n)

        # Pass as dense ndarray — should trigger dense fallback since n <= 900
        # and n <= k+1 (k=20 by default, so 20 <= 21)
        max_re, evals = spectral_abscissa_sparse(A, k=20)
        assert np.isfinite(max_re)
        # Should have all n eigenvalues from dense path
        assert len(evals) == n

        # Verify against direct numpy eigvals
        expected = np.max(np.linalg.eigvals(A).real)
        assert max_re == pytest.approx(expected, abs=1e-10)

    def test_identity_spectral_abscissa(self):
        """Identity matrix has spectral abscissa = 1."""
        L = sp.eye(30, format="csr")
        max_re, evals = spectral_abscissa_sparse(L, k=5)
        assert max_re == pytest.approx(1.0, abs=1e-10)

    def test_negative_identity(self):
        """-I has spectral abscissa = -1."""
        L = -sp.eye(30, format="csr")
        max_re, evals = spectral_abscissa_sparse(L, k=5)
        assert max_re == pytest.approx(-1.0, abs=1e-10)

    def test_known_eigenvalues_tridiagonal(self):
        """Tridiagonal matrix with known eigenvalues."""
        # Symmetric tridiagonal: -2 on diagonal, 1 on off-diagonals
        # Eigenvalues: -2 + 2*cos(k*pi/(n+1)) for k=1..n
        n = 50
        diag_main = -2.0 * np.ones(n)
        diag_off = np.ones(n - 1)
        L = sp.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format="csr")

        max_re, evals = spectral_abscissa_sparse(L, k=10)

        # Largest eigenvalue is at k=1: -2 + 2*cos(pi/(n+1))
        expected_max = -2.0 + 2.0 * np.cos(np.pi / (n + 1))
        assert max_re == pytest.approx(expected_max, abs=1e-8)


# ---------------------------------------------------------------------------
# TestSpectralAbscissaDeterminism (45.6b.1)
# ---------------------------------------------------------------------------


class TestSpectralAbscissaDeterminism:
    """Tests that spectral_abscissa_sparse is cross-process deterministic.

    scipy 1.17's eigs() draws a fresh OS-entropy Generator per call when
    rng is None. For BL42-like operators whose eigenvalues cluster on the
    imaginary axis, ARPACK convergence is highly sensitive to the starting
    vector, so two calls with the same input can land on different
    representative eigenvalues. Passing a fixed rng seed pins the starting
    vector and eliminates that variance.
    """

    def _bl42_like_matrix(self, n: int = 100):
        """Build a sparse matrix whose eigenvalues cluster on the imaginary axis.

        This is the pathological regime where unpinned starting vectors
        produce different ARPACK convergence paths across processes.
        """
        rng = np.random.default_rng(2026)
        # Skew-symmetric matrices have purely imaginary eigenvalues, so this
        # is a strong stand-in for the BL42 reflecting-hyperbolic operator.
        A = rng.standard_normal((n, n))
        L = 0.5 * (A - A.T)
        return sp.csr_matrix(L)

    def test_cross_process_deterministic(self):
        """Two subprocess calls with the same rng_seed return byte-identical stdout."""
        import subprocess
        import sys

        script = (
            "import numpy as np, scipy.sparse as sp\n"
            "from stencil_gen.non_normality import spectral_abscissa_sparse\n"
            "rng = np.random.default_rng(2026)\n"
            "n = 100\n"
            "A = rng.standard_normal((n, n))\n"
            "L = sp.csr_matrix(0.5 * (A - A.T))\n"
            "max_re, _ = spectral_abscissa_sparse(L, k=10, rng_seed=0)\n"
            "print(repr(max_re))\n"
        )
        out1 = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, check=True,
        ).stdout
        out2 = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, check=True,
        ).stdout
        assert out1 == out2, f"non-deterministic:\n  run1={out1!r}\n  run2={out2!r}"

    def test_same_seed_same_result_in_process(self):
        """Two calls with the same rng_seed return bit-identical max_re."""
        L = self._bl42_like_matrix()
        max_re_a, _ = spectral_abscissa_sparse(L, k=10, rng_seed=0)
        max_re_b, _ = spectral_abscissa_sparse(L, k=10, rng_seed=0)
        assert max_re_a == max_re_b

    def test_rng_seed_override_quality_equivalent(self):
        """Different seeds may trace different Arnoldi paths but agree in quality.

        Both answers are within ARPACK's own tolerance of the true spectral
        abscissa (0 for a skew-symmetric operator), so rng_seed is a path
        selector, not a correctness knob.
        """
        L = self._bl42_like_matrix()
        max_re_0, _ = spectral_abscissa_sparse(L, k=10, rng_seed=0)
        max_re_1, _ = spectral_abscissa_sparse(L, k=10, rng_seed=1)
        # Skew-symmetric ⇒ true spectral abscissa is 0.
        assert abs(max_re_0) < 1e-8
        assert abs(max_re_1) < 1e-8

    def test_default_seed_is_zero(self):
        """Omitting rng_seed uses the same path as rng_seed=0."""
        L = self._bl42_like_matrix()
        max_re_default, _ = spectral_abscissa_sparse(L, k=10)
        max_re_explicit, _ = spectral_abscissa_sparse(L, k=10, rng_seed=0)
        assert max_re_default == max_re_explicit


# ---------------------------------------------------------------------------
# TestNumericalAbscissaDeterminism (46.1b)
# ---------------------------------------------------------------------------


class TestNumericalAbscissaDeterminism:
    """Tests that numerical_abscissa_sparse is cross-call deterministic via rng_seed.

    Mirrors TestSpectralAbscissaDeterminism but for the eigsh (Hermitian-part)
    path inside numerical_abscissa_sparse. Only triggers above the n > 900
    sparse Arnoldi threshold.
    """

    @staticmethod
    def _nonsymmetric_matrix(n: int = 1000):
        """Build an n x n non-symmetric sparse matrix whose Hermitian part has non-trivial spectrum.

        Do not use a skew-symmetric construction here: H = (L + L^T) / 2 = 0
        causes eigsh(which='LA') to fail to converge, raising RuntimeError
        before the determinism path is exercised.
        """
        rng = np.random.default_rng(2026)
        A = rng.standard_normal((n, n)) - 5.0 * np.eye(n)
        return sp.csr_matrix(A)

    def test_numerical_abscissa_sparse_deterministic_across_calls(self):
        """Two calls at the default seed return byte-identical results; rng_seed=42 differs at the ULP level."""
        L = self._nonsymmetric_matrix(1000)

        # Default seed (rng_seed=0): byte-identical across consecutive calls.
        na_a = numerical_abscissa_sparse(L)
        na_b = numerical_abscissa_sparse(L)
        assert na_a == na_b, (
            f"default-seed calls diverged: {na_a!r} vs {na_b!r}; "
            "rng_seed not threaded through eigsh"
        )

        # Different seed traces a different Arnoldi path. Difference is at
        # the ULP level (~1e-14 relative); use != not np.isclose, since
        # equality would mean the seed isn't reaching ARPACK.
        na_seed42 = numerical_abscissa_sparse(L, rng_seed=42)
        assert na_seed42 != na_a, (
            f"rng_seed=42 produced same result as default ({na_a!r}); "
            "seed not actually being threaded through ARPACK"
        )


# ---------------------------------------------------------------------------
# TestNormMetrics (41.8c)
# ---------------------------------------------------------------------------


class TestNormMetrics:
    """Tests for numerical_abscissa_sparse, henrici_departure, eigenvector_condition."""

    def test_diagonal_numerical_abscissa(self):
        """Diagonal -diag(1..50): numerical abscissa = spectral abscissa = -1."""
        diag_vals = -np.arange(1, 51, dtype=float)
        L = sp.diags(diag_vals, format="csr")
        na = numerical_abscissa_sparse(L)
        # For a real diagonal (hence symmetric) matrix, numerical abscissa
        # equals spectral abscissa: max eigenvalue of H = max eigenvalue of L.
        assert na == pytest.approx(-1.0, abs=1e-10)

    def test_diagonal_henrici_zero(self):
        """Diagonal matrix is normal: Henrici departure = 0."""
        diag_vals = -np.arange(1, 51, dtype=float)
        L = sp.diags(diag_vals, format="csr")
        h = henrici_departure(L)
        assert h == pytest.approx(0.0, abs=1e-12)

    def test_diagonal_eigenvector_condition_one(self):
        """Diagonal matrix has eigenvector condition number ≈ 1."""
        diag_vals = -np.arange(1, 51, dtype=float)
        L = sp.diags(diag_vals, format="csr")
        cond_v = eigenvector_condition(L)
        # Eigenvectors of a diagonal matrix are the identity columns,
        # so V = permutation of I, cond(V) = 1.
        assert cond_v == pytest.approx(1.0, abs=1e-8)

    def test_numerical_abscissa_dense_input(self):
        """numerical_abscissa_sparse works with dense ndarray input."""
        A = np.diag([-3.0, -2.0, -1.0])
        na = numerical_abscissa_sparse(A)
        assert na == pytest.approx(-1.0, abs=1e-10)

    def test_henrici_dense_input(self):
        """henrici_departure works with dense ndarray input."""
        A = np.diag([1.0, 2.0, 3.0])
        h = henrici_departure(A)
        assert h == pytest.approx(0.0, abs=1e-12)

    def test_eigenvector_condition_large_returns_nan(self):
        """eigenvector_condition returns NaN when N exceeds threshold."""
        L = sp.eye(1000, format="csr")
        cond_v = eigenvector_condition(L, small_dense_threshold=500)
        assert np.isnan(cond_v)

    def test_non_normal_matrix_positive_henrici(self):
        """A non-normal matrix (upper triangular shift) has Henrici > 0."""
        n = 30
        # Upper-shift matrix: L[i, i+1] = 1
        L = sp.diags([np.ones(n - 1)], [1], shape=(n, n), format="csr")
        h = henrici_departure(L)
        assert h > 0.0

    def test_non_normal_matrix_large_eigenvector_condition(self):
        """A non-normal matrix has cond(V) >> 1."""
        n = 30
        # Jordan-like: -I + nilpotent shift
        A = -np.eye(n) + n * np.diag(np.ones(n - 1), 1)
        cond_v = eigenvector_condition(A)
        assert cond_v > 10.0

    def test_numerical_abscissa_ge_spectral_abscissa(self):
        """Numerical abscissa >= spectral abscissa (fundamental inequality)."""
        rng = np.random.default_rng(99)
        A = rng.standard_normal((30, 30)) - 3.0 * np.eye(30)
        na = numerical_abscissa_sparse(A)
        sa, _ = spectral_abscissa_sparse(A)
        assert na >= sa - 1e-9

    def test_zero_matrix_henrici(self):
        """Zero matrix: henrici_departure returns 0 (guard against div-by-zero)."""
        L = sp.csr_matrix((10, 10))
        h = henrici_departure(L)
        assert h == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# TestSigmaField (41.8d)
# ---------------------------------------------------------------------------


class TestSigmaField:
    """Tests for _sigma_field: sigma_min(sI - L) over a complex grid."""

    def test_diagonal_matches_brute_force(self):
        """On a small diagonal matrix, compare _sigma_field to dense SVD."""
        diag_vals = np.array([-3.0, -2.0, -1.0, 0.5, 1.5])
        L = sp.diags(diag_vals, format="csr")
        n = L.shape[0]

        # Build a small grid
        re_vals = np.linspace(-1, 3, 5)
        im_vals = np.linspace(-2, 2, 5)
        s_grid = re_vals[:, None] + 1j * im_vals[None, :]

        result = _sigma_field(L, s_grid)
        assert result.shape == s_grid.shape

        # Brute-force reference
        for i in range(s_grid.shape[0]):
            for j in range(s_grid.shape[1]):
                s = s_grid[i, j]
                M = s * np.eye(n) - np.diag(diag_vals)
                sv_ref = np.linalg.svd(M, compute_uv=False)[-1]
                assert result[i, j] == pytest.approx(sv_ref, abs=1e-10)

    def test_dense_input(self):
        """_sigma_field works with a dense ndarray as L."""
        A = np.diag([-2.0, -1.0, 0.0])
        s_grid = np.array([0.0 + 0j, 1.0 + 0j, 0.0 + 1j])

        result = _sigma_field(A, s_grid)
        assert result.shape == s_grid.shape

        # Check each point via brute-force
        for idx, s in enumerate(s_grid):
            M = s * np.eye(3) - A
            sv_ref = np.linalg.svd(M, compute_uv=False)[-1]
            assert result[idx] == pytest.approx(sv_ref, abs=1e-10)

    def test_at_eigenvalue_sigma_min_near_zero(self):
        """sigma_min(sI - L) ≈ 0 when s is an eigenvalue of L."""
        diag_vals = np.array([-3.0, -1.0, 2.0])
        L = sp.diags(diag_vals, format="csr")

        # Evaluate exactly at each eigenvalue
        s_grid = np.array([-3.0 + 0j, -1.0 + 0j, 2.0 + 0j])
        result = _sigma_field(L, s_grid)

        for val in result:
            assert val == pytest.approx(0.0, abs=1e-10)

    def test_identity_sigma_min(self):
        """For L = I, sigma_min(sI - I) = |s - 1| (all singular values equal)."""
        n = 10
        L = sp.eye(n, format="csr")
        s_grid = np.array([0.5 + 0j, 1.0 + 0j, 2.0 + 1j, -1.0 + 0.5j])

        result = _sigma_field(L, s_grid)
        for idx, s in enumerate(s_grid):
            expected = abs(s - 1.0)
            assert result[idx] == pytest.approx(expected, abs=1e-10)

    def test_2d_grid_shape_preserved(self):
        """Output shape matches a 2D input grid shape."""
        L = sp.diags([-1.0, -2.0, -3.0], format="csr")
        s_grid = np.zeros((4, 7), dtype=complex)
        s_grid.real = np.linspace(-1, 1, 4)[:, None]
        s_grid.imag = np.linspace(-3, 3, 7)[None, :]

        result = _sigma_field(L, s_grid)
        assert result.shape == (4, 7)

    def test_dense_large_input(self):
        """Dense ndarray with n > 900 does not crash (41.8d-followup).

        Previously, _sigma_field would crash with NameError on I_sp/L_sp
        when given a dense matrix with n > 900, because those variables
        were only set in the sp.issparse(L) branch.
        """
        n = 901
        # Diagonal matrix: eigenvalues are -1, -2, ..., -901
        diag_vals = -np.arange(1, n + 1, dtype=float)
        L_dense = np.diag(diag_vals)

        # Avoid s values that coincide with eigenvalues (sigma_min = 0
        # is poorly handled by sparse svds).  Use points away from the
        # spectrum so sigma_min is comfortably positive.
        s_grid = np.array([0.5 + 0j, 0.0 + 3j, -0.5 + 1j])
        result = _sigma_field(L_dense, s_grid)

        # Verify against brute-force dense SVD
        for idx, s in enumerate(s_grid):
            M = s * np.eye(n) - L_dense
            sv_ref = np.linalg.svd(M, compute_uv=False)[-1]
            assert result[idx] == pytest.approx(sv_ref, abs=1e-4)

    def test_sparse_medium_size(self):
        """Sparse path exercised for N > 200 (if we make L sparse and large)."""
        n = 250
        # Stable tridiagonal
        diag_main = -2.0 * np.ones(n)
        diag_off = np.ones(n - 1)
        L = sp.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format="csc")

        # Small grid — just a few points to keep test fast
        s_grid = np.array([0.0 + 0j, 0.0 + 1j, -1.0 + 0j])
        result = _sigma_field(L, s_grid)

        # Compare to dense
        L_dense = L.toarray()
        for idx, s in enumerate(s_grid):
            M = s * np.eye(n) - L_dense
            sv_ref = np.linalg.svd(M, compute_uv=False)[-1]
            assert result[idx] == pytest.approx(sv_ref, abs=1e-6)


# ---------------------------------------------------------------------------
# TestPseudoAndKreiss (41.8e)
# ---------------------------------------------------------------------------


class TestPseudoAndKreiss:
    """Tests for pseudospectral_abscissa_estimate and kreiss_constant_estimate.

    Uses the Wilkinson bidiagonal matrix  L = -I + N*upper_shift  at N=30,
    which is spectrally stable (spectral abscissa = -1) but highly non-normal
    (numerical abscissa >> 0, Kreiss constant >> 1).
    """

    @staticmethod
    def _wilkinson_bidiagonal(n: int = 30):
        """Build L = -I + n * upper-shift (bidiagonal).

        Returns a dense ndarray because this matrix is so non-normal that
        Arnoldi (ARPACK) fails to converge to the true eigenvalues.  Dense
        eigensolver handles it correctly.
        """
        return -np.eye(n) + float(n) * np.diag(np.ones(n - 1), 1)

    @staticmethod
    def _make_rhs_grid(re_max: float = 2.0, n_re: int = 30,
                       im_max: float = 5.0, n_im: int = 40):
        """Build a right-half-plane grid for resolvent sampling."""
        re_vals = np.linspace(1e-3, re_max, n_re)
        im_vals = np.linspace(-im_max, im_max, n_im)
        return re_vals[:, None] + 1j * im_vals[None, :]

    def test_wilkinson_spectral_abscissa_minus_one(self):
        """Wilkinson bidiagonal has spectral abscissa ≈ -1."""
        L = self._wilkinson_bidiagonal(30)
        sa, _ = spectral_abscissa_sparse(L)
        assert sa == pytest.approx(-1.0, abs=1e-8)

    def test_wilkinson_numerical_abscissa_strongly_positive(self):
        """Wilkinson bidiagonal has strongly positive numerical abscissa."""
        L = self._wilkinson_bidiagonal(30)
        na = numerical_abscissa_sparse(L)
        assert na > 5.0  # much larger than spectral abscissa

    def test_pseudospectral_abscissa_monotone_in_epsilon(self):
        """alpha_eps is non-decreasing in epsilon (larger perturbation => larger pseudospectrum)."""
        L = self._wilkinson_bidiagonal(30)
        s_grid = self._make_rhs_grid()
        epsilons = [1e-4, 1e-3, 1e-2, 1e-1]
        result = pseudospectral_abscissa_estimate(L, epsilons, s_grid)

        # Monotonicity: alpha_eps1 <= alpha_eps2 when eps1 < eps2
        prev = float("-inf")
        for eps in epsilons:
            assert result[eps] >= prev - 1e-12
            prev = result[eps]

    def test_pseudospectral_abscissa_returns_neg_inf_for_tiny_epsilon(self):
        """For extremely small epsilon, no grid point satisfies => -inf."""
        L = self._wilkinson_bidiagonal(30)
        # Grid far from spectrum: sigma_min is large at all points
        s_grid = np.array([10.0 + 0j, 10.0 + 5j])
        result = pseudospectral_abscissa_estimate(L, [1e-20], s_grid)
        assert result[1e-20] == float("-inf")

    def test_pseudospectral_abscissa_at_eigenvalue(self):
        """alpha_eps includes the eigenvalue vicinity for large enough epsilon."""
        # Diagonal L: eigenvalues at -1, -2, -3
        L = sp.diags([-1.0, -2.0, -3.0], format="csr")
        # Grid includes a point near eigenvalue -1
        s_grid = np.array([-1.0 + 0j, -0.99 + 0j, -0.5 + 0j, 0.5 + 0j])
        result = pseudospectral_abscissa_estimate(L, [0.1], s_grid)
        # sigma_min(sI - L) at s = -0.99 is min(|0.01|, |1.01|, |2.01|) = 0.01 < 0.1
        assert result[0.1] >= -1.0

    def test_kreiss_constant_wilkinson_large(self):
        """Wilkinson bidiagonal has Kreiss constant >> 1 (strong transient growth)."""
        L = self._wilkinson_bidiagonal(30)
        s_grid = self._make_rhs_grid()
        K = kreiss_constant_estimate(L, s_grid)
        assert K > 10.0  # highly non-normal => large Kreiss constant

    def test_kreiss_constant_stable_diagonal(self):
        """Stable normal matrix has Kreiss constant ≈ 1."""
        # L = -I: all eigenvalues at -1, normal matrix
        n = 30
        L = -sp.eye(n, format="csr")
        # Grid in right half-plane: Re(s) > 0
        re_vals = np.linspace(0.01, 2.0, 20)
        im_vals = np.linspace(-3.0, 3.0, 30)
        s_grid = re_vals[:, None] + 1j * im_vals[None, :]
        K = kreiss_constant_estimate(L, s_grid)
        # For a normal matrix with spectral abscissa -1:
        # sigma_min(sI - L) = min|s - lambda_i| = |s + 1| >= Re(s) + 1
        # so Re(s) / sigma_min <= Re(s) / (Re(s) + 1) < 1
        assert K < 1.0

    def test_kreiss_constant_no_rhs_returns_zero(self):
        """If no grid points have Re(s) > 0, Kreiss constant is 0."""
        L = sp.eye(5, format="csr")
        s_grid = np.array([-1.0 + 0j, -0.5 + 1j])  # all Re(s) <= 0
        K = kreiss_constant_estimate(L, s_grid)
        assert K == 0.0

    def test_shared_sigma_field_consistency(self):
        """Both functions produce results consistent with direct _sigma_field call."""
        L = self._wilkinson_bidiagonal(30)
        s_grid = self._make_rhs_grid(re_max=1.5, n_re=15, im_max=3.0, n_im=20)

        sigma = _sigma_field(L, s_grid)
        flat_sigma = sigma.ravel()
        flat_re = s_grid.ravel().real

        # Manual pseudospectral abscissa for eps=0.1
        eps = 0.1
        mask = flat_sigma <= eps
        expected_alpha = float(np.max(flat_re[mask])) if np.any(mask) else float("-inf")
        result = pseudospectral_abscissa_estimate(L, [eps], s_grid)
        assert result[eps] == pytest.approx(expected_alpha, abs=1e-12)

        # Manual Kreiss constant
        rhs_mask = flat_re > 0
        expected_K = float(np.max(flat_re[rhs_mask] / np.maximum(flat_sigma[rhs_mask], 1e-300)))
        K = kreiss_constant_estimate(L, s_grid)
        assert K == pytest.approx(expected_K, abs=1e-12)


# ---------------------------------------------------------------------------
# TestComputeNonNormality (41.8f)
# ---------------------------------------------------------------------------


class TestComputeNonNormality:
    """Tests for compute_non_normality orchestrator."""

    def test_small_diagonal(self):
        """Diagonal -diag(1..30): normal matrix, all diagnostics consistent."""
        diag_vals = -np.arange(1, 31, dtype=float)
        L = sp.diags(diag_vals, format="csr")

        report = compute_non_normality(L)
        assert isinstance(report, NonNormalityReport)
        assert report.n == 30

        # Spectral abscissa = max eigenvalue = -1
        assert report.spectral_abscissa == pytest.approx(-1.0, abs=1e-8)

        # For a normal matrix, numerical abscissa == spectral abscissa
        assert report.numerical_abscissa == pytest.approx(-1.0, abs=1e-8)

        # Henrici departure ≈ 0 for normal matrix
        assert report.henrici_departure == pytest.approx(0.0, abs=1e-10)

        # Eigenvector condition ≈ 1 for normal matrix
        assert report.eigenvector_condition == pytest.approx(1.0, abs=1e-6)

        # Kreiss constant should be < 1 for stable normal matrix
        assert report.kreiss_constant < 1.0

        # Transient growth bound = e * K
        import math
        assert report.transient_growth_bound == pytest.approx(
            math.e * report.kreiss_constant, rel=1e-12
        )

        # compute_time recorded
        assert report.compute_time > 0.0

        # Cross-check: numerical_abscissa >= spectral_abscissa
        assert report.numerical_abscissa >= report.spectral_abscissa - 1e-9

    def test_non_normal_wilkinson(self):
        """Wilkinson bidiagonal: non-normal, large Kreiss constant."""
        n = 30
        L = -np.eye(n) + float(n) * np.diag(np.ones(n - 1), 1)

        report = compute_non_normality(L)

        # Spectral abscissa ≈ -1 (all eigenvalues at -1)
        assert report.spectral_abscissa == pytest.approx(-1.0, abs=1e-6)

        # Numerical abscissa strongly positive (non-normal transient growth)
        assert report.numerical_abscissa > 5.0

        # Henrici departure > 0
        assert report.henrici_departure > 0.0

        # Kreiss constant >> 1
        assert report.kreiss_constant > 10.0

        # Transient growth bound >> 1
        assert report.transient_growth_bound > 10.0

        # Cross-check: numerical_abscissa >= spectral_abscissa
        assert report.numerical_abscissa >= report.spectral_abscissa - 1e-9

        # Pseudospectral abscissae monotone in epsilon
        prev = float("-inf")
        for eps in sorted(report.pseudospectral_abscissae.keys()):
            assert report.pseudospectral_abscissae[eps] >= prev - 1e-12
            prev = report.pseudospectral_abscissae[eps]

    def test_custom_s_grid_params(self):
        """Custom s_grid_params are respected."""
        L = sp.diags([-2.0, -1.0, -0.5], format="csr")
        report = compute_non_normality(
            L,
            s_grid_params={"re_min": 0.1, "re_max": 3.0, "n_re": 10,
                           "im_max": 2.0, "n_im": 15},
            epsilon_values=(1e-2, 1e-1),
        )
        assert isinstance(report, NonNormalityReport)
        assert len(report.pseudospectral_abscissae) == 2
        assert 1e-2 in report.pseudospectral_abscissae
        assert 1e-1 in report.pseudospectral_abscissae

    def test_cross_check_numerical_ge_spectral(self):
        """Numerical abscissa >= spectral abscissa for random matrix."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((40, 40)) - 3.0 * np.eye(40)
        report = compute_non_normality(A)
        assert report.numerical_abscissa >= report.spectral_abscissa - 1e-9

    @pytest.mark.slow
    def test_compute_non_normality_on_bl_sized_matrix(self):
        """BL-sized 2D test matrix via kron(D, I) + kron(I, D) at n=31.

        Builds a representative 2D advection operator and verifies that
        compute_non_normality completes within 30 seconds with all fields
        populated.
        """
        from stencil_gen.phs import build_diff_matrix_rbf

        n_1d = 31
        D = build_diff_matrix_rbf(n_1d, p=2, q=3, epsilon=3.0,
                                  kernel="tension", nu=1, nextra=0)
        I_1d = np.eye(n_1d)
        # 2D operator: D_x (x) I_y + I_x (x) D_y
        L_2d = np.kron(D, I_1d) + np.kron(I_1d, D)

        # Convert to sparse for the orchestrator
        L_sp = sp.csr_matrix(L_2d)
        n_2d = n_1d * n_1d  # 961

        report = compute_non_normality(L_sp, small_dense_threshold=900)

        assert report.n == n_2d
        assert report.compute_time < 30.0

        # Spectral abscissa should be finite
        assert np.isfinite(report.spectral_abscissa)

        # Numerical abscissa should be finite
        assert np.isfinite(report.numerical_abscissa)

        # Henrici departure should be finite and positive (non-normal operator)
        assert np.isfinite(report.henrici_departure)
        assert report.henrici_departure > 0.0

        # Eigenvector condition should be NaN (N=961 > threshold=900)
        assert np.isnan(report.eigenvector_condition)

        # Kreiss constant should be finite
        assert np.isfinite(report.kreiss_constant)

        # Transient growth bound should be finite
        assert np.isfinite(report.transient_growth_bound)

        # All pseudospectral abscissae should be finite or -inf
        for eps, alpha in report.pseudospectral_abscissae.items():
            assert np.isfinite(alpha) or alpha == float("-inf")

        # Cross-check
        assert report.numerical_abscissa >= report.spectral_abscissa - 1e-9
