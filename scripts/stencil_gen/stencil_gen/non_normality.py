"""Non-normality diagnostics for semi-discrete spatial operators.

Computes spectral and pseudospectral stability metrics that detect
transient growth not visible from eigenvalues alone.  Calibration bands
follow Trefethen & Embree, *Spectra and Pseudospectra* (2005), ch. 14.

Metrics provided:
- spectral abscissa: max Re(lambda) — asymptotic stability indicator
- numerical abscissa: max eigenvalue of (L + L^T)/2 — instantaneous growth rate
- Henrici departure from normality: ||LL^T - L^TL||_F / ||L||_F^2
- eigenvector condition number: cond(V) where L = V diag(lambda) V^{-1}
- pseudospectral abscissa: max Re(s) such that sigma_min(sI - L) <= epsilon
- Kreiss constant: max Re(s)/sigma_min(sI - L) for Re(s) > 0
- transient growth bound: e * Kreiss constant (Kreiss matrix theorem)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("stencil_gen.non_normality")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NonNormalityReport:
    """Collected non-normality diagnostics for a spatial operator."""

    spectral_abscissa: float
    """max Re(lambda(L)) — asymptotic stability indicator."""

    numerical_abscissa: float
    """max eigenvalue of (L + L^T)/2 — instantaneous growth rate."""

    henrici_departure: float
    """||LL^T - L^TL||_F / ||L||_F^2 — departure from normality."""

    eigenvector_condition: float
    """cond(V) where L = V diag(lambda) V^{-1}.  NaN if N too large."""

    pseudospectral_abscissae: dict[float, float]
    """epsilon -> alpha_epsilon for each requested epsilon."""

    kreiss_constant: float
    """max Re(s) / sigma_min(sI - L) over Re(s) > 0."""

    transient_growth_bound: float
    """e * kreiss_constant — upper bound on max_{t>=0} ||exp(Lt)||."""

    n: int
    """Matrix dimension."""

    compute_time: float
    """Wall-clock seconds for the full computation."""

    notes: list[str] = field(default_factory=list)
    """Diagnostic notes (convergence warnings, fallbacks, etc.)."""


# ---------------------------------------------------------------------------
# Individual metric functions (stubs — implemented in 41.8b through 41.8e)
# ---------------------------------------------------------------------------


def spectral_abscissa_sparse(L, k: int = 20, shift_invert: bool = True, rng_seed: int = 0):
    """Compute max Re(lambda) of sparse matrix L via Arnoldi iteration.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.
    k : int
        Number of eigenvalues to compute.
    shift_invert : bool
        Whether to use shift-invert mode on ArpackNoConvergence.
    rng_seed : int
        Seed for the ARPACK starting vector. scipy 1.17's ``eigs`` draws a
        fresh OS-entropy Generator per call when rng is None, making
        convergence path non-deterministic across processes for operators
        (e.g. BL42) whose eigenvalues cluster on the imaginary axis. Passing
        a fixed integer here pins that starting vector so repeated calls —
        including across subprocess invocations — return identical values.

    Returns
    -------
    tuple[float, np.ndarray]
        (max_real_part, all_computed_eigenvalues)
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigs, ArpackNoConvergence, ArpackError

    n = L.shape[0]

    # Dense fallback for small matrices
    if n <= 900 and (not sp.issparse(L) or n <= k + 1):
        A = L.toarray() if sp.issparse(L) else np.asarray(L)
        evals = np.linalg.eigvals(A)
        return float(np.max(evals.real)), evals

    # Ensure sparse format for Arnoldi
    if not sp.issparse(L):
        L = sp.csr_matrix(L)

    # Clamp k to valid range: k must be < n for eigs
    k_use = min(k, n - 2) if n > 2 else 1

    # Primary: standard Arnoldi for rightmost eigenvalues
    try:
        evals = eigs(L, k=k_use, which="LR", return_eigenvectors=False,
                     rng=rng_seed)
        return float(np.max(evals.real)), evals
    except ArpackNoConvergence as exc:
        logger.debug("eigs(which='LR') did not converge: %s", exc)
        if exc.eigenvalues is not None and len(exc.eigenvalues) > 0:
            # Some eigenvalues did converge — use them
            evals = exc.eigenvalues
            logger.debug("Using %d partially converged eigenvalues", len(evals))
            return float(np.max(evals.real)), evals

    # Retry with shift-invert around the imaginary axis
    if shift_invert:
        try:
            evals = eigs(L, k=k_use, sigma=0.0, which="LR",
                         return_eigenvectors=False, rng=rng_seed)
            return float(np.max(evals.real)), evals
        except (ArpackNoConvergence, ArpackError) as exc:
            logger.debug("Shift-invert eigs failed: %s", exc)
            if isinstance(exc, ArpackNoConvergence) and exc.eigenvalues is not None and len(exc.eigenvalues) > 0:
                evals = exc.eigenvalues
                return float(np.max(evals.real)), evals

    # Final fallback: densify if small enough
    if n <= 900:
        A = L.toarray() if sp.issparse(L) else np.asarray(L)
        evals = np.linalg.eigvals(A)
        return float(np.max(evals.real)), evals

    raise RuntimeError(
        f"spectral_abscissa_sparse: all Arnoldi attempts failed for {n}x{n} "
        f"matrix and N > 900 prevents dense fallback"
    )


def numerical_abscissa_sparse(L, rng_seed: int = 0) -> float:
    """Compute max eigenvalue of (L + L^T)/2 — the numerical abscissa.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.
    rng_seed : int
        Seed for the ARPACK starting vector. scipy 1.17's ``eigsh`` draws a
        fresh OS-entropy Generator per call when rng is None, making
        convergence path non-deterministic across processes for operators
        whose top Hermitian-part eigenvalues are clustered. Passing a fixed
        integer here pins that starting vector so repeated calls — including
        across subprocess invocations — return identical values. Only
        relevant for the sparse Arnoldi path (``n > 900``).

    Returns
    -------
    float
        The numerical abscissa (instantaneous growth rate).
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh, ArpackNoConvergence, ArpackError

    n = L.shape[0]

    # Build Hermitian part H = (L + L^T) / 2
    if sp.issparse(L):
        H = (L + L.T) / 2.0
    else:
        A = np.asarray(L, dtype=float)
        H = (A + A.T) / 2.0

    # Dense path for small matrices
    if n <= 900 and (not sp.issparse(H) or n <= 2):
        H_dense = H.toarray() if sp.issparse(H) else np.asarray(H)
        evals = np.linalg.eigvalsh(H_dense)
        return float(evals[-1])

    if not sp.issparse(H):
        H = sp.csr_matrix(H)

    k_use = min(1, n - 2) if n > 2 else 1
    try:
        evals = eigsh(H, k=k_use, which="LA", return_eigenvectors=False,
                      rng=rng_seed)
        return float(np.max(evals))
    except (ArpackNoConvergence, ArpackError) as exc:
        logger.debug("eigsh(which='LA') failed: %s", exc)
        if isinstance(exc, ArpackNoConvergence) and exc.eigenvalues is not None and len(exc.eigenvalues) > 0:
            return float(np.max(exc.eigenvalues))

    # Dense fallback
    if n <= 900:
        H_dense = H.toarray() if sp.issparse(H) else np.asarray(H)
        evals = np.linalg.eigvalsh(H_dense)
        return float(evals[-1])

    raise RuntimeError(
        f"numerical_abscissa_sparse: eigsh failed for {n}x{n} matrix "
        f"and N > 900 prevents dense fallback"
    )


def henrici_departure(L) -> float:
    """Compute Henrici departure from normality: ||LL^T - L^TL||_F / ||L||_F^2.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.

    Returns
    -------
    float
        Non-negative scalar; 0 for normal operators.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import norm as sp_norm

    if sp.issparse(L):
        LLt = L @ L.T
        LtL = L.T @ L
        diff = LLt - LtL
        numer = sp_norm(diff, "fro")
        denom = sp_norm(L, "fro") ** 2
    else:
        A = np.asarray(L, dtype=float)
        diff = A @ A.T - A.T @ A
        numer = np.linalg.norm(diff, "fro")
        denom = np.linalg.norm(A, "fro") ** 2

    if denom == 0.0:
        return 0.0
    return float(numer / denom)


def eigenvector_condition(L, small_dense_threshold: int = 900) -> float:
    """Compute condition number of the eigenvector matrix.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.
    small_dense_threshold : int
        If N > threshold, return np.nan (too expensive for dense eig).

    Returns
    -------
    float
        cond(V) where L = V diag(lambda) V^{-1}, or np.nan if N too large.
    """
    import scipy.sparse as sp

    n = L.shape[0]
    if n > small_dense_threshold:
        return np.nan

    A = L.toarray() if sp.issparse(L) else np.asarray(L, dtype=float)
    evals, V = np.linalg.eig(A)
    return float(np.linalg.cond(V))


def _sigma_field(L, s_grid: np.ndarray) -> np.ndarray:
    """Compute sigma_min(sI - L) over a grid of complex s values.

    For each complex *s* in *s_grid*, forms ``M = sI - L`` and returns
    the smallest singular value of *M*.  This is the resolvent norm
    ``||R(s, L)||^{-1}`` used to define ε-pseudospectra.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.
    s_grid : np.ndarray
        Complex-valued array of s points (arbitrary shape).

    Returns
    -------
    np.ndarray
        Array of same shape as s_grid with sigma_min values.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import svds, ArpackError

    n = L.shape[0]
    # Dense SVD is preferred for n ≤ 1200: sparse svds(which='SM') is
    # unreliable for non-symmetric operators and ARPACK failures are very
    # expensive (~5 s per point at n ≈ 1000), while dense SVD at n=961 is
    # only ~0.2 s per point.
    use_dense = n <= 1200

    if sp.issparse(L):
        L_sp = L.tocsc()
        I_sp = sp.eye(n, format="csc")
    else:
        L_dense = np.asarray(L, dtype=complex)
        if not use_dense:
            # Sparse SVD path needs L_sp/I_sp even when input was dense
            L_sp = sp.csc_matrix(L_dense)
            I_sp = sp.eye(n, format="csc")

    flat = s_grid.ravel()
    result = np.empty(flat.shape[0], dtype=float)

    for idx, s in enumerate(flat):
        if use_dense:
            if sp.issparse(L):
                M = (s * I_sp - L_sp).toarray()
            else:
                M = s * np.eye(n) - L_dense
            sv = np.linalg.svd(M, compute_uv=False)
            result[idx] = float(sv[-1])
        else:
            M_sp = s * I_sp - L_sp
            try:
                sv = svds(M_sp, k=1, which="SM",
                          return_singular_vectors=False)
                result[idx] = float(sv[0])
            except (ArpackError, Exception) as exc:
                # Fallback: densify if small enough.  Threshold raised from
                # 900 to 2000 to support BL-sized 2D operators (n ~ 961).
                if n <= 2000:
                    logger.debug("svds failed at s=%s, densifying: %s", s, exc)
                    M_dense = M_sp.toarray()
                    sv = np.linalg.svd(M_dense, compute_uv=False)
                    result[idx] = float(sv[-1])
                else:
                    raise RuntimeError(
                        f"_sigma_field: svds failed at s={s} for {n}x{n} "
                        f"matrix and N > 2000 prevents dense fallback"
                    ) from exc

    return result.reshape(s_grid.shape)


def pseudospectral_abscissa_estimate(
    L, epsilon_values, s_grid: np.ndarray
) -> dict[float, float]:
    """Estimate pseudospectral abscissa for each epsilon.

    For each epsilon, alpha_epsilon = max Re(s) such that sigma_min(sI - L) <= epsilon.
    Both functions share the same ``_sigma_field`` evaluation to avoid
    duplicate SVD cost.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.
    epsilon_values : sequence of float
        Perturbation levels.
    s_grid : np.ndarray
        Complex-valued grid for the search.

    Returns
    -------
    dict[float, float]
        epsilon -> alpha_epsilon.  -inf if no grid point satisfies.
    """
    sigma = _sigma_field(L, s_grid)
    flat_sigma = sigma.ravel()
    flat_re = s_grid.ravel().real

    result: dict[float, float] = {}
    for eps in epsilon_values:
        mask = flat_sigma <= eps
        if np.any(mask):
            result[eps] = float(np.max(flat_re[mask]))
        else:
            result[eps] = float("-inf")
    return result


def kreiss_constant_estimate(L, s_grid: np.ndarray) -> float:
    """Estimate Kreiss constant: max Re(s) / sigma_min(sI - L) for Re(s) > 0.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.
    s_grid : np.ndarray
        Complex-valued grid for the search.

    Returns
    -------
    float
        Estimated Kreiss constant.  Returns 0.0 if no grid point has Re(s) > 0.
    """
    sigma = _sigma_field(L, s_grid)
    flat_sigma = sigma.ravel()
    flat_re = s_grid.ravel().real

    # Only consider points in the open right half-plane
    mask = flat_re > 0
    if not np.any(mask):
        return 0.0

    # K(L) = max_{Re(s) > 0} Re(s) / sigma_min(sI - L)
    ratios = flat_re[mask] / np.maximum(flat_sigma[mask], 1e-300)
    return float(np.max(ratios))


# ---------------------------------------------------------------------------
# Orchestrator (41.8f)
# ---------------------------------------------------------------------------


def compute_non_normality(
    L,
    *,
    small_dense_threshold: int = 900,
    epsilon_values: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
    s_grid_params: dict | None = None,
) -> NonNormalityReport:
    """Compute all non-normality diagnostics for a spatial operator.

    Parameters
    ----------
    L : scipy.sparse matrix or dense ndarray
        The spatial operator.
    small_dense_threshold : int
        Threshold for dense-only computations (eigenvector condition).
    epsilon_values : tuple of float
        Perturbation levels for pseudospectral abscissa.
    s_grid_params : dict or None
        Parameters for the resolvent s-grid.  Keys:
        ``re_min``, ``re_max``, ``n_re``, ``im_max``, ``n_im``.
        If None, uses defaults based on the spectral abscissa.

    Returns
    -------
    NonNormalityReport
        Fully populated report with all metrics.
    """
    import math
    import time

    from scipy.sparse.linalg import ArpackNoConvergence

    t0 = time.perf_counter()
    n = L.shape[0]
    notes: list[str] = []

    # --- Spectral abscissa ---
    try:
        sa, evals = spectral_abscissa_sparse(L, k=min(20, max(1, n - 2)),
                                             shift_invert=True)
    except ArpackNoConvergence as exc:
        notes.append(f"spectral_abscissa: ArpackNoConvergence ({exc})")
        if exc.eigenvalues is not None and len(exc.eigenvalues) > 0:
            sa = float(np.max(exc.eigenvalues.real))
            evals = exc.eigenvalues
        else:
            sa = np.nan
            evals = np.array([])

    # --- Numerical abscissa ---
    try:
        na = numerical_abscissa_sparse(L)
    except Exception as exc:
        notes.append(f"numerical_abscissa: {type(exc).__name__} ({exc})")
        na = np.nan

    # --- Henrici departure ---
    hd = henrici_departure(L)

    # --- Eigenvector condition ---
    ec = eigenvector_condition(L, small_dense_threshold=small_dense_threshold)

    # --- Build the s-grid for resolvent sampling ---
    # Default grid resolution: ~30×60 for small matrices, ~8×12 for large
    # ones (n > 500) to keep runtime feasible with dense SVD (~0.2s/point
    # at n≈1000, so 96 points ≈ 19s).
    default_n_re = 8 if n > 500 else 30
    default_n_im = 12 if n > 500 else 60

    if s_grid_params is not None:
        re_min = s_grid_params.get("re_min", 1e-3)
        re_max = s_grid_params.get("re_max", 2.0 * abs(sa) + 1.0 if np.isfinite(sa) else 5.0)
        n_re = s_grid_params.get("n_re", default_n_re)
        im_max = s_grid_params.get("im_max", None)
        n_im = s_grid_params.get("n_im", default_n_im)
    else:
        re_min = 1e-3
        re_max = 2.0 * abs(sa) + 1.0 if np.isfinite(sa) else 5.0
        n_re = default_n_re
        im_max = None
        n_im = default_n_im

    # Estimate ω_max from the imaginary parts of the computed eigenvalues
    if im_max is None:
        if len(evals) > 0:
            im_max = float(np.max(np.abs(evals.imag))) + 1.0
        else:
            im_max = 10.0
    im_max = max(im_max, 1.0)  # ensure at least ±1

    re_vals = np.linspace(re_min, re_max, n_re)
    im_vals = np.linspace(-im_max, im_max, n_im)
    s_grid = re_vals[:, None] + 1j * im_vals[None, :]

    # --- Pseudospectral abscissa and Kreiss constant ---
    # Compute _sigma_field once and reuse for both
    sigma = _sigma_field(L, s_grid)
    flat_sigma = sigma.ravel()
    flat_re = s_grid.ravel().real

    # Pseudospectral abscissa
    psa: dict[float, float] = {}
    for eps in epsilon_values:
        mask = flat_sigma <= eps
        if np.any(mask):
            psa[eps] = float(np.max(flat_re[mask]))
        else:
            psa[eps] = float("-inf")

    # Kreiss constant
    rhs_mask = flat_re > 0
    if np.any(rhs_mask):
        ratios = flat_re[rhs_mask] / np.maximum(flat_sigma[rhs_mask], 1e-300)
        kc = float(np.max(ratios))
    else:
        kc = 0.0

    tgb = math.e * kc

    elapsed = time.perf_counter() - t0

    return NonNormalityReport(
        spectral_abscissa=sa,
        numerical_abscissa=na,
        henrici_departure=hd,
        eigenvector_condition=ec,
        pseudospectral_abscissae=psa,
        kreiss_constant=kc,
        transient_growth_bound=tgb,
        n=n,
        compute_time=elapsed,
        notes=notes,
    )
