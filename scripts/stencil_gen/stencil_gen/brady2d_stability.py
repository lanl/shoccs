"""Layered stability scoring for the Brady-Livescu 2D benchmark.

Implements a multi-layer analytical stability pipeline for the Brady & Livescu
2019 §4.3 two-dimensional varying-coefficient scalar advection test.  Each layer
is strictly cheaper than the next, allowing early rejection of unstable schemes.

Layers:
    L1 — Interior + boundary group velocity error (1D, per direction)
    L2 — Rigorous GKS Kreiss determinant test
    L3 — 1D eigenvalue max Re(lambda(-D_bc)) at multiple N
    L4 — Per-point local GV error on the 2D varying-coefficient field
    L5 — 2D anisotropy over the coefficient field
    L6 — Non-normality diagnostics (spectral/numerical abscissa, Kreiss constant)
    L7 — Sparse 2D Arnoldi eigenvalue of the full BL operator
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from stencil_gen.gks_kreiss import BoundaryRow, KreissResult, kreiss_stability_check
from stencil_gen.group_velocity import (
    GroupVelocityProfile,
    anisotropy_over_coefficient_field,
    boundary_group_velocity,
    boundary_group_velocity_classical,
    interior_group_velocity,
    local_group_velocity_2d_varying,
    max_local_gv_error_2d,
)
from stencil_gen.non_normality import (
    NonNormalityReport,
    compute_non_normality,
    spectral_abscissa_sparse,
)
from stencil_gen.phs import (
    build_diff_matrix_rbf,
    stability_eigenvalue,
    stability_eigenvalue_from_matrix,
)
from stencil_gen.cpp_bridge import BridgeResult, run_cpp_brady2d

logger = logging.getLogger("stencil_gen.brady2d_stability")

# Scheme parameters (duplicated from sweeps._common to avoid circular dep)
_SCHEME_PARAMS = {
    "E2": {"p": 1, "q": 1, "nextra": 1, "nu": 1},
    "E4": {"p": 2, "q": 3, "nextra": 0, "nu": 1},
}

# Layer-1 thresholds
L1_TOL = 0.05  # 5% dispersion error

# Layer-3 threshold: max Re(eigenvalue of -D_bc) must be non-positive
STABILITY_TOL = 1e-10

# Layer-4 threshold: 10% — looser than L1 because the varying-coefficient
# scaling amplifies the baseline dispersion error.
L4_TOL = 0.1

# Layer-5 threshold: 5% anisotropy error projected onto local propagation direction.
L5_TOL = 0.05

# BL §4.2 tolerance: continuous spectrum is exactly imaginary; tight tolerance.
BL42_TOL = 1e-10

# Fraction of the resolved band over which to measure max |gv_error|.
# Using 10% of the cutoff restricts evaluation to very well-resolved
# wavenumbers where even boundary stencils (especially the outermost
# one-sided row) are expected to be accurate.  This makes L1 a coarse
# filter that only rejects schemes with fundamentally broken dispersion.
_RESOLVED_FRAC = 0.1


@dataclass
class StabilityReport:
    """Result of the layered stability analysis.

    Each layer populates its corresponding field with a dict of per-layer
    metrics.  If a layer is not run (skipped or short-circuited), its field
    remains None.

    Numeric ``layerN`` are the primary cascade; ``layer_bl42`` runs during
    the L3 tier (parallel 1D eigenvalue check on the Brady-Livescu §4.2
    reflecting-hyperbolic model problem).
    """

    layer1: dict | None = None
    layer2: KreissResult | None = None
    layer3: dict | None = None
    layer4: dict | None = None
    layer5: dict | None = None
    layer6: dict | None = None  # reserved for standalone non-normality
    layer7: dict | None = None
    layer8: dict | None = None
    layer_bl42: dict | None = None
    non_normality: NonNormalityReport | None = None
    kreiss: KreissResult | None = None
    overall_verdict: Literal["pass", "fail", "unknown"] = "unknown"
    failed_layer: int | None = None
    failed_reason: str = ""
    compute_time: float = 0.0

    @classmethod
    def empty(cls) -> StabilityReport:
        """Create an empty report with all layers set to None."""
        return cls()

    def __str__(self) -> str:
        """Per-layer summary table."""
        lines = ["Brady-Livescu 2D Stability Report"]
        lines.append("=" * 60)

        # Layer 1
        if self.layer1 is not None:
            d = self.layer1
            status = "PASS" if d.get("boundary_gv_err", 1.0) <= L1_TOL else "FAIL"
            lines.append(
                f"L1  GV error        : {status:4s}  "
                f"boundary_gv_err={d.get('boundary_gv_err', float('nan')):.4e}"
            )

        # Layer 2 (Kreiss)
        if self.layer2 is not None:
            kr = self.layer2
            status = "PASS" if kr.is_stable else "FAIL"
            lines.append(
                f"L2  Kreiss GKS      : {status:4s}  "
                f"sigma_min={kr.witness_sigma_min:.4e}  "
                f"verdict={kr.imaginary_axis_perturbation_verdict}"
            )

        # Layer 3
        if self.layer3 is not None:
            d = self.layer3
            mse = d.get("max_stab_eig", float("nan"))
            status = "PASS" if mse <= STABILITY_TOL else "FAIL"
            lines.append(
                f"L3  1D eigenvalue   : {status:4s}  "
                f"max_stab_eig={mse:.4e}"
            )

        # Layer 3r (BL §4.2 reflecting hyperbolic)
        if self.layer_bl42 is not None:
            d = self.layer_bl42
            msa = d.get("max_spectral_abscissa", float("nan"))
            status = "PASS" if msa <= BL42_TOL else "FAIL"
            per_n = ", ".join(
                f"{n}:{v:.2e}"
                for n, v in sorted(d.get("spectral_abscissa_by_n", {}).items())
            )
            lines.append(
                f"L3r BL42 reflecting: {status:4s}  "
                f"max_re={msa:.4e}  per_n=[{per_n}]"
            )

        # Layer 4
        if self.layer4 is not None:
            d = self.layer4
            err = d.get("max_local_gv_error", float("nan"))
            status = "PASS" if err <= L4_TOL else "FAIL"
            lines.append(
                f"L4  Local GV 2D     : {status:4s}  "
                f"max_local_gv_error={err:.4e}"
            )

        # Layer 5
        if self.layer5 is not None:
            d = self.layer5
            err = d.get("max_aligned_error", float("nan"))
            status = "PASS" if err <= L5_TOL else "FAIL"
            lines.append(
                f"L5  Anisotropy      : {status:4s}  "
                f"max_aligned_error={err:.4e}"
            )

        # Layer 6 (standalone 1D non-normality)
        if self.layer6 is not None:
            d = self.layer6
            sa = d.get("spectral_abscissa", float("nan"))
            tgb = d.get("transient_growth_bound", float("nan"))
            sa_ok = sa <= STABILITY_TOL
            tgb_ok = tgb <= L6_TRANSIENT_GROWTH_TOL
            status = "PASS" if (sa_ok and tgb_ok) else "FAIL"
            lines.append(
                f"L6  Non-normality   : {status:4s}  "
                f"kreiss_K={d.get('kreiss_constant', float('nan')):.2f}  "
                f"tgb={tgb:.2f}  "
                f"henrici={d.get('henrici_departure', float('nan')):.4e}"
            )

        # Layer 7
        if self.layer7 is not None:
            d = self.layer7
            msa = d.get("max_spectral_abscissa", float("nan"))
            status = "PASS" if msa <= L7_TOL else "FAIL"
            lines.append(
                f"L7  2D eigenvalue   : {status:4s}  "
                f"max_spectral_abscissa={msa:.4e}"
            )

        # Non-normality (2D, from L7)
        if self.non_normality is not None and self.layer7 is not None:
            nn = self.non_normality
            lines.append(
                f"L7+ Non-normality  :       "
                f"kreiss_K={nn.kreiss_constant:.2f}  "
                f"tgb={nn.transient_growth_bound:.2f}  "
                f"henrici={nn.henrici_departure:.4e}"
            )

        # Layer 8
        if self.layer8 is not None:
            d = self.layer8
            stable = bool(d.get("stable", False))
            linf = float(d.get("final_linf", float("nan")))
            status = "PASS" if (stable and linf <= L8_FINAL_LINF_TOL) else "FAIL"
            lines.append(
                f"L8  C++ simulation  : {status:4s}  "
                f"final_linf={linf:.4e}  "
                f"wall={float(d.get('wall_time_s', 0.0)):.2f}s"
            )

        lines.append("-" * 60)
        lines.append(
            f"Overall: {self.overall_verdict.upper()}"
            + (f"  (failed at layer {self.failed_layer}: {self.failed_reason})"
               if self.failed_layer is not None else "")
        )
        lines.append(f"Compute time: {self.compute_time:.2f}s")
        return "\n".join(lines)


def _gv_error_scalar(profile: GroupVelocityProfile) -> float:
    """Max absolute GV error in the resolved portion of the spectrum.

    Evaluates over xi in (0, cutoff_xi * _RESOLVED_FRAC], giving a
    conservative measure that focuses on wavenumbers the scheme is expected
    to resolve well.
    """
    xi_max = profile.cutoff_xi * _RESOLVED_FRAC
    mask = (profile.xi > 0) & (profile.xi <= xi_max)
    if not np.any(mask):
        return 1.0  # no resolved wavenumbers → maximum error
    return float(np.max(np.abs(profile.gv_error[mask])))


def _derive_classical_boundary(p: int, nu: int, alpha_list: list[float]):
    """Derive classical boundary rows with conservation and substitute alphas.

    Parameters
    ----------
    p : int
        Interior half-bandwidth.
    nu : int
        Derivative order.
    alpha_list : list[float]
        Alpha values ordered by symbol index (alpha_0, alpha_1, ...).

    Returns
    -------
    (updated_rows, alpha_values_dict)
        updated_rows: list[BoundaryRow] with conservation applied.
        alpha_values_dict: {Symbol: float} mapping for substitution.
    """
    from stencil_gen.boundary import derive_boundary
    from stencil_gen.conservation import build_conservation_system, solve_conservation

    result = derive_boundary(p=p, nu=nu, s=0)
    equations, w_syms, last_free = build_conservation_system(
        result.r, result.t, p, result.rows, result.interior_coeffs,
    )
    _, updated_rows = solve_conservation(
        equations, w_syms, last_free, result.all_free_params, result.rows,
    )
    alpha_values = dict(zip(result.all_free_params, alpha_list))
    return updated_rows, alpha_values


def layer1_interior_boundary_gv(
    scheme: str,
    kernel: str,
    params: dict,
    n_xi: int = 200,
) -> dict:
    """L1: Interior + boundary group velocity error (1D, per direction).

    Checks dispersion quality by computing GV error profiles for the interior
    and boundary stencils, then reducing each to a single scalar (max absolute
    error over the well-resolved portion of the spectrum).

    Parameters
    ----------
    scheme : str
        Scheme name ("E2" or "E4").
    kernel : str
        Kernel type ("classical", "tension", "gaussian", "multiquadric").
    params : dict
        Kernel-specific parameters.  For classical: {"alpha": [float, ...]}.
        For RBF kernels: {"sigma": float} or {"epsilon": float}.
    n_xi : int
        Number of wavenumber samples in [0.01, pi].

    Returns
    -------
    dict with keys:
        interior_gv_err_x : float
            Max |gv_error| for interior stencil over resolved wavenumbers.
        interior_gv_err_y : float
            Same as x (Cartesian grid → identical stencil in both directions).
        boundary_gv_err : float
            Max over all boundary rows of max |gv_error| over resolved band.
        cutoff_fraction : float
            min(cutoff_xi) over boundary rows / pi.
    """
    sp = _SCHEME_PARAMS[scheme]
    p, q, nextra, nu = sp["p"], sp["q"], sp["nextra"], sp["nu"]
    xi_array = np.linspace(0.01, np.pi, n_xi)

    # Interior GV profile (same stencil for x and y on Cartesian grid)
    interior_prof = interior_group_velocity(p, nu, xi_array)
    interior_err = _gv_error_scalar(interior_prof)

    # Boundary GV profiles
    if kernel == "classical":
        alpha_list = params["alpha"]
        boundary_rows, alpha_values = _derive_classical_boundary(p, nu, alpha_list)
        boundary_profiles = boundary_group_velocity_classical(
            boundary_rows, alpha_values, order=q, xi_array=xi_array,
        )
    else:
        # RBF kernels: sigma for tension, epsilon for gaussian/multiquadric
        sigma = params.get("sigma", params.get("epsilon", 0.0))
        boundary_profiles = boundary_group_velocity(
            p, q, nextra, nu, sigma, kernel, xi_array,
        )

    # Scalar reductions
    boundary_errs = [_gv_error_scalar(prof) for prof in boundary_profiles.values()]
    max_boundary_err = max(boundary_errs) if boundary_errs else 0.0

    cutoffs = [prof.cutoff_xi for prof in boundary_profiles.values()]
    min_cutoff = min(cutoffs) if cutoffs else 0.0
    cutoff_frac = min_cutoff / np.pi

    return {
        "interior_gv_err_x": interior_err,
        "interior_gv_err_y": interior_err,
        "boundary_gv_err": max_boundary_err,
        "cutoff_fraction": cutoff_frac,
    }


def _extract_stencil_data(
    D: np.ndarray, p: int,
) -> tuple[np.ndarray, np.ndarray, list[BoundaryRow]]:
    """Extract interior weights/offsets and boundary rows from a D matrix.

    Uses the middle row for interior weights and the first p rows as
    boundary rows (matching the number of admissible kappa roots for a
    2p+1-point centered stencil in the GKS framework).

    Parameters
    ----------
    D : np.ndarray
        Full N×N differentiation matrix.
    p : int
        Interior half-bandwidth.

    Returns
    -------
    (interior_weights, interior_offsets, boundary_rows)
    """
    n = D.shape[0]
    mid = n // 2
    offsets = np.arange(-p, p + 1)
    interior_weights = D[mid, mid + offsets]

    boundary_rows: list[BoundaryRow] = []
    for i in range(p):
        brow = D[i, :]
        bcols = np.nonzero(np.abs(brow) > 1e-15)[0]
        boundary_rows.append((brow[bcols], bcols))

    return interior_weights, offsets, boundary_rows


def layer2_kreiss_gks(
    scheme: str,
    kernel: str,
    params: dict,
    n: int = 20,
) -> KreissResult:
    """L2: rigorous GKS Kreiss determinant stability check.

    Builds the 1D differentiation matrix, extracts the interior stencil and
    boundary rows, and runs the full Kreiss determinant sweep to test for
    boundary-closure instability.

    Parameters
    ----------
    scheme : str
        Scheme name ("E2" or "E4").
    kernel : str
        Kernel type ("classical", "tension", "gaussian", "multiquadric").
    params : dict
        Kernel-specific parameters.
    n : int
        Grid size for the 1D differentiation matrix.

    Returns
    -------
    KreissResult
        Full Kreiss determinant result including stability verdict.
    """
    sp = _SCHEME_PARAMS[scheme]
    p, q, nextra, nu = sp["p"], sp["q"], sp["nextra"], sp["nu"]

    if kernel == "classical":
        D = _build_classical_diff_matrix(n, p, nu, params["alpha"])
    else:
        epsilon = params.get("sigma", params.get("epsilon", 0.0))
        D = build_diff_matrix_rbf(n, p, q, epsilon, kernel, nu, nextra)

    interior_weights, interior_offsets, boundary_rows = _extract_stencil_data(D, p)

    return kreiss_stability_check(
        interior_weights, interior_offsets, boundary_rows,
        s_grid_params={"s_max": 10.0, "n_radial": 30, "n_imag": 80, "imag_max": 15.0},
    )


def _build_classical_diff_matrix(
    n: int,
    p: int,
    nu: int,
    alpha_list: list[float],
) -> np.ndarray:
    """Build an n×n differentiation matrix for the classical-alpha family.

    Uses the TEMO boundary derivation with conservation enforcement,
    substitutes alpha values, and assembles the full matrix with
    antisymmetric (nu=1) right boundary closure.

    Requires p >= 2 (E4+).  For E2 (p=1) the boundary closure has zero free
    alpha parameters — ``derive_boundary(p=1)`` computes a negative symbol
    count and crashes.  Use the RBF path (``kernel="tension"``, ``sigma=0.0``)
    for E2 instead.
    """
    from stencil_gen.interior import derive_interior, full_gamma_array

    boundary_rows, alpha_values = _derive_classical_boundary(p, nu, alpha_list)

    # Dimensions come from the boundary derivation itself
    r = len(boundary_rows)
    t = len(boundary_rows[0].coefficients)

    # Build numeric boundary block
    B = np.zeros((r, t))
    for i, brow in enumerate(boundary_rows):
        for j, coeff in enumerate(brow.coefficients):
            B[i, j] = float(coeff.subs(alpha_values))

    # Interior weights
    interior_coeffs = derive_interior(0, p, nu)
    interior_w = [float(c) for c in full_gamma_array(interior_coeffs)]

    # Assemble full matrix
    D = np.zeros((n, n))
    # Left boundary
    for i in range(r):
        for j in range(t):
            D[i, j] = B[i, j]
    # Interior
    for i in range(r, n - r):
        for k_idx, j in enumerate(range(i - p, i + p + 1)):
            D[i, j] = interior_w[k_idx]
    # Right boundary (antisymmetric for nu=1)
    sign = -1 if nu % 2 == 1 else 1
    for i in range(r):
        row = n - 1 - i
        for j in range(t):
            D[row, n - 1 - j] = sign * B[i, j]
    return D


def layer3_1d_eigenvalue(
    scheme: str,
    kernel: str,
    params: dict,
    n_values: tuple[int, ...] = (20, 40, 80),
) -> dict:
    """L3: 1D eigenvalue stability check at multiple grid sizes.

    For each grid size n, computes the maximum real part of eigenvalues of
    -D_bc (the semi-discrete advection operator with inflow BC removed).
    A non-positive value means the 1D constant-coefficient scheme is stable.

    Parameters
    ----------
    scheme : str
        Scheme name ("E2" or "E4").
    kernel : str
        Kernel type ("classical", "tension", "gaussian", "multiquadric").
    params : dict
        Kernel-specific parameters.  For classical: {"alpha": [float, ...]}.
        For RBF kernels: {"sigma": float} or {"epsilon": float}.
    n_values : tuple[int, ...]
        Grid sizes at which to evaluate stability.

    Returns
    -------
    dict with keys:
        eigenvalues : dict[int, float]
            {n: max_real_eigenvalue} for each grid size.
        max_stab_eig : float
            Maximum over all grid sizes.
    """
    sp = _SCHEME_PARAMS[scheme]
    p, q, nextra, nu = sp["p"], sp["q"], sp["nextra"], sp["nu"]

    eigenvalues = {}
    for n in n_values:
        if kernel == "classical":
            alpha_list = params["alpha"]
            D = _build_classical_diff_matrix(n, p, nu, alpha_list)
            se = stability_eigenvalue_from_matrix(D)
        else:
            epsilon = params.get("sigma", params.get("epsilon", 0.0))
            se = stability_eigenvalue(n, p, q, epsilon, kernel, nu, nextra)
        eigenvalues[n] = se

    return {
        "eigenvalues": eigenvalues,
        "max_stab_eig": max(eigenvalues.values()),
    }


# ---------------------------------------------------------------------------
# BL §4.2 reflecting-hyperbolic block operator
# ---------------------------------------------------------------------------


def build_bl42_operator(D: np.ndarray) -> "scipy.sparse.csr_matrix":
    """Build the reduced 2×2 block operator for Brady-Livescu §4.2.

    Constructs L = [[0, D/h], [D/h, 0]] for the coupled hyperbolic system
    u_t = v_x, v_t = u_x on [0, 1], then removes the Dirichlet DOFs
    (u at x=0, v at x=1) to produce the (2N-2) × (2N-2) reduced operator.

    Parameters
    ----------
    D : np.ndarray
        1D differentiation matrix of shape (N, N) on a unit-spaced grid.

    Returns
    -------
    scipy.sparse.csr_matrix
        Reduced operator of shape (2N-2, 2N-2).
    """
    import scipy.sparse as sp

    from stencil_gen.benchmarks.brady_livescu_4_2 import L_DOMAIN

    N = D.shape[0]
    h = L_DOMAIN / (N - 1)
    D_scaled = sp.csr_matrix(D) / h

    Z = sp.csr_matrix((N, N))
    L_full = sp.bmat([[Z, D_scaled], [D_scaled, Z]], format="csr")

    keep = np.concatenate([np.arange(1, N), np.arange(N, 2 * N - 1)])
    L_red = L_full[np.ix_(keep, keep)]
    return L_red.tocsr()


def layer_bl42_reflecting_hyperbolic(
    scheme: str,
    kernel: str,
    params: dict,
    n_values: tuple[int, ...] = (21, 41, 81),
) -> dict:
    """L3r: BL §4.2 reflecting-hyperbolic eigenvalue stability check.

    For each grid size N, builds the 2×2 block operator for the coupled
    hyperbolic system u_t = v_x, v_t = u_x with reflecting BCs, and
    computes the spectral abscissa.  The continuous spectrum is purely
    imaginary; any positive real part indicates boundary-closure instability.

    Parameters
    ----------
    scheme : str
        Scheme name ("E2" or "E4").
    kernel : str
        Kernel type ("classical", "tension", "gaussian", "multiquadric").
    params : dict
        Kernel-specific parameters.
    n_values : tuple[int, ...]
        Grid sizes at which to evaluate stability.

    Returns
    -------
    dict with keys:
        spectral_abscissa_by_n : dict[int, float]
            {N: max Re(lambda)} for each grid size.
        max_spectral_abscissa : float
            Maximum over all grid sizes.
        purely_imaginary : bool
            True iff max_spectral_abscissa < BL42_TOL.
    """
    sp = _SCHEME_PARAMS[scheme]
    p, q, nextra, nu = sp["p"], sp["q"], sp["nextra"], sp["nu"]

    spectral_abscissa_by_n = {}
    for N in n_values:
        if kernel == "classical":
            D = _build_classical_diff_matrix(N, p, nu, params["alpha"])
        else:
            epsilon = params.get("sigma", params.get("epsilon", 0.0))
            D = build_diff_matrix_rbf(N, p, q, epsilon, kernel, nu, nextra)
        L_red = build_bl42_operator(D)
        max_re, _ = spectral_abscissa_sparse(L_red, k=10)
        spectral_abscissa_by_n[N] = max_re

    max_sa = max(spectral_abscissa_by_n.values())
    return {
        "spectral_abscissa_by_n": spectral_abscissa_by_n,
        "max_spectral_abscissa": max_sa,
        "purely_imaginary": max_sa < BL42_TOL,
    }


def layer4_local_gv_2d(
    scheme: str,
    kernel: str,
    params: dict,
    N: int = 31,
) -> dict:
    """L4: per-point local group velocity error on the Brady-Livescu 2D field.

    Freezes coefficients at each grid point and evaluates the interior stencil's
    group velocity error scaled by the local wave speed.  This is the first-order
    WKB approximation to the varying-coefficient dispersion error.

    Parameters
    ----------
    scheme : str
        Scheme name ("E2" or "E4").
    kernel : str
        Kernel type (unused for interior stencil — interior weights are
        scheme-determined — but kept for API consistency with other layers).
    params : dict
        Kernel-specific parameters (unused at this layer).
    N : int
        Grid resolution for the coefficient field.

    Returns
    -------
    dict with keys:
        max_local_gv_error : float
            Maximum absolute local GV error over all grid points and wavenumbers.
        worst_point : tuple[int, int]
            (i, j) indices of the grid point with the largest error.
        worst_xi : float
            Wavenumber at which the largest error occurs.
    """
    from stencil_gen.benchmarks.brady_livescu_2d import make_coefficient_field
    from stencil_gen.interior import derive_interior, full_gamma_array

    sp = _SCHEME_PARAMS[scheme]
    p, nu = sp["p"], sp["nu"]

    # Build interior stencil (same for x and y on Cartesian grid)
    coeffs = derive_interior(0, p, nu)
    w = np.array([float(c) for c in full_gamma_array(coeffs)])
    offsets = np.arange(-p, p + 1, dtype=float)
    stencil = (w, offsets)

    # Compute the interior GV profile to determine the resolved band cutoff
    profile = interior_group_velocity(p, nu, np.linspace(0.01, np.pi, 200))
    xi_max = profile.cutoff_xi * _RESOLVED_FRAC
    xi_array = np.linspace(0.01, xi_max, 200)

    _, _, c_x, c_y = make_coefficient_field(N)

    result = local_group_velocity_2d_varying(stencil, stencil, c_x, c_y, xi_array)

    # Compute max absolute error over all points and the resolved band
    err_x = np.abs(result["gv_error_x_field"])
    err_y = np.abs(result["gv_error_y_field"])
    combined = np.maximum(err_x, err_y)
    max_err = float(np.max(combined))

    # Find the worst point and wavenumber
    flat_idx = int(np.argmax(combined))
    i, j, k = np.unravel_index(flat_idx, combined.shape)

    return {
        "max_local_gv_error": max_err,
        "worst_point": (int(i), int(j)),
        "worst_xi": float(xi_array[k]),
    }


def layer5_anisotropy(
    scheme: str,
    kernel: str,
    params: dict,
    N: int = 31,
) -> dict:
    """L5: 2D anisotropy error projected onto the Brady-Livescu coefficient field.

    Evaluates the scheme's angular group velocity error at a representative
    wavenumber, then projects onto the local propagation direction at each
    grid point of the varying-coefficient field.  This detects grid anisotropy
    interacting with the radial flow pattern of the BL benchmark.

    Parameters
    ----------
    scheme : str
        Scheme name ("E2" or "E4").
    kernel : str
        Kernel type (unused — interior anisotropy is scheme-determined — but
        kept for API consistency with other layers).
    params : dict
        Kernel-specific parameters (unused at this layer).
    N : int
        Grid resolution for the coefficient field.

    Returns
    -------
    dict with keys:
        max_aligned_error : float
            Maximum |C_numerical - C_exact| projected onto local propagation.
        worst_point : tuple[int, int]
            (i, j) index of the grid point with the largest error.
        worst_theta : float
            Local propagation angle at the worst point.
    """
    from stencil_gen.benchmarks.brady_livescu_2d import make_coefficient_field

    sp = _SCHEME_PARAMS[scheme]
    p, nu = sp["p"], sp["nu"]

    _, _, c_x, c_y = make_coefficient_field(N)

    # Cover the range of local propagation angles in the BL field.
    # The BL field has angles in (0, pi/4) approximately (first quadrant,
    # below the diagonal), so [0.01, pi/2 - 0.01] comfortably covers it.
    theta_array = np.linspace(0.01, np.pi / 2 - 0.01, 200)

    # Use a representative wavenumber at 20% of the cutoff — inside the
    # well-resolved band but large enough for anisotropy to be visible.
    profile = interior_group_velocity(p, nu, np.linspace(0.01, np.pi, 200))
    xi_mag = profile.cutoff_xi * 0.2

    return anisotropy_over_coefficient_field(scheme, c_x, c_y, theta_array, xi_mag)


def layer6_non_normality(
    scheme: str,
    kernel: str,
    params: dict,
    n: int = 80,
) -> dict:
    """L6: non-normality diagnostics on the 1D differentiation matrix.

    Runs the full non-normality analysis (spectral/numerical abscissa, Henrici
    departure, Kreiss constant, transient growth bound) on the 1D first-derivative
    operator with inflow BC removed.  This is cheaper than the 2D L7 check
    (~seconds vs ~tens of seconds) and catches operators with dangerous transient
    growth even when eigenvalues are stable.

    Parameters
    ----------
    scheme : str
        Scheme name ("E2" or "E4").
    kernel : str
        Kernel type ("classical", "tension", "gaussian", "multiquadric").
    params : dict
        Kernel-specific parameters.
    n : int
        Grid size for the 1D differentiation matrix.

    Returns
    -------
    dict with keys:
        spectral_abscissa : float
        numerical_abscissa : float
        henrici_departure : float
        kreiss_constant : float
        transient_growth_bound : float
        compute_time : float
        non_normality_report : NonNormalityReport
    """
    sp = _SCHEME_PARAMS[scheme]
    p, q, nextra, nu = sp["p"], sp["q"], sp["nextra"], sp["nu"]

    if kernel == "classical":
        D = _build_classical_diff_matrix(n, p, nu, params["alpha"])
    else:
        epsilon = params.get("sigma", params.get("epsilon", 0.0))
        D = build_diff_matrix_rbf(n, p, q, epsilon, kernel, nu, nextra)

    # Remove inflow row/column (row 0) to form the semi-discrete operator
    # for the homogeneous stability problem: -D_bc
    L_1d = -D[1:, 1:]

    report = compute_non_normality(L_1d)
    logger.debug(
        "L6 %s/%s n=%d: spectral_abscissa=%.6e, kreiss_K=%.2f, "
        "tgb=%.2f, compute_time=%.1fs",
        scheme, kernel, n, report.spectral_abscissa,
        report.kreiss_constant, report.transient_growth_bound,
        report.compute_time,
    )

    return {
        "spectral_abscissa": report.spectral_abscissa,
        "numerical_abscissa": report.numerical_abscissa,
        "henrici_departure": report.henrici_departure,
        "kreiss_constant": report.kreiss_constant,
        "transient_growth_bound": report.transient_growth_bound,
        "compute_time": report.compute_time,
        "non_normality_report": report,
    }


def build_sparse_2d_operator(
    scheme: str,
    kernel: str,
    params: dict,
    N: int,
) -> tuple:
    """Build the sparse 2D semi-discrete operator for the Brady-Livescu benchmark.

    Constructs the full N²×N² advection operator L = -(Cx @ Dx_2D + Cy @ Dy_2D)
    using Kronecker products of 1D differentiation matrices, then removes inflow
    rows/columns (i=0 and j=0) to produce the reduced operator for stability
    analysis.

    Parameters
    ----------
    scheme : str
        Scheme name ("E2" or "E4").
    kernel : str
        Kernel type ("classical", "tension", "gaussian", "multiquadric").
    params : dict
        Kernel-specific parameters.
    N : int
        Number of grid points per direction.

    Returns
    -------
    (L_red, keep_idx) : tuple[scipy.sparse.csr_matrix, np.ndarray]
        L_red : the reduced operator of shape ((N-1)², (N-1)²).
        keep_idx : integer array of retained DOF indices into the full N² system.
    """
    import scipy.sparse as sp

    from stencil_gen.benchmarks.brady_livescu_2d import L_DOMAIN, make_coefficient_field

    spar = _SCHEME_PARAMS[scheme]
    p, q, nextra, nu = spar["p"], spar["q"], spar["nextra"], spar["nu"]

    # Build 1D differentiation matrix
    if kernel == "classical":
        D1 = _build_classical_diff_matrix(N, p, nu, params["alpha"])
    else:
        epsilon = params.get("sigma", params.get("epsilon", 0.0))
        D1 = build_diff_matrix_rbf(N, p, q, epsilon, kernel, nu, nextra)

    # build_diff_matrix_rbf returns weights for unit grid spacing (h=1,
    # integer grid {0, 1, ..., N-1}).  The physical domain [0, L_DOMAIN] has
    # spacing h = L_DOMAIN / (N - 1), so scale D by 1/h to get the physical
    # derivative operator.
    h = L_DOMAIN / (N - 1)
    D1_sp = sp.csr_matrix(D1) / h
    I_N = sp.eye(N, format="csr")

    # Kronecker products for column-major (Fortran-order) flattening:
    # u_flat[j*N + i] = u[i, j], where i=x-index, j=y-index
    Dx_2D = sp.kron(I_N, D1_sp, format="csr")  # x-derivative
    Dy_2D = sp.kron(D1_sp, I_N, format="csr")  # y-derivative

    # Coefficient fields from Brady-Livescu benchmark
    _, _, cx_field, cy_field = make_coefficient_field(N)

    # Flatten in Fortran order to match Kronecker product convention
    cx_vec = cx_field.flatten("F")
    cy_vec = cy_field.flatten("F")

    Cx = sp.diags(cx_vec, format="csr")
    Cy = sp.diags(cy_vec, format="csr")

    # Full 2D advection operator: L = -(Cx @ Dx + Cy @ Dy)
    L_2D = -(Cx @ Dx_2D + Cy @ Dy_2D)

    # Remove inflow DOFs: i=0 (x=0) and j=0 (y=0)
    N2 = N * N
    flat_indices = np.arange(N2)
    ii = flat_indices % N   # x-index
    jj = flat_indices // N  # y-index
    keep_mask = (ii > 0) & (jj > 0)
    keep_idx = flat_indices[keep_mask]

    L_red = L_2D[np.ix_(keep_idx, keep_idx)].tocsr()

    return L_red, keep_idx


# Layer-7 threshold: max Re(eigenvalue of 2D varying-coefficient operator).
#
# The Brady-Livescu radial flow field has div(c) = 1/psi > 0 (diverging
# flow), so the continuous operator is not skew-symmetric.  The homogeneous
# stability problem (inflow DOFs removed) inherently has positive spectral
# abscissa because the divergence-driven energy growth is not balanced by
# inflow energy supply.  The full problem is stable because the outflow
# boundary closures provide dissipation and RK4 at CFL=0.8 accommodates
# the eigenvalue locations.
#
# Calibration (with correct h = L_DOMAIN/(N-1) scaling):
#   - Known-stable tension E4 sigma=3.0: max Re ~ 0.018
#   - Known-unstable Gaussian E4 eps=0.1: max Re ~ 3.1
# A threshold of 0.1 cleanly separates these regimes.
L7_TOL = 0.1


def layer7_sparse_2d_eigenvalue(
    scheme: str,
    kernel: str,
    params: dict,
    n_values: tuple[int, ...] = (21, 31, 61),
) -> dict:
    """L7: sparse 2D Arnoldi eigenvalue check on the full varying-coefficient operator.

    For each grid size N, builds the full 2D Brady-Livescu advection operator
    with inflow DOFs removed, then computes the spectral abscissa (max Re(lambda))
    via sparse Arnoldi iteration.  This is the definitive semi-discrete stability
    test for the 2D varying-coefficient problem.

    Parameters
    ----------
    scheme : str
        Scheme name ("E2" or "E4").
    kernel : str
        Kernel type ("classical", "tension", "gaussian", "multiquadric").
    params : dict
        Kernel-specific parameters.
    n_values : tuple[int, ...]
        Grid sizes per direction at which to evaluate stability.

    Returns
    -------
    dict with keys:
        eigenvalues : dict[int, float]
            {N: max_real_eigenvalue} for each grid size.
        max_spectral_abscissa : float
            Maximum over all grid sizes.
    """
    eigenvalues = {}
    for n in n_values:
        L_red, _ = build_sparse_2d_operator(scheme, kernel, params, n)
        max_re, _ = spectral_abscissa_sparse(L_red, k=20)
        eigenvalues[n] = max_re
        logger.debug(
            "L7 %s/%s N=%d: max Re(lambda) = %.6e",
            scheme, kernel, n, max_re,
        )

    return {
        "eigenvalues": eigenvalues,
        "max_spectral_abscissa": max(eigenvalues.values()),
    }


# Layer-6 threshold: transient growth bound on the 1D operator.
# This is the standalone non-normality check on the 1D diff matrix, cheaper
# than the full 2D L7 check.  The threshold is the same as L7 since the
# 1D operator should be at least as well-behaved as the 2D one.
L6_TRANSIENT_GROWTH_TOL = 50.0

# Transient growth bound threshold for L7+non-normality combined check.
L7_TRANSIENT_GROWTH_TOL = 50.0


def layer7_with_non_normality(
    scheme: str,
    kernel: str,
    params: dict,
    N: int = 31,
) -> NonNormalityReport:
    """L7 + L6: non-normality diagnostics on the full 2D BL operator.

    Builds the reduced 2D Brady-Livescu operator at a single grid size and
    computes the full non-normality report (spectral abscissa, numerical
    abscissa, Henrici departure, pseudospectral abscissa, Kreiss constant,
    transient growth bound).

    This links L6 infrastructure to the actual BL operator — L6 defines the
    metric functions, this wires them to the BL coefficient field.

    Failure criteria:
    - spectral_abscissa > L7_TOL (5e-3), OR
    - transient_growth_bound > L7_TRANSIENT_GROWTH_TOL (50.0).

    Parameters
    ----------
    scheme : str
        Scheme name ("E2" or "E4").
    kernel : str
        Kernel type ("classical", "tension", "gaussian", "multiquadric").
    params : dict
        Kernel-specific parameters.
    N : int
        Grid points per direction.  Default 31 gives a (30×30 = 900)-DOF
        reduced operator, small enough for dense SVD in the resolvent
        computation.

    Returns
    -------
    NonNormalityReport
        Full non-normality diagnostics for the 2D BL operator.
    """
    L_red, _ = build_sparse_2d_operator(scheme, kernel, params, N)
    report = compute_non_normality(L_red)
    logger.debug(
        "L7+non-normality %s/%s N=%d: spectral_abscissa=%.6e, "
        "transient_growth_bound=%.2f, kreiss_constant=%.2f, compute_time=%.1fs",
        scheme, kernel, N, report.spectral_abscissa,
        report.transient_growth_bound, report.kreiss_constant,
        report.compute_time,
    )
    return report


# Layer-8 threshold: final L∞ error at t=10 must be bounded. Classical E4u on
# the uniform Brady-Livescu domain produces L∞ ~ 2e-3 at N=31, t=10; so 1.0 is
# a loose ceiling that only catches blow-up, not high-order accuracy loss.
L8_FINAL_LINF_TOL = 1.0


# Dispatch table: (scheme, kernel) → Lua scheme.type string understood by
# stencil::from_lua. Plan 42.7a wires the three spline families alongside
# the classical branch; E2 variants remain deferred (plan 42.10a).
_L8_SCHEME_TYPE = {
    ("E4", "classical"): "E4u",
    ("E4", "tension"): "tension_E4u",
    ("E4", "gaussian"): "gaussian_E4u",
    ("E4", "multiquadric"): "multiquadric_E4u",
}


def layer8_cpp_simulation(
    scheme: str,
    kernel: str,
    params: dict,
    *,
    N: int = 31,
    t_final: float = 10.0,
) -> dict:
    """L8: end-to-end C++ simulation via the shoccs Brady-Livescu 2D harness.

    Renders a Lua config from the plan-42 template with the supplied scheme
    parameters, invokes the compiled shoccs binary, and reports the final
    L∞ error. A layer-8 pass means the analytical predictions of L1–L7 survive
    contact with the real solver; a failure flags an inconsistency to chase.

    Parameters
    ----------
    scheme : str
        Scheme name ("E2" or "E4"). Plan 42 first cut supports "E4" only.
    kernel : str
        Kernel type ("classical", "tension", "gaussian", "multiquadric").
        Plan 42.7a wires all four for the E4 scheme; E2 variants remain
        deferred (plan 42.10a).
    params : dict
        Scheme-specific parameters; passed through verbatim to
        :func:`run_cpp_brady2d`. Classical uses ``{"alpha": [...]}``,
        tension uses ``{"sigma": ...}``, gaussian and multiquadric use
        ``{"epsilon": ...}``.
    N : int
        Grid resolution (points per side). Default 31 matches the
        Brady-Livescu §4.3 coarsest grid.
    t_final : float
        Physical end time. Default 10.0 — short enough to run in a
        few seconds, long enough to catch an exponential blow-up.

    Returns
    -------
    dict with keys:
        final_linf : float
            Final L∞ error at t_final (nan on simulation failure).
        stable : bool
            True iff the simulation ran to completion with a finite L∞
            below the hard blow-up ceiling used by
            :func:`run_cpp_brady2d`.
        wall_time_s : float
            Wall-clock seconds for the C++ invocation.
        bridge_result : BridgeResult
            Full bridge result for debugging (exit_code, stderr, traces).

    Raises
    ------
    NotImplementedError
        When ``(scheme, kernel)`` has no registered Lua scheme type. Items
        42.7a onward extend the dispatch table.
    """
    key = (scheme, kernel)
    if key not in _L8_SCHEME_TYPE:
        raise NotImplementedError(
            f"layer8_cpp_simulation has no dispatch for (scheme={scheme!r}, "
            f"kernel={kernel!r}); only {sorted(_L8_SCHEME_TYPE)} supported so far",
        )
    scheme_type = _L8_SCHEME_TYPE[key]

    result: BridgeResult = run_cpp_brady2d(
        scheme_type, params, N=N, t_final=t_final,
    )

    logger.debug(
        "L8 %s/%s N=%d t_final=%g: final_linf=%.4e stable=%s wall=%.1fs",
        scheme, kernel, N, t_final, result.final_linf, result.stable,
        result.wall_time_s,
    )

    return {
        "final_linf": result.final_linf,
        "stable": result.stable,
        "wall_time_s": result.wall_time_s,
        "bridge_result": result,
    }


def brady2d_stability_score(
    scheme: str,
    kernel: str,
    params: dict,
    *,
    max_layer: int = 7,
    short_circuit: bool = True,
    layer8_N: int = 31,
    layer8_t_final: float = 10.0,
) -> StabilityReport:
    """Run the layered stability analysis pipeline.

    Executes layers 1 through ``max_layer`` in order, populating the
    corresponding fields of a :class:`StabilityReport`.  If
    ``short_circuit`` is True (default), execution stops at the first
    failing layer.

    Parameters
    ----------
    scheme : str
        Scheme name ("E2" or "E4").
    kernel : str
        Kernel type ("classical", "tension", "gaussian", "multiquadric").
    params : dict
        Kernel-specific parameters.
    max_layer : int
        Highest layer to run (1–8). Layer 8 invokes the compiled shoccs
        binary via :func:`layer8_cpp_simulation` and is gated behind
        earlier layers passing when ``short_circuit`` is True.
    short_circuit : bool
        If True, stop at the first failing layer.
    layer8_N : int
        Grid resolution forwarded to :func:`layer8_cpp_simulation` when
        ``max_layer >= 8``.
    layer8_t_final : float
        Physical end time forwarded to :func:`layer8_cpp_simulation`
        when ``max_layer >= 8``.

    Returns
    -------
    StabilityReport
        Populated report with per-layer results and overall verdict.
    """
    t0 = time.perf_counter()
    report = StabilityReport.empty()

    def _record_failure(layer_num: int, reason: str) -> None:
        """Record a failure if this is the first one."""
        if report.failed_layer is None:
            report.failed_layer = layer_num
            report.failed_reason = reason

    def _should_stop() -> bool:
        return short_circuit and report.failed_layer is not None

    # --- Layer 1: interior + boundary group velocity error ---
    if max_layer >= 1:
        report.layer1 = layer1_interior_boundary_gv(scheme, kernel, params)
        err = report.layer1["boundary_gv_err"]
        if err > L1_TOL:
            _record_failure(1, f"boundary_gv_err={err:.4e} > L1_TOL={L1_TOL}")
        if _should_stop():
            report.overall_verdict = "fail"
            report.compute_time = time.perf_counter() - t0
            return report

    # --- Layer 2: rigorous GKS Kreiss determinant test ---
    if max_layer >= 2:
        kr = layer2_kreiss_gks(scheme, kernel, params)
        report.layer2 = kr
        report.kreiss = kr
        if not kr.is_stable:
            _record_failure(
                2, f"Kreiss GKS unstable: sigma_min={kr.witness_sigma_min:.4e}"
            )
        if _should_stop():
            report.overall_verdict = "fail"
            report.compute_time = time.perf_counter() - t0
            return report

    # --- Layer 3: 1D eigenvalue check at multiple grid sizes ---
    if max_layer >= 3:
        report.layer3 = layer3_1d_eigenvalue(scheme, kernel, params)
        mse = report.layer3["max_stab_eig"]
        if mse > STABILITY_TOL:
            _record_failure(3, f"max_stab_eig={mse:.4e} > STABILITY_TOL={STABILITY_TOL}")
        if _should_stop():
            report.overall_verdict = "fail"
            report.compute_time = time.perf_counter() - t0
            return report

    # --- Layer 3r: BL §4.2 reflecting-hyperbolic eigenvalue check ---
    if max_layer >= 3:
        report.layer_bl42 = layer_bl42_reflecting_hyperbolic(
            scheme, kernel, params,
        )
        max_re = report.layer_bl42["max_spectral_abscissa"]
        if max_re > BL42_TOL:
            _record_failure(
                3,
                f"BL42 max_spectral_abscissa={max_re:.4e} > BL42_TOL={BL42_TOL}",
            )
        if _should_stop():
            report.overall_verdict = "fail"
            report.compute_time = time.perf_counter() - t0
            return report

    # --- Layer 4: per-point local GV error on 2D varying-coefficient field ---
    if max_layer >= 4:
        report.layer4 = layer4_local_gv_2d(scheme, kernel, params)
        gv_err = report.layer4["max_local_gv_error"]
        if gv_err > L4_TOL:
            _record_failure(4, f"max_local_gv_error={gv_err:.4e} > L4_TOL={L4_TOL}")
        if _should_stop():
            report.overall_verdict = "fail"
            report.compute_time = time.perf_counter() - t0
            return report

    # --- Layer 5: 2D anisotropy over the coefficient field ---
    if max_layer >= 5:
        report.layer5 = layer5_anisotropy(scheme, kernel, params)
        aniso_err = report.layer5["max_aligned_error"]
        if aniso_err > L5_TOL:
            _record_failure(5, f"max_aligned_error={aniso_err:.4e} > L5_TOL={L5_TOL}")
        if _should_stop():
            report.overall_verdict = "fail"
            report.compute_time = time.perf_counter() - t0
            return report

    # --- Layer 6: non-normality diagnostics on 1D operator ---
    if max_layer >= 6:
        l6 = layer6_non_normality(scheme, kernel, params)
        report.layer6 = l6
        report.non_normality = l6["non_normality_report"]
        sa = l6["spectral_abscissa"]
        tgb = l6["transient_growth_bound"]
        if sa > STABILITY_TOL:
            _record_failure(
                6, f"1D spectral_abscissa={sa:.4e} > STABILITY_TOL={STABILITY_TOL}"
            )
        elif tgb > L6_TRANSIENT_GROWTH_TOL:
            _record_failure(
                6,
                f"1D transient_growth_bound={tgb:.2f} "
                f"> L6_TRANSIENT_GROWTH_TOL={L6_TRANSIENT_GROWTH_TOL}",
            )
        if _should_stop():
            report.overall_verdict = "fail"
            report.compute_time = time.perf_counter() - t0
            return report

    # --- Layer 7: sparse 2D Arnoldi eigenvalue ---
    if max_layer >= 7:
        report.layer7 = layer7_sparse_2d_eigenvalue(scheme, kernel, params)
        msa = report.layer7["max_spectral_abscissa"]
        if msa > L7_TOL:
            _record_failure(
                7, f"max_spectral_abscissa={msa:.4e} > L7_TOL={L7_TOL}"
            )
        if _should_stop():
            report.overall_verdict = "fail"
            report.compute_time = time.perf_counter() - t0
            return report

        # Non-normality diagnostics on the 2D BL operator (combined L6+L7)
        report.non_normality = layer7_with_non_normality(scheme, kernel, params)
        if report.non_normality.transient_growth_bound > L7_TRANSIENT_GROWTH_TOL:
            _record_failure(
                7,
                f"transient_growth_bound="
                f"{report.non_normality.transient_growth_bound:.2f} "
                f"> L7_TRANSIENT_GROWTH_TOL={L7_TRANSIENT_GROWTH_TOL}",
            )
        if _should_stop():
            report.overall_verdict = "fail"
            report.compute_time = time.perf_counter() - t0
            return report

    # --- Layer 8: end-to-end C++ simulation ---
    if max_layer >= 8:
        report.layer8 = layer8_cpp_simulation(
            scheme, kernel, params,
            N=layer8_N, t_final=layer8_t_final,
        )
        stable = bool(report.layer8["stable"])
        linf = float(report.layer8["final_linf"])
        if not stable:
            _record_failure(
                8,
                f"C++ simulation unstable: final_linf={linf:.4e} "
                f"(exit_code={report.layer8['bridge_result'].exit_code})",
            )
        elif linf > L8_FINAL_LINF_TOL:
            _record_failure(
                8,
                f"final_linf={linf:.4e} > L8_FINAL_LINF_TOL={L8_FINAL_LINF_TOL}",
            )

    # Final verdict
    report.overall_verdict = "fail" if report.failed_layer is not None else "pass"
    report.compute_time = time.perf_counter() - t0
    return report
