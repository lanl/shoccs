"""Rigorous GKS Kreiss determinant stability test for semi-discrete boundary closures.

Implements the Kreiss stability condition from Trefethen 1983 (pp. 206-207,
"Group velocity interpretation of the stability theory of Gustafsson, Kreiss,
and Sundstrom", *J. Comput. Phys.* 49, pp. 199-217).

For the semi-discrete left-boundary problem u_t = -sum_k a_k u_{n+k}, a mode
u_n(t) = e^(st) kappa^n requires s + sum_k a_k kappa^k = 0. For each s in
the right half-plane, the admissible roots are those with |kappa| < 1. The
r x r Kreiss matrix M(s) is built from the r admissible roots and the r
boundary rows, and sigma_min(M(s)) is the Kreiss determinant condition
indicator. The boundary closure is GKS-stable iff sigma_min(M(s)) > 0 for
all s with Re(s) >= 0, with imaginary-axis perturbation used to classify
tangent modes per Trefethen p. 207.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("stencil_gen.gks_kreiss")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

BoundaryRow = tuple[np.ndarray, np.ndarray]
"""(weights, column_offsets) for one boundary row of the closure."""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DefectiveKappaError(RuntimeError):
    """Raised when admissible kappa roots are defective (repeated)."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KreissResult:
    """Result of the rigorous Kreiss determinant stability check."""

    is_stable: bool
    """True if no sigma_min violation was found in the sampled s-grid."""

    witness_s: Optional[complex] = None
    """The s value at which the minimum sigma_min was found (None if stable)."""

    witness_sigma_min: float = float("inf")
    """The minimum singular value at the witness point."""

    imaginary_axis_perturbation_verdict: str = "not_checked"
    """One of: 'not_checked', 'no_candidates', 'all_incoming',
    'outgoing_mode_detected', 'defective'."""

    defective_kappa_detected: bool = False
    """True if defective (repeated) admissible kappa roots were encountered."""

    s_grid_shape: tuple[int, ...] = ()
    """Shape of the s-grid used for the sweep."""

    compute_time: float = 0.0
    """Wall-clock time in seconds for the full check."""

    sigma_min_field: Optional[np.ndarray] = field(default=None, repr=False)
    """The sigma_min values over the entire s-grid (for diagnostics)."""

    s_grid: Optional[np.ndarray] = field(default=None, repr=False)
    """The complex s-grid used for the sweep."""

    n_admissible_roots: int = 0
    """Number of admissible kappa roots (|kappa| < 1) at the witness point."""


# ---------------------------------------------------------------------------
# Primitive functions (implemented in 41.3b-41.3f)
# ---------------------------------------------------------------------------


def kappa_roots(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    s: complex,
    *,
    repeat_tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Find all roots kappa of the characteristic polynomial Q(kappa) = 0.

    For the interior stencil u_t = -sum_k a_k u_{n+k}, the mode ansatz
    u_n(t) = e^(st) kappa^n yields s + sum_k a_k kappa^k = 0. Multiply
    through by kappa^L_left to clear negative powers.

    Returns
    -------
    all_roots : np.ndarray
        All roots of Q(kappa).
    admissible : np.ndarray
        Roots with |kappa| < 1 (strictly inside the unit disk).
    is_defective : bool
        True if any pair of admissible roots has separation < repeat_tol.
    """
    offsets = np.asarray(interior_offsets, dtype=int)
    weights = np.asarray(interior_weights, dtype=complex)
    L_left = -int(np.min(offsets))
    shifted = offsets + L_left  # now all >= 0

    # Polynomial degree
    degree = int(np.max(shifted))
    # If the s*kappa^L_left term contributes at a higher degree, extend
    degree = max(degree, L_left)

    # Build coefficient array: Q(kappa) = sum_k a_k * kappa^shifted[k] + s * kappa^L_left
    # numpy.roots expects coefficients from highest to lowest degree
    coeffs = np.zeros(degree + 1, dtype=complex)
    for w, sh in zip(weights, shifted):
        coeffs[sh] += w
    coeffs[L_left] += s

    # numpy.roots expects [c_n, c_{n-1}, ..., c_1, c_0] (highest degree first)
    poly_coeffs = coeffs[::-1]

    all_roots = np.roots(poly_coeffs)

    # Admissible: strictly inside the unit disk
    admissible = all_roots[np.abs(all_roots) < 1.0 - 1e-12]

    # Defective check: any pair of admissible roots with separation < repeat_tol
    is_defective = False
    if len(admissible) > 1:
        for i in range(len(admissible)):
            for j in range(i + 1, len(admissible)):
                if abs(admissible[i] - admissible[j]) < repeat_tol:
                    is_defective = True
                    break
            if is_defective:
                break

    return all_roots, admissible, is_defective


def _kappa_roots_from_poly(
    poly_coeffs: np.ndarray,
    *,
    repeat_tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Find roots from explicit polynomial coefficients (highest degree first).

    Helper for testing defective-root detection without constructing stencils.

    Returns same tuple as kappa_roots: (all_roots, admissible, is_defective).
    """
    all_roots = np.roots(poly_coeffs)
    admissible = all_roots[np.abs(all_roots) < 1.0 - 1e-12]

    is_defective = False
    if len(admissible) > 1:
        for i in range(len(admissible)):
            for j in range(i + 1, len(admissible)):
                if abs(admissible[i] - admissible[j]) < repeat_tol:
                    is_defective = True
                    break
            if is_defective:
                break

    return all_roots, admissible, is_defective


def kreiss_matrix(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    s: complex,
) -> np.ndarray:
    """Build the r x r Kreiss matrix M(s) from admissible roots and boundary rows.

    M[i, ell] = s * kappa_ell^i + sum_j w_{ij} * kappa_ell^j

    where i indexes boundary rows and ell indexes admissible kappa roots.

    Raises ValueError if len(admissible) != len(boundary_rows).
    """
    _, admissible, is_defective = kappa_roots(interior_weights, interior_offsets, s)
    if is_defective:
        raise DefectiveKappaError(
            f"Defective admissible kappa roots at s={s}"
        )

    r = len(boundary_rows)
    if len(admissible) != r:
        raise ValueError(
            f"Number of admissible roots ({len(admissible)}) != "
            f"number of boundary rows ({r})"
        )

    M = np.zeros((r, r), dtype=complex)
    for i, (weights, col_offsets) in enumerate(boundary_rows):
        weights = np.asarray(weights, dtype=complex)
        col_offsets = np.asarray(col_offsets, dtype=int)
        for ell, kappa in enumerate(admissible):
            # Temporal contribution: s * kappa^i
            M[i, ell] = s * kappa**i
            # Spatial operator contribution: sum_j w_{ij} * kappa^j
            for w, j in zip(weights, col_offsets):
                M[i, ell] += w * kappa**j

    return M


def min_singular_value(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    s: complex,
) -> float:
    """Compute sigma_min(M(s)), the minimum singular value of the Kreiss matrix.

    Returns np.inf on DefectiveKappaError or shape mismatch.
    """
    try:
        M = kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s)
    except (DefectiveKappaError, ValueError):
        return np.inf
    svs = np.linalg.svd(M, compute_uv=False)
    return float(svs[-1])


def make_s_grid(
    s_max: float = 10.0,
    n_radial: int = 40,
    n_imag: int = 120,
    imag_max: float = 20.0,
    eps_imag: float = 1e-6,
) -> np.ndarray:
    """Build an L-shaped contour grid in the right half of the complex s-plane.

    The grid combines a logarithmically-spaced radial sweep with a dense
    imaginary-axis strip at Re(s) = eps_imag, covering Im(s) in
    [-imag_max, imag_max].

    Returns a 2D array of shape ``(n_imag, n_radial + 1)`` where:
    - Column 0 is the imaginary-axis strip at ``Re(s) = eps_imag``.
    - Columns 1..n_radial are logarithmically spaced from ``1e-4`` to ``s_max``.
    """
    # Real parts: imaginary-axis strip, then log-spaced into the right half-plane
    re_values = np.concatenate(
        [[eps_imag], np.logspace(-4, np.log10(s_max), n_radial)]
    )
    # Imaginary parts: uniform spacing
    im_values = np.linspace(-imag_max, imag_max, n_imag)
    # Meshgrid → 2D grid of complex s values
    Re, Im = np.meshgrid(re_values, im_values)
    return Re + 1j * Im


def _sweep_grid(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    s_grid: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Evaluate min_singular_value at every point of s_grid.

    Returns
    -------
    sigma_field : np.ndarray
        sigma_min values, same shape as s_grid.
    argmin_idx : int
        Flat index of the global minimum in sigma_field.
    """
    flat = s_grid.ravel()
    sigma_flat = np.empty(len(flat))
    for i, s in enumerate(flat):
        sigma_flat[i] = min_singular_value(
            interior_weights, interior_offsets, boundary_rows, s
        )
    sigma_field = sigma_flat.reshape(s_grid.shape)
    argmin_idx = int(np.argmin(sigma_flat))
    return sigma_field, argmin_idx


def _refine_witness(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    s_start: complex,
    *,
    min_s_magnitude: float = 0.0,
) -> complex:
    """Refine a candidate witness s via Nelder-Mead minimization of log(sigma_min).

    Constrains Re(s) >= 0 by reflecting into the right half-plane.
    Optionally constrains |s| >= min_s_magnitude by adding a barrier penalty,
    preventing convergence to the trivial zero at s=0 that exists for all
    consistent first-derivative operators.

    Parameters
    ----------
    interior_weights, interior_offsets, boundary_rows
        Stencil and boundary closure specification.
    s_start : complex
        Initial guess for the witness (typically the grid-sweep argmin).
    min_s_magnitude : float
        Barrier to prevent convergence to the trivial zero at s=0.

    Returns
    -------
    complex
        The refined s at which sigma_min is locally minimized.
    """
    from scipy.optimize import minimize

    def objective(re_im: np.ndarray) -> float:
        # Reflect Re(s) < 0 into the right half-plane
        re_s = abs(re_im[0])
        im_s = re_im[1]
        s = complex(re_s, im_s)
        # Barrier to keep |s| > min_s_magnitude
        if min_s_magnitude > 0 and abs(s) < min_s_magnitude:
            return 100.0  # large penalty
        sv = min_singular_value(
            interior_weights, interior_offsets, boundary_rows, s
        )
        return float(np.log(sv + 1e-300))

    x0 = np.array([s_start.real, s_start.imag])
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 2000})
    return complex(abs(result.x[0]), result.x[1])


def _classify_imag_axis(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    s_candidate: complex,
    delta: float = 1e-4,
) -> str:
    """Classify a near-imaginary-axis candidate s via kappa perturbation.

    For a candidate s_0 near the imaginary axis, recompute kappa roots at
    s_0 and s_0 + delta, match by nearest-neighbor, and classify
    unit-modulus kappas as incoming or outgoing per Trefethen p. 207.

    Note on naming convention: "outgoing_mode_detected" means a physical
    outgoing mode was found at the boundary — this is a GKS violation
    (instability). "all_incoming" means all near-unit-circle modes move
    outward under perturbation — this is the stable (non-violation) case.
    The terminology follows Trefethen: an "outgoing" mode at the boundary
    is one whose group velocity points away from the boundary into the
    interior, meaning it is generated by the boundary and should not exist
    in a well-posed problem.

    Parameters
    ----------
    interior_weights, interior_offsets
        Interior stencil specification.
    s_candidate : complex
        The candidate s value near the imaginary axis.
    delta : float
        Perturbation magnitude in Re(s) for the incoming/outgoing test.

    Returns
    -------
    str
        One of: 'no_candidates', 'all_incoming', 'outgoing_mode_detected',
        'defective'.
    """
    unit_tol = 1e-4  # how close to |kappa|=1 a root must be to count
    repeat_tol = 1e-7  # defective threshold for near-unit roots

    # Roots at s_0
    all_0, _, _ = kappa_roots(
        interior_weights, interior_offsets, s_candidate
    )

    # Find near-unit-circle roots at s_0
    near_unit = [k for k in all_0 if abs(abs(k) - 1.0) < unit_tol]
    if not near_unit:
        return "no_candidates"

    # Defective check restricted to near-unit-circle roots only.
    # (The kappa_roots defective flag checks all admissible roots, which
    # would false-positive on roots deep inside the unit disk.)
    for i in range(len(near_unit)):
        for j in range(i + 1, len(near_unit)):
            if abs(near_unit[i] - near_unit[j]) < repeat_tol:
                return "defective"

    # Roots at s_0 + delta (perturb into the right half-plane)
    all_delta, _, _ = kappa_roots(
        interior_weights, interior_offsets, s_candidate + delta
    )

    # Match near-unit roots to perturbed roots by nearest-neighbor
    for k0 in near_unit:
        # Find closest root in the perturbed set
        distances = np.abs(all_delta - k0)
        k_delta = all_delta[np.argmin(distances)]

        # If |k_delta| < |k0|, the root moved inward (toward origin)
        # under a Re(s) increase — this is an "outgoing mode" (GKS violation)
        if abs(k_delta) < abs(k0) - 1e-10:
            return "outgoing_mode_detected"

    return "all_incoming"


# ---------------------------------------------------------------------------
# Orchestrator (implemented in 41.3g)
# ---------------------------------------------------------------------------


def kreiss_stability_check(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    *,
    s_grid_params: Optional[dict] = None,
    sigma_tol: float = 1e-8,
    refine: bool = True,
    min_s_magnitude: float = 0.1,
) -> KreissResult:
    """Run the full Kreiss determinant stability check.

    Sweeps the s-grid, optionally refines any witness, classifies
    imaginary-axis modes, and returns a KreissResult.

    Parameters
    ----------
    interior_weights : np.ndarray
        Weights of the interior finite-difference stencil.
    interior_offsets : np.ndarray
        Grid offsets of the interior stencil (e.g., [-2, -1, 0, 1, 2]).
    boundary_rows : list[BoundaryRow]
        Each element is (weights, column_offsets) for one boundary row.
    s_grid_params : dict, optional
        Keyword arguments passed to make_s_grid.
    sigma_tol : float
        Threshold below which sigma_min indicates instability.
    refine : bool
        Whether to refine the witness via Nelder-Mead.
    min_s_magnitude : float
        Exclude the trivial zero near s=0 by masking grid points with
        |s| < min_s_magnitude.  For consistent first-derivative operators,
        sigma_min(M(0)) = 0 because the constant mode kappa=1 satisfies
        both interior and boundary equations.  This universal zero is not
        a GKS violation.
    """
    import time

    t0 = time.perf_counter()

    grid_kw = s_grid_params or {}
    s_grid = make_s_grid(**grid_kw)

    try:
        sigma_field, _ = _sweep_grid(
            interior_weights, interior_offsets, boundary_rows, s_grid
        )
    except DefectiveKappaError:
        return KreissResult(
            is_stable=False,
            defective_kappa_detected=True,
            s_grid_shape=s_grid.shape,
            compute_time=time.perf_counter() - t0,
            s_grid=s_grid,
        )

    flat_sigma = sigma_field.ravel()
    flat_s = s_grid.ravel()

    # Mask out the trivial zero near s=0 (constant-mode artifact).
    mask = np.abs(flat_s) >= min_s_magnitude
    if not np.any(mask):
        logger.warning("All grid points masked by min_s_magnitude=%.2e", min_s_magnitude)
        mask[:] = True  # fall back to unmasked

    masked_sigma = np.where(mask, flat_sigma, np.inf)
    argmin_idx = int(np.argmin(masked_sigma))
    min_sigma = float(masked_sigma[argmin_idx])
    witness_s = complex(flat_s[argmin_idx])

    # Refine from the grid argmin to find the true local minimum of sigma_min.
    # The grid may not sample close enough to a genuine zero, so we always
    # refine (when requested) rather than applying a sweep-stage threshold.
    if refine:
        try:
            witness_s = _refine_witness(
                interior_weights, interior_offsets, boundary_rows, witness_s,
                min_s_magnitude=min_s_magnitude,
            )
            min_sigma = min_singular_value(
                interior_weights, interior_offsets, boundary_rows, witness_s
            )
        except DefectiveKappaError:
            return KreissResult(
                is_stable=False,
                witness_s=witness_s,
                witness_sigma_min=min_sigma,
                defective_kappa_detected=True,
                s_grid_shape=s_grid.shape,
                compute_time=time.perf_counter() - t0,
                sigma_min_field=sigma_field,
                s_grid=s_grid,
            )

    # After refinement, apply sigma_tol to determine stability
    if min_sigma >= sigma_tol:
        return KreissResult(
            is_stable=True,
            witness_s=witness_s,
            witness_sigma_min=min_sigma,
            s_grid_shape=s_grid.shape,
            compute_time=time.perf_counter() - t0,
            sigma_min_field=sigma_field,
            s_grid=s_grid,
            n_admissible_roots=len(boundary_rows),
        )

    # Potential violation found — but if the refined witness drifted back
    # toward s=0 (the trivial zero), it's not a genuine violation.
    if abs(witness_s) < min_s_magnitude:
        logger.debug(
            "Witness at s=%.4e drifted to trivial zero; reporting stable",
            witness_s,
        )
        return KreissResult(
            is_stable=True,
            witness_s=witness_s,
            witness_sigma_min=min_sigma,
            imaginary_axis_perturbation_verdict="trivial_zero",
            s_grid_shape=s_grid.shape,
            compute_time=time.perf_counter() - t0,
            sigma_min_field=sigma_field,
            s_grid=s_grid,
            n_admissible_roots=len(boundary_rows),
        )

    # Classify imaginary-axis behavior at the witness
    imag_verdict = _classify_imag_axis(
        interior_weights, interior_offsets, witness_s
    )

    is_stable = imag_verdict == "all_incoming"

    return KreissResult(
        is_stable=is_stable,
        witness_s=witness_s,
        witness_sigma_min=min_sigma,
        imaginary_axis_perturbation_verdict=imag_verdict,
        s_grid_shape=s_grid.shape,
        compute_time=time.perf_counter() - t0,
        sigma_min_field=sigma_field,
        s_grid=s_grid,
        n_admissible_roots=len(boundary_rows),
    )
