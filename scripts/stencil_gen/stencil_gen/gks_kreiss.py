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

Augmented Kreiss matrix (plan 48 item 0.2): real closures typically have
more rows that differ from the interior than admissible roots (m > p, with
p = L_left = -min(interior offsets) the admissible-root count for
Re(s) > 0). The interior recurrence then constrains the mode ansatz only
for n >= m - p, so the general decaying solution is

    phi_n = sum_{l=1..p} alpha_l kappa_l^n   for n >= m - p,

with phi_0 .. phi_{m-p-1} free nodal unknowns. The unknown vector
x = (phi_0, ..., phi_{m-p-1}, alpha_1, ..., alpha_p) in C^m yields the
augmented m x m matrix built by kreiss_matrix_general; a GKS eigensolution
exists iff det M(s) = 0 for some Re(s) >= 0. For m = p this reduces
exactly to kreiss_matrix, and all sigma_min-based entry points dispatch to
the general matrix whenever m != p (or row_indices is given).

Near-origin treatment (plan 48 item 0.4): conservative (telescoping)
closures have a structural determinant zero at s = 0 — the kappa = 1
consistency mode satisfies both the interior recurrence and every
constant-annihilating closure row. The old fix masked the whole disk
|s| < 0.1, excising exactly the z ~ 1 borderline where generalized
eigensolutions (Trefethen 1984 Thm 1b) live. Now the mask shrinks to
|s| < 1e-3 (kreiss_stability_check's min_s_magnitude default) and any
candidate witness with |s| < _NEAR_ORIGIN_RADIUS = 0.1 is discriminated
STRUCTURALLY, not via the perturbation classifier (for any consistent
derivative closure the kappa = 1 consistency root perturbs inward,
kappa ~ 1 - s since sum k*a_k = 1, so _classify_imag_axis would return
its violation label for perfectly stable closures — zero discrimination).
The structural test: the candidate is the consistency zero iff the
sigma_min right singular vector aligns with the constant-mode coordinate
vector x_const (nodal entries 1; amplitude 1 on the root continuing to
kappa = 1 as s -> 0) AND every closure row annihilates constants
(full-support row sums ~ 0). A genuine near-origin violation has a
near-null vector NOT aligned with x_const and is reported as a real
witness. Consistency zeros additionally get a vanishing-order fit
sigma_min(s) ~ |s|^nu along a ray into the right half-plane and a
deflated margin sigma_min(s)/|s|^nu (diagnostics only, not a gate).
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
    'outgoing_mode_detected', 'defective'. Stays 'not_checked' for
    near-origin candidates (|s| < _NEAR_ORIGIN_RADIUS), which are
    discriminated structurally instead (plan 48 item 0.4; see
    near_origin_verdict)."""

    defective_kappa_detected: bool = False
    """True if anomalous kappa roots — defective (repeated) admissible roots
    or an admissible-root count that mismatches the boundary-row count —
    were encountered anywhere during the sweep or witness refinement
    (plan 48 item 0.1: anomalies are observable, never masked as inf-stable)."""

    anomalous_points: tuple = ()
    """Deduplicated anomalous s-points as (s, status, classification) triples.
    status is 'defective' or 'count_mismatch' (see min_singular_value_ex);
    classification is the _classify_imag_axis verdict for points on/near the
    imaginary axis, or 'not_classified' for points in the interior of the
    right half-plane where the perturbation test does not apply."""

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

    near_origin_verdict: str = "not_checked"
    """Structural discrimination of a near-origin candidate witness
    (|s| < _NEAR_ORIGIN_RADIUS; plan 48 item 0.4). One of:

    - 'not_checked': the witness is not near the origin (or was resolved
      earlier by the anomaly triage).
    - 'consistency_zero': the near-null vector aligns with the constant
      mode AND all closure rows annihilate constants — the structural
      kappa = 1 consistency zero of a conservative closure, not a GKS
      violation (is_stable is unaffected by the small sigma_min there).
    - 'near_origin_violation': sigma_min < sigma_tol at a near-origin
      witness that is NOT the consistency zero — a genuine GKS
      eigensolution near z ~ 1 (is_stable = False).
    - 'no_violation': near-origin witness, not the consistency zero, but
      sigma_min >= sigma_tol — nothing singular there."""

    vanishing_order: float = float("nan")
    """Fitted exponent nu of sigma_min(s) ~ |s|^nu along a ray into the
    right half-plane, computed only for 'consistency_zero' witnesses.
    The plan-48 reference case (m = 2 telescoping closure) gives nu ~ 2;
    empirically nu grows with the number of constant-annihilating closure
    rows (nu ~ m for the RBF closures). Diagnostic only, not a gate."""

    deflated_margin: float = float("inf")
    """sigma_min(witness_s) / |witness_s|^vanishing_order — the
    consistency-zero-deflated stability margin at the witness (only for
    'consistency_zero' witnesses; inf otherwise)."""


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


def kreiss_matrix_general(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    s: complex,
    *,
    row_indices: Optional[list[int]] = None,
) -> np.ndarray:
    """Build the augmented m x m Kreiss matrix for m closure rows >= p roots.

    Sign convention: this module analyzes u_t = -sum_k a_k u_{n+k} (the
    kappa_roots docstring convention); under the "raw operator weights"
    reading all w-signs flip.

    Let p = number of admissible roots for Re(s) > 0 = L_left =
    -min(interior offsets). (This equals the half-width only for centered
    interiors; the L_left definition is the general one and is used
    throughout.) On the left half-line with m closure rows i = 0..m-1 at
    grid nodes n_i (``row_indices``, default n_i = i) and interior rows for
    n >= m, the interior recurrence constrains the mode ansatz only on
    n >= m - p, so the general decaying solution is

        phi_n = sum_{l=1..p} alpha_l kappa_l^n   for n >= m - p,

    with phi_0 .. phi_{m-p-1} free nodal unknowns. The unknown vector is
    x = (phi_0, ..., phi_{m-p-1}, alpha_1, ..., alpha_p) in C^m, and row i
    of M(s) is:

    - nodal column q (q < m - p)::

        M[i, q] = s * delta_{n_i, q} + w_{i, q}

      (the temporal delta is against the row's grid node n_i, not the row
      index i — they differ once row_indices is non-default; w_{i,q} = 0
      if q is not in the row's support);

    - root column l (0-based, stored at column m - p + l)::

        M[i, m-p+l] = s * kappa_l^{n_i} * [n_i >= m-p]
                      + sum_{j >= m-p, j in support(i)} w_{i,j} * kappa_l^j.

    A GKS eigensolution exists iff det M(s) = 0 for some Re(s) >= 0
    (imaginary-axis subtleties are handled by the perturbation classifier).
    For m = p (and default row_indices) this reduces exactly to
    kreiss_matrix.

    Raises
    ------
    ValueError
        If m < p, if row_indices has the wrong length, if a boundary row
        references a negative grid node, or if the number of admissible
        roots at this s differs from p = L_left (e.g. exactly on the
        imaginary axis where a root sits on the unit circle).
    DefectiveKappaError
        If the admissible roots coalesce at this s.
    """
    offsets = np.asarray(interior_offsets, dtype=int)
    p = -int(np.min(offsets))
    m = len(boundary_rows)
    if m < p:
        raise ValueError(
            f"Number of boundary rows ({m}) < admissible root count p={p}; "
            "the augmented Kreiss matrix requires m >= p"
        )
    if row_indices is None:
        row_indices = list(range(m))
    if len(row_indices) != m:
        raise ValueError(
            f"row_indices has length {len(row_indices)} != number of "
            f"boundary rows ({m})"
        )

    _, admissible, is_defective = kappa_roots(interior_weights, interior_offsets, s)
    if is_defective:
        raise DefectiveKappaError(
            f"Defective admissible kappa roots at s={s}"
        )
    if len(admissible) != p:
        raise ValueError(
            f"Number of admissible roots ({len(admissible)}) != p = L_left "
            f"({p}) at s={s}"
        )

    n_nodal = m - p
    M = np.zeros((m, m), dtype=complex)
    for i, (weights, col_offsets) in enumerate(boundary_rows):
        weights = np.asarray(weights, dtype=complex)
        col_offsets = np.asarray(col_offsets, dtype=int)
        n_i = int(row_indices[i])
        # Temporal contribution s * phi_{n_i}
        if n_i < n_nodal:
            M[i, n_i] += s
        else:
            for ell, kappa in enumerate(admissible):
                M[i, n_nodal + ell] += s * kappa**n_i
        # Spatial operator contribution sum_j w_{ij} phi_j
        for w, j in zip(weights, col_offsets):
            if j < 0:
                raise ValueError(
                    f"Boundary row {i} references negative grid node j={j}; "
                    "left half-line closure rows must have offsets >= 0"
                )
            if j < n_nodal:
                M[i, j] += w
            else:
                for ell, kappa in enumerate(admissible):
                    M[i, n_nodal + ell] += w * kappa**j

    return M


def min_singular_value_ex(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    s: complex,
    *,
    row_indices: Optional[list[int]] = None,
) -> tuple[float, str]:
    """Compute sigma_min(M(s)) with an explicit anomaly status.

    Sign convention: the module analyzes u_t = -sum_k a_k u_{n+k} (see the
    kappa_roots docstring); under the "raw operator weights" reading all
    w-signs flip.

    Dispatch (plan 48 item 0.2): with m = len(boundary_rows) closure rows
    and p = L_left = -min(interior offsets) admissible roots, the square
    kreiss_matrix is used when m == p and row_indices is None; otherwise
    the augmented kreiss_matrix_general is used (m > p adds m - p nodal
    unknowns; m < p is a hard mismatch reported as "count_mismatch").

    Parameters
    ----------
    row_indices : list[int], optional
        Grid node n_i of each closure row (default 0..m-1); forwarded to
        kreiss_matrix_general.

    Returns
    -------
    (sigma, status) : tuple[float, str]
        status is one of:

        - ``"ok"``: sigma is the genuine minimum singular value of M(s).
        - ``"defective"``: admissible kappa roots coalesce at this s
          (DefectiveKappaError); sigma is np.inf, meaning *not computable* —
          NOT "maximally stable". Anomalies are deliberately not mapped to
          sigma=0, which would create false witnesses for the refiner to
          chase (plan 48 item 0.1).
        - ``"count_mismatch"``: the number of admissible roots at this s
          differs from its structural count p = L_left (or m < p, which is
          unresolvable); sigma is np.inf as above.
    """
    p = -int(np.min(np.asarray(interior_offsets, dtype=int)))
    m = len(boundary_rows)
    try:
        if m != p or row_indices is not None:
            M = kreiss_matrix_general(
                interior_weights, interior_offsets, boundary_rows, s,
                row_indices=row_indices,
            )
        else:
            M = kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s)
    except DefectiveKappaError:
        return np.inf, "defective"
    except ValueError:
        return np.inf, "count_mismatch"
    svs = np.linalg.svd(M, compute_uv=False)
    return float(svs[-1]), "ok"


def min_singular_value(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    s: complex,
    *,
    row_indices: Optional[list[int]] = None,
) -> float:
    """Compute sigma_min(M(s)), the minimum singular value of the Kreiss matrix.

    Thin compatibility wrapper around min_singular_value_ex that discards the
    anomaly status (and inherits its m == p / m != p dispatch, plan 48 item
    0.2). Returns np.inf on DefectiveKappaError or root-count mismatch;
    callers that must distinguish anomalies from genuinely large sigma_min
    should use min_singular_value_ex instead.
    """
    sigma, _ = min_singular_value_ex(
        interior_weights, interior_offsets, boundary_rows, s,
        row_indices=row_indices,
    )
    return sigma


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
    *,
    anomalies: Optional[list] = None,
    row_indices: Optional[list[int]] = None,
) -> tuple[np.ndarray, int]:
    """Evaluate min_singular_value at every point of s_grid.

    Parameters
    ----------
    anomalies : list, optional
        Side list: every grid point where sigma_min is not computable
        (status != "ok" from min_singular_value_ex) is appended as
        ``(s, status)``. Such points carry sigma=inf in the field so they
        can never be selected as a witness — the caller must inspect the
        side list to keep them observable (plan 48 item 0.1).
    row_indices : list[int], optional
        Grid node of each closure row; forwarded to min_singular_value_ex
        (plan 48 item 0.2).

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
        sigma_flat[i], status = min_singular_value_ex(
            interior_weights, interior_offsets, boundary_rows, s,
            row_indices=row_indices,
        )
        if status != "ok" and anomalies is not None:
            anomalies.append((complex(s), status))
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
    anomalies: Optional[list] = None,
    row_indices: Optional[list[int]] = None,
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
    anomalies : list, optional
        Side list: every evaluation point where sigma_min is not computable
        is appended as ``(s, status)`` — see _sweep_grid. Anomalies carry
        sigma=inf (never sigma=0), so Nelder-Mead moves away from them
        rather than chasing false witnesses (plan 48 item 0.1).
    row_indices : list[int], optional
        Grid node of each closure row; forwarded to min_singular_value_ex
        (plan 48 item 0.2).

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
        sv, status = min_singular_value_ex(
            interior_weights, interior_offsets, boundary_rows, s,
            row_indices=row_indices,
        )
        if status != "ok" and anomalies is not None:
            anomalies.append((s, status))
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


_ANOMALY_AXIS_TOL = 1e-3
"""Re(s) threshold below which an anomalous point counts as on/near the
imaginary axis and is classified with _classify_imag_axis."""


def _triage_anomalies(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    anomalies: list,
) -> tuple:
    """Deduplicate and classify anomalous s-points from sweep/refinement.

    Each anomaly is a ``(s, status)`` pair with status in
    {'defective', 'count_mismatch'}. Points on/near the imaginary axis
    (Re(s) <= _ANOMALY_AXIS_TOL) are classified with _classify_imag_axis;
    points in the interior of the right half-plane cannot be classified by
    the imaginary-axis perturbation test and are marked 'not_classified'.

    Returns a tuple of ``(s, status, classification)`` triples. Only a
    classification of 'all_incoming' resolves an anomaly as benign;
    'outgoing_mode_detected' is a violation; anything else
    ('defective', 'no_candidates', 'not_classified') leaves the anomaly
    unresolved and the caller must fail conservatively (plan 48 item 0.1).
    """
    seen = set()
    triaged = []
    for s_a, status in anomalies:
        key = (s_a, status)
        if key in seen:
            continue
        seen.add(key)
        if abs(s_a.real) <= _ANOMALY_AXIS_TOL:
            classification = _classify_imag_axis(
                interior_weights, interior_offsets, s_a
            )
        else:
            classification = "not_classified"
        triaged.append((s_a, status, classification))
    return tuple(triaged)


_NEAR_ORIGIN_RADIUS = 0.1
"""|s| threshold below which a candidate witness gets the structural
consistency-zero discrimination instead of the perturbation classifier
(plan 48 item 0.4). Matches the old min_s_magnitude disk that used to be
masked outright."""

_CONST_ANGLE_TOL = 0.05
"""Maximum angle (radians) between the sigma_min right singular vector and
the constant-mode coordinate vector x_const for a near-origin candidate to
classify as the structural consistency zero. Calibrated empirically: true
consistency zeros at the |s| = 1e-3 refinement barrier show angles ~5e-4
rad (growing ~linearly in |s|, ~0.023 rad at |s| = 0.05), while an
engineered genuine zero at s = 0.05 shows ~0.22 rad."""

_ROWSUM_TOL = 1e-8
"""Relative tolerance for the closure-row constant-annihilation test
(|sum w| <= _ROWSUM_TOL * max(1, sum |w|))."""


def _constant_mode_vector(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    s: complex,
) -> np.ndarray:
    """Build the normalized constant-mode coordinate vector x_const.

    In the augmented coordinates x = (phi_0, ..., phi_{m-p-1}, alpha_1,
    ..., alpha_p) of kreiss_matrix_general, the kappa = 1 consistency mode
    phi_n = 1 has nodal entries 1 and amplitude 1 on the admissible root
    continuing to kappa = 1 as s -> 0 (identified as argmin |kappa - 1|
    at this s; for a consistent derivative interior kappa ~ 1 - s), with
    all other amplitudes 0 (plan 48 item 0.4). For m = p there are no
    nodal entries and x_const is the pure amplitude vector.
    """
    p = -int(np.min(np.asarray(interior_offsets, dtype=int)))
    m = len(boundary_rows)
    n_nodal = m - p
    _, admissible, _ = kappa_roots(interior_weights, interior_offsets, s)
    if len(admissible) == 0:
        raise ValueError(f"No admissible roots at s={s}; cannot build x_const")
    ell_star = int(np.argmin(np.abs(admissible - 1.0)))
    x = np.zeros(m, dtype=complex)
    x[:n_nodal] = 1.0
    x[n_nodal + ell_star] = 1.0
    return x / np.linalg.norm(x)


def _rows_annihilate_constants(boundary_rows: list[BoundaryRow]) -> bool:
    """True iff every closure row has full-support row sum ~ 0.

    The structural consistency zero at s = 0 exists only for conservative
    (telescoping) closures whose rows annihilate constants; a Dirichlet-type
    row (sum != 0) removes the constant mode and has no such zero.
    """
    for weights, _ in boundary_rows:
        w = np.asarray(weights, dtype=complex)
        if abs(np.sum(w)) > _ROWSUM_TOL * max(1.0, float(np.sum(np.abs(w)))):
            return False
    return True


def _classify_near_origin(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    s: complex,
    *,
    row_indices: Optional[list[int]] = None,
    angle_tol: float = _CONST_ANGLE_TOL,
) -> tuple[bool, float]:
    """Structurally discriminate a near-origin candidate witness.

    Plan 48 item 0.4: the perturbation classifier cannot discriminate here
    (the kappa = 1 consistency root of ANY consistent derivative closure
    perturbs inward, kappa ~ 1 - s, so _classify_imag_axis returns its
    violation label for perfectly stable closures). Instead the candidate
    is the structural consistency zero iff

    1. the sigma_min right singular vector of M(s) aligns with the
       constant-mode coordinate vector x_const (angle < angle_tol), AND
    2. every closure row annihilates constants (full-support row sums ~ 0).

    A genuine near-origin violation has a near-null vector NOT aligned
    with x_const (or lives in a non-conservative closure) and is a real
    witness.

    Returns
    -------
    (is_consistency_zero, angle) : tuple[bool, float]
        angle is the alignment angle in radians (nan if M(s) was not
        computable, in which case is_consistency_zero is False).
    """
    p = -int(np.min(np.asarray(interior_offsets, dtype=int)))
    m = len(boundary_rows)
    try:
        if m != p or row_indices is not None:
            M = kreiss_matrix_general(
                interior_weights, interior_offsets, boundary_rows, s,
                row_indices=row_indices,
            )
        else:
            M = kreiss_matrix(interior_weights, interior_offsets, boundary_rows, s)
        x_const = _constant_mode_vector(
            interior_weights, interior_offsets, boundary_rows, s
        )
    except (DefectiveKappaError, ValueError):
        return False, float("nan")

    _, _, Vh = np.linalg.svd(M)
    v = Vh[-1, :].conj()  # right singular vector for sigma_min
    alignment = min(1.0, abs(np.vdot(v, x_const)))
    angle = float(np.arccos(alignment))

    is_cz = angle < angle_tol and _rows_annihilate_constants(boundary_rows)
    return is_cz, angle


def _fit_vanishing_order(
    interior_weights: np.ndarray,
    interior_offsets: np.ndarray,
    boundary_rows: list[BoundaryRow],
    witness_s: complex,
    witness_sigma: float,
    *,
    row_indices: Optional[list[int]] = None,
    t_min: float = 1e-3,
    t_max: float = 3e-2,
    n_pts: int = 6,
) -> tuple[float, float]:
    """Fit sigma_min(s) ~ |s|^nu along a ray into the right half-plane.

    Corroborates a consistency-zero classification (plan 48 item 0.4): the
    structural zero vanishes algebraically, with nu ~ 2 for the m = 2
    telescoping reference closure (empirically nu grows with the number of
    constant-annihilating rows, nu ~ m for the RBF closures). The ray is
    the positive real axis — the vanishing order is direction-independent
    to leading order, and the real axis avoids the imaginary-axis
    root-count knife edge.

    Returns
    -------
    (nu, deflated_margin) : tuple[float, float]
        nu is the fitted exponent (nan if fewer than 3 finite samples);
        deflated_margin is witness_sigma / |witness_s|^nu (inf if nu is
        nan).
    """
    ts = np.logspace(np.log10(t_min), np.log10(t_max), n_pts)
    log_t, log_sigma = [], []
    for t in ts:
        sigma, status = min_singular_value_ex(
            interior_weights, interior_offsets, boundary_rows, complex(t, 0.0),
            row_indices=row_indices,
        )
        if status == "ok" and np.isfinite(sigma) and sigma > 0:
            log_t.append(np.log(t))
            log_sigma.append(np.log(sigma))
    if len(log_t) < 3:
        return float("nan"), float("inf")
    nu = float(np.polyfit(log_t, log_sigma, 1)[0])
    deflated = float(witness_sigma / abs(witness_s) ** nu)
    return nu, deflated


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
    min_s_magnitude: float = 1e-3,
    row_indices: Optional[list[int]] = None,
) -> KreissResult:
    """Run the full Kreiss determinant stability check.

    Sweeps the s-grid, optionally refines any witness, classifies
    imaginary-axis modes, and returns a KreissResult.

    Sign convention: the module analyzes u_t = -sum_k a_k u_{n+k} (see the
    kappa_roots docstring); under the "raw operator weights" reading all
    w-signs flip.

    Parameters
    ----------
    interior_weights : np.ndarray
        Weights of the interior finite-difference stencil.
    interior_offsets : np.ndarray
        Grid offsets of the interior stencil (e.g., [-2, -1, 0, 1, 2]).
    boundary_rows : list[BoundaryRow]
        Each element is (weights, column_offsets) for one boundary row.
        The row count m may exceed the admissible-root count
        p = L_left = -min(interior_offsets): whenever m != p (or
        row_indices is given) the augmented Kreiss matrix
        kreiss_matrix_general is used (plan 48 item 0.2).
    s_grid_params : dict, optional
        Keyword arguments passed to make_s_grid.
    sigma_tol : float
        Threshold below which sigma_min indicates instability.
    refine : bool
        Whether to refine the witness via Nelder-Mead.
    min_s_magnitude : float
        Radius of the excluded disk around s=0 for grid masking and the
        witness-refinement barrier (default 1e-3; plan 48 item 0.4 —
        previously 0.1, which excised the z ~ 1 borderline where
        generalized eigensolutions live). For conservative closures
        sigma_min(M(0)) = 0 structurally (the constant mode kappa=1
        satisfies interior and closure equations); that zero is handled by
        the near-origin discrimination below, not by masking.
    row_indices : list[int], optional
        Grid node n_i of each closure row (default 0..m-1); forwarded to
        the sigma_min evaluations (plan 48 item 0.2).

    Near-origin treatment (plan 48 item 0.4)
    ----------------------------------------
    Any candidate witness with |s| < _NEAR_ORIGIN_RADIUS = 0.1 is
    discriminated STRUCTURALLY (never via the perturbation classifier,
    which cannot distinguish the consistency zero from a violation there):
    it is the structural consistency zero iff the sigma_min right singular
    vector aligns with the constant-mode vector x_const AND all closure
    rows annihilate constants — then is_stable is unaffected by the small
    sigma_min and a vanishing-order fit / deflated margin is reported.
    Otherwise the witness is genuine: sigma_min < sigma_tol there means
    is_stable=False with near_origin_verdict='near_origin_violation'.

    Anomaly handling (plan 48 item 0.1)
    -----------------------------------
    Points where sigma_min is not computable (defective admissible roots or
    admissible-root count != boundary-row count) are recorded during the
    sweep and refinement, deduplicated, and triaged:

    - a near-axis anomaly classified 'all_incoming' is benign (the anomaly
      is a knife-edge artifact of modes that are incoming just off the axis);
    - a near-axis anomaly classified 'outgoing_mode_detected' is a GKS
      violation;
    - any other anomaly (off-axis, or near-axis with 'defective' /
      'no_candidates') is unresolved: the check NEVER reports
      is_stable=True while unresolved anomalies exist — the verdict becomes
      'defective' (conservative fail).

    Whenever any anomaly was recorded, defective_kappa_detected is True and
    the triaged points are exposed in KreissResult.anomalous_points.
    """
    import time

    t0 = time.perf_counter()

    # Structural admissible-root count p = L_left (equals len(boundary_rows)
    # only in the square m == p case; plan 48 item 0.2).
    p_structural = -int(np.min(np.asarray(interior_offsets, dtype=int)))

    grid_kw = s_grid_params or {}
    s_grid = make_s_grid(**grid_kw)

    anomalies: list = []
    sigma_field, _ = _sweep_grid(
        interior_weights, interior_offsets, boundary_rows, s_grid,
        anomalies=anomalies,
        row_indices=row_indices,
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
        witness_s = _refine_witness(
            interior_weights, interior_offsets, boundary_rows, witness_s,
            min_s_magnitude=min_s_magnitude,
            anomalies=anomalies,
            row_indices=row_indices,
        )
        min_sigma = min_singular_value(
            interior_weights, interior_offsets, boundary_rows, witness_s,
            row_indices=row_indices,
        )

    # Triage anomalous points: classify near-axis ones, and fail
    # conservatively on anything unresolved (plan 48 item 0.1).
    anomalous_points = _triage_anomalies(
        interior_weights, interior_offsets, anomalies
    )
    any_anomaly = len(anomalous_points) > 0
    axis_violations = [
        s_a for s_a, _, cls in anomalous_points
        if cls == "outgoing_mode_detected"
    ]
    unresolved = [
        (s_a, status) for s_a, status, cls in anomalous_points
        if cls not in ("all_incoming", "outgoing_mode_detected")
    ]

    if axis_violations:
        return KreissResult(
            is_stable=False,
            witness_s=axis_violations[0],
            witness_sigma_min=float("inf"),
            imaginary_axis_perturbation_verdict="outgoing_mode_detected",
            defective_kappa_detected=True,
            anomalous_points=anomalous_points,
            s_grid_shape=s_grid.shape,
            compute_time=time.perf_counter() - t0,
            sigma_min_field=sigma_field,
            s_grid=s_grid,
            n_admissible_roots=p_structural,
        )

    if unresolved:
        return KreissResult(
            is_stable=False,
            witness_s=unresolved[0][0],
            witness_sigma_min=float("inf"),
            imaginary_axis_perturbation_verdict="defective",
            defective_kappa_detected=True,
            anomalous_points=anomalous_points,
            s_grid_shape=s_grid.shape,
            compute_time=time.perf_counter() - t0,
            sigma_min_field=sigma_field,
            s_grid=s_grid,
            n_admissible_roots=p_structural,
        )

    # Near-origin candidate witness: discriminate the structural
    # consistency zero STRUCTURALLY, never via the perturbation classifier
    # (plan 48 item 0.4). This runs regardless of sigma_tol so that a
    # conservative closure's near-origin dip is classified (and its
    # deflated margin reported) even when sigma_min is above tolerance.
    if abs(witness_s) < _NEAR_ORIGIN_RADIUS and np.isfinite(min_sigma):
        is_cz, angle = _classify_near_origin(
            interior_weights, interior_offsets, boundary_rows, witness_s,
            row_indices=row_indices,
        )
        if is_cz:
            nu, deflated = _fit_vanishing_order(
                interior_weights, interior_offsets, boundary_rows,
                witness_s, min_sigma,
                row_indices=row_indices,
            )
            logger.debug(
                "Witness at s=%s is the structural consistency zero "
                "(angle=%.2e rad, nu=%.2f, deflated margin=%.3e)",
                witness_s, angle, nu, deflated,
            )
            return KreissResult(
                is_stable=True,
                witness_s=witness_s,
                witness_sigma_min=min_sigma,
                defective_kappa_detected=any_anomaly,
                anomalous_points=anomalous_points,
                s_grid_shape=s_grid.shape,
                compute_time=time.perf_counter() - t0,
                sigma_min_field=sigma_field,
                s_grid=s_grid,
                n_admissible_roots=p_structural,
                near_origin_verdict="consistency_zero",
                vanishing_order=nu,
                deflated_margin=deflated,
            )
        # Not the consistency zero: a genuine near-origin zero is a real
        # witness (near-null vector not aligned with x_const).
        return KreissResult(
            is_stable=min_sigma >= sigma_tol,
            witness_s=witness_s,
            witness_sigma_min=min_sigma,
            defective_kappa_detected=any_anomaly,
            anomalous_points=anomalous_points,
            s_grid_shape=s_grid.shape,
            compute_time=time.perf_counter() - t0,
            sigma_min_field=sigma_field,
            s_grid=s_grid,
            n_admissible_roots=p_structural,
            near_origin_verdict=(
                "no_violation" if min_sigma >= sigma_tol
                else "near_origin_violation"
            ),
        )

    # After refinement, apply sigma_tol to determine stability
    if min_sigma >= sigma_tol:
        return KreissResult(
            is_stable=True,
            witness_s=witness_s,
            witness_sigma_min=min_sigma,
            defective_kappa_detected=any_anomaly,
            anomalous_points=anomalous_points,
            s_grid_shape=s_grid.shape,
            compute_time=time.perf_counter() - t0,
            sigma_min_field=sigma_field,
            s_grid=s_grid,
            n_admissible_roots=p_structural,
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
        defective_kappa_detected=any_anomaly,
        anomalous_points=anomalous_points,
        s_grid_shape=s_grid.shape,
        compute_time=time.perf_counter() - t0,
        sigma_min_field=sigma_field,
        s_grid=s_grid,
        n_admissible_roots=p_structural,
    )
