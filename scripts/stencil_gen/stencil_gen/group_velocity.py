"""Group velocity analysis for finite difference stencils.

Provides tools for computing modified wavenumber, phase velocity, and group
velocity from stencil coefficients.  For the model equation u_t + u_x = 0
semi-discretized as du/dt = -D*u, the modified wavenumber kappa*(xi) of D
gives the dispersion relation omega = Im(kappa*(xi)).  The group velocity
is C(xi) = d(omega)/d(xi) = d(Im(kappa*))/d(xi).

References:
  Trefethen, "Group velocity in finite difference schemes", 1982.
  Trefethen, "Stability and group velocity", 1983 (GKS connection).
"""

from dataclasses import dataclass

import numpy as np


def modified_wavenumber(
    weights,
    i_eval: int,
    node_indices,
    xi_array: np.ndarray,
) -> np.ndarray:
    """Compute modified wavenumber kappa*(xi) for given stencil weights.

    Parameters
    ----------
    weights : array-like
        Stencil coefficients w_j.
    i_eval : int
        Grid index where derivative is evaluated.
    node_indices : array-like of int
        Grid indices used by the stencil.
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].

    Returns
    -------
    np.ndarray (complex)
        kappa*(xi) = sum_j w_j exp(i (j - i_eval) xi)
    """
    offsets = np.asarray(node_indices) - i_eval
    return modified_wavenumber_nonuniform(weights, offsets, xi_array)


def group_velocity_numerical(kappa_star: np.ndarray, xi_array: np.ndarray) -> np.ndarray:
    """Compute group velocity C(xi) = d(Im(kappa*))/d(xi) numerically.

    Uses numpy.gradient for the numerical differentiation.

    Parameters
    ----------
    kappa_star : np.ndarray (complex)
        Modified wavenumber array.
    xi_array : np.ndarray
        Wavenumber values xi.

    Returns
    -------
    np.ndarray (real)
        Group velocity C(xi).
    """
    return np.gradient(np.imag(kappa_star), xi_array)


def group_velocity_exact(
    weights,
    i_eval: int,
    node_indices,
    xi_array: np.ndarray,
) -> np.ndarray:
    """Compute group velocity analytically from stencil weights.

    C(xi) = Re(sum_j w_j (j - i_eval) exp(i (j - i_eval) xi))

    This avoids numerical differentiation entirely.

    Parameters
    ----------
    weights : array-like
        Stencil coefficients w_j.
    i_eval : int
        Grid index where derivative is evaluated.
    node_indices : array-like of int
        Grid indices used by the stencil.
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].

    Returns
    -------
    np.ndarray (real)
        Group velocity C(xi).
    """
    offsets = np.asarray(node_indices) - i_eval
    return group_velocity_exact_nonuniform(weights, offsets, xi_array)


def modified_wavenumber_nonuniform(
    weights,
    offsets,
    xi_array: np.ndarray,
) -> np.ndarray:
    """Compute modified wavenumber for non-uniformly spaced stencil nodes.

    Generalization of :func:`modified_wavenumber` for real-valued offsets
    (e.g. cut-cell stencils where the wall position is at a fractional
    distance from the evaluation point).

    Parameters
    ----------
    weights : array-like
        Stencil coefficients w_j.
    offsets : array-like of float
        Normalized distances (x_j - x_i)/h from the evaluation point.
        May be non-integer for cut-cell grids.
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].

    Returns
    -------
    np.ndarray (complex)
        kappa*(xi) = sum_j w_j exp(i * offset_j * xi)
    """
    w = np.asarray(weights, dtype=complex)
    d = np.asarray(offsets, dtype=float)
    phase = np.exp(1j * np.outer(xi_array, d))
    return phase @ w


def group_velocity_exact_nonuniform(
    weights,
    offsets,
    xi_array: np.ndarray,
) -> np.ndarray:
    """Compute group velocity analytically for non-uniform offsets.

    C(xi) = Re(sum_j w_j * offset_j * exp(i * offset_j * xi))

    Generalization of :func:`group_velocity_exact` for real-valued offsets.

    Parameters
    ----------
    weights : array-like
        Stencil coefficients w_j.
    offsets : array-like of float
        Normalized distances (x_j - x_i)/h from the evaluation point.
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].

    Returns
    -------
    np.ndarray (real)
        Group velocity C(xi).
    """
    w = np.asarray(weights, dtype=complex)
    d = np.asarray(offsets, dtype=float)
    phase = np.exp(1j * np.outer(xi_array, d))
    return np.real(phase @ (w * d))


def phase_velocity(kappa_star: np.ndarray, xi_array: np.ndarray) -> np.ndarray:
    """Compute phase velocity c(xi) = Im(kappa*(xi)) / xi.

    At xi=0 the limit is taken from the next nonzero xi value.

    Parameters
    ----------
    kappa_star : np.ndarray (complex)
        Modified wavenumber array.
    xi_array : np.ndarray
        Wavenumber values xi.

    Returns
    -------
    np.ndarray (real)
        Phase velocity c(xi).
    """
    c = np.empty_like(xi_array, dtype=float)
    nonzero = xi_array != 0.0
    c[nonzero] = np.imag(kappa_star[nonzero]) / xi_array[nonzero]
    # Handle xi=0 via L'Hopital: lim Im(kappa*)/xi = group velocity at 0
    zero_mask = ~nonzero
    if np.any(zero_mask):
        # Use the first nonzero point as approximation
        first_nonzero = np.argmax(nonzero)
        c[zero_mask] = c[first_nonzero] if first_nonzero > 0 else 1.0
    return c


def group_velocity_error(
    C: np.ndarray,
    C_exact: float = 1.0,
) -> np.ndarray:
    """Compute relative group velocity error.

    Parameters
    ----------
    C : np.ndarray
        Computed group velocity.
    C_exact : float
        Exact group velocity (default 1.0 for u_t + u_x = 0).

    Returns
    -------
    np.ndarray
        (C - C_exact) / C_exact
    """
    return (C - C_exact) / C_exact


@dataclass
class GroupVelocity2DResult:
    """2D group velocity analysis results for tensor-product stencils.

    For dimension-by-dimension operators where the 2D dispersion relation
    factors as omega = a*kappa_x*(xi) + b*kappa_y*(eta), the group velocity
    vector is C = (C_x, C_y) with C_x = a * d(Im(kappa_x*))/d(xi) and
    C_y = b * d(Im(kappa_y*))/d(eta).

    All 2D arrays use shape (N_xi, N_eta) with indexing='ij'.
    """

    xi: np.ndarray  # 1D wavenumber array for x-direction
    eta: np.ndarray  # 1D wavenumber array for y-direction
    C_x: np.ndarray  # 2D group velocity x-component
    C_y: np.ndarray  # 2D group velocity y-component
    speed: np.ndarray  # 2D group speed |C|
    angle: np.ndarray  # 2D group propagation angle atan2(C_y, C_x)
    angle_error: np.ndarray  # 2D angle deviation from wave normal


@dataclass
class AnisotropyResult:
    """Anisotropy profile for a given interior scheme at fixed wavenumber magnitude.

    For a wave propagating at angle theta (advection velocity (cos(theta),
    sin(theta))) with wavenumber magnitude xi_mag, reports the numerical
    group velocity vector and its deviation from the exact result.

    The exact group velocity is (cos(theta), sin(theta)) with speed 1.
    """

    theta: np.ndarray  # propagation angle array
    C_x: np.ndarray  # group velocity x-component
    C_y: np.ndarray  # group velocity y-component
    speed: np.ndarray  # group speed |C| (equals speed ratio since exact speed = 1)
    angle: np.ndarray  # group propagation angle atan2(C_y, C_x)
    angle_error: np.ndarray  # angle deviation from propagation direction theta


def _make_anisotropy_result(
    theta: np.ndarray, C_x: np.ndarray, C_y: np.ndarray,
) -> "AnisotropyResult":
    """Construct AnisotropyResult from 2D group velocity components."""
    speed = np.sqrt(C_x**2 + C_y**2)
    angle = np.arctan2(C_y, C_x)
    angle_error = angle - theta
    return AnisotropyResult(
        theta=theta, C_x=C_x, C_y=C_y,
        speed=speed, angle=angle, angle_error=angle_error,
    )


@dataclass
class GroupVelocityProfile:
    """Group velocity analysis results for a single stencil row."""

    xi: np.ndarray
    kappa_star: np.ndarray
    phase_velocity: np.ndarray
    group_velocity: np.ndarray
    gv_error: np.ndarray
    order: int
    cutoff_xi: float  # first xi beyond which C stays permanently non-positive


@dataclass
class GKSModeInfo:
    """Diagnostic information for a boundary-localized eigenmode.

    Bridges per-stencil group velocity analysis with full-operator eigenvalue
    analysis by identifying nearly-neutral eigenmodes concentrated near a
    boundary and checking whether the interior stencil's group velocity at
    the mode's dominant wavenumber directs energy into the domain.
    """

    eigenvalue: complex  # eigenvalue of -D_bc
    boundary_wavenumber: float  # dominant xi from FFT of boundary eigenvector portion
    group_velocity: float  # interior C(xi) at boundary_wavenumber
    is_outgoing: bool  # True if mode radiates energy from boundary into domain


def group_velocity_2d(
    kappa_x_star: np.ndarray,
    kappa_y_star: np.ndarray,
    xi_array: np.ndarray,
    eta_array: np.ndarray,
    a: float = 1.0,
    b: float = 1.0,
) -> GroupVelocity2DResult:
    """Compute 2D group velocity for tensor-product stencils.

    For dimension-by-dimension operators where the 2D dispersion relation
    factors as ``omega = a*kappa_x*(xi) + b*kappa_y*(eta)``, the group
    velocity vector is:

    - ``C_x(xi, eta) = a * d(Im(kappa_x*))/d(xi)``  (depends only on xi)
    - ``C_y(xi, eta) = b * d(Im(kappa_y*))/d(eta)``  (depends only on eta)

    Parameters
    ----------
    kappa_x_star : np.ndarray (complex)
        Modified wavenumber for x-direction stencil, shape (N_xi,).
    kappa_y_star : np.ndarray (complex)
        Modified wavenumber for y-direction stencil, shape (N_eta,).
    xi_array : np.ndarray
        Wavenumber values for x-direction, shape (N_xi,).
    eta_array : np.ndarray
        Wavenumber values for y-direction, shape (N_eta,).
    a : float
        Wave speed coefficient in x-direction (default 1.0).
    b : float
        Wave speed coefficient in y-direction (default 1.0).

    Returns
    -------
    GroupVelocity2DResult
    """
    # 1D group velocities via numerical differentiation
    C_x_1d = a * np.gradient(np.imag(kappa_x_star), xi_array)
    C_y_1d = b * np.gradient(np.imag(kappa_y_star), eta_array)

    # Broadcast to 2D grid with shape (N_xi, N_eta)
    C_x = np.broadcast_to(C_x_1d[:, np.newaxis], (len(xi_array), len(eta_array))).copy()
    C_y = np.broadcast_to(C_y_1d[np.newaxis, :], (len(xi_array), len(eta_array))).copy()

    speed = np.sqrt(C_x**2 + C_y**2)
    angle = np.arctan2(C_y, C_x)

    # Wave normal angle on the 2D grid
    xi_2d, eta_2d = np.meshgrid(xi_array, eta_array, indexing="ij")
    theta_wave = np.arctan2(eta_2d, xi_2d)
    angle_error = angle - theta_wave

    return GroupVelocity2DResult(
        xi=xi_array,
        eta=eta_array,
        C_x=C_x,
        C_y=C_y,
        speed=speed,
        angle=angle,
        angle_error=angle_error,
    )


def anisotropy_profile(
    p: int,
    nu: int,
    theta_array: np.ndarray,
    xi_mag: float,
) -> AnisotropyResult:
    """Compute group speed and angle error vs propagation angle for an interior scheme.

    For the advection equation ``u_t + cos(theta)*u_x + sin(theta)*u_y = 0``
    discretized with tensor-product stencils of half-bandwidth *p* and
    derivative order *nu*, this function evaluates the numerical group velocity
    at wavenumber magnitude *xi_mag* for each propagation angle in
    *theta_array*.

    The exact group velocity is ``(cos(theta), sin(theta))`` with unit speed.
    The numerical group velocity deviates due to grid anisotropy — e.g., for
    E2, Trefethen (1982) shows that diagonal propagation (theta = pi/4) has
    higher group speed than axis-aligned propagation (theta = 0).

    Parameters
    ----------
    p : int
        Interior half-bandwidth (E2 → p=1, E4 → p=2, E6 → p=3, E8 → p=4).
    nu : int
        Derivative order (typically 1 for advection).
    theta_array : np.ndarray
        Wave propagation angles in radians.
    xi_mag : float
        Wavenumber magnitude |xi| in [0, pi].

    Returns
    -------
    AnisotropyResult
    """
    from stencil_gen.interior import derive_interior, full_gamma_array

    # Interior stencil weights
    coeffs = derive_interior(0, p, nu)
    w = [float(c) for c in full_gamma_array(coeffs)]
    nodes = list(range(-p, p + 1))

    # Wavenumber components (g is even, so use |cos|, |sin|)
    xi_vals = xi_mag * np.abs(np.cos(theta_array))
    eta_vals = xi_mag * np.abs(np.sin(theta_array))

    # Evaluate 1D group velocity analytically at these wavenumbers
    C_at_xi = group_velocity_exact(w, 0, nodes, xi_vals)
    C_at_eta = group_velocity_exact(w, 0, nodes, eta_vals)

    # 2D group velocity components: C = (a * g(xi), b * g(eta))
    C_x = np.cos(theta_array) * C_at_xi
    C_y = np.sin(theta_array) * C_at_eta

    return _make_anisotropy_result(theta_array, C_x, C_y)


def boundary_group_velocity_2d(
    boundary_rows_x: dict[int, "GroupVelocityProfile"],
    interior_y: "GroupVelocityProfile",
    theta_array: np.ndarray,
    xi_mag: float,
) -> dict[int, AnisotropyResult]:
    """Compute 2D group velocity at a boundary using boundary x-stencils and interior y.

    At a boundary in x (left wall), the x-direction uses boundary stencils
    while the y-direction uses interior stencils.  The 2D dispersion relation
    near the boundary is ``omega = a*kappa_x_bdy*(xi) + b*kappa_y_int*(eta)``
    where ``a = cos(theta)``, ``b = sin(theta)`` for a wave propagating at
    angle *theta*.

    This reveals whether the boundary distorts the group velocity angle,
    bending waves toward or away from the boundary.

    Parameters
    ----------
    boundary_rows_x : dict[int, GroupVelocityProfile]
        Boundary group velocity profiles for the x-direction (from
        :func:`boundary_group_velocity` or similar).
    interior_y : GroupVelocityProfile
        Interior group velocity profile for the y-direction.
    theta_array : np.ndarray
        Wave propagation angles in radians.
    xi_mag : float
        Wavenumber magnitude |xi| in [0, pi].

    Returns
    -------
    dict[int, AnisotropyResult]
        Keyed by boundary row index.  Each entry gives the 2D group velocity
        at that boundary row for all propagation angles.
    """
    # Wavenumber components along each direction
    xi_vals = xi_mag * np.abs(np.cos(theta_array))
    eta_vals = xi_mag * np.abs(np.sin(theta_array))

    # Interior y-direction group velocity interpolated at eta_vals
    C_y_1d = np.interp(eta_vals, interior_y.xi, interior_y.group_velocity)

    results: dict[int, AnisotropyResult] = {}
    for row_idx, prof in boundary_rows_x.items():
        # Boundary x-direction group velocity interpolated at xi_vals
        C_x_1d = np.interp(xi_vals, prof.xi, prof.group_velocity)

        # 2D group velocity components
        C_x = np.cos(theta_array) * C_x_1d
        C_y = np.sin(theta_array) * C_y_1d

        results[row_idx] = _make_anisotropy_result(theta_array, C_x, C_y)

    return results


def _build_profile(
    weights,
    offsets,
    xi_array: np.ndarray,
    order: int,
) -> GroupVelocityProfile:
    """Build a GroupVelocityProfile from stencil weights and node offsets."""
    kstar = modified_wavenumber_nonuniform(weights, offsets, xi_array)
    C = group_velocity_exact_nonuniform(weights, offsets, xi_array)
    c = phase_velocity(kstar, xi_array)
    gv_err = group_velocity_error(C)

    # Find cutoff: first xi beyond which C stays non-positive.
    # Scan from high end to handle non-monotonic boundary stencils where
    # C(xi) may dip below zero briefly then recover.
    last_positive_idx = 0
    for idx in range(1, len(xi_array)):
        if C[idx] > 0.0:
            last_positive_idx = idx
    if last_positive_idx + 1 < len(xi_array):
        cutoff = float(xi_array[last_positive_idx + 1])
    else:
        cutoff = float(xi_array[-1])

    return GroupVelocityProfile(
        xi=xi_array,
        kappa_star=kstar,
        phase_velocity=c,
        group_velocity=C,
        gv_error=gv_err,
        order=order,
        cutoff_xi=cutoff,
    )


def interior_group_velocity(
    p: int,
    nu: int,
    xi_array: np.ndarray,
) -> GroupVelocityProfile:
    """Compute group velocity profile for an interior scheme.

    Parameters
    ----------
    p : int
        RHS half-bandwidth (explicit scheme, s=0).
    nu : int
        Derivative order (1 or 2).
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].

    Returns
    -------
    GroupVelocityProfile
    """
    from stencil_gen.interior import derive_interior, full_gamma_array

    coeffs = derive_interior(0, p, nu)
    w = [float(c) for c in full_gamma_array(coeffs)]
    nodes = list(range(-p, p + 1))

    return _build_profile(w, np.asarray(nodes), xi_array, order=2 * p)


def boundary_group_velocity(
    p: int,
    q: int,
    nextra: int,
    nu: int,
    sigma: float,
    kernel: str,
    xi_array: np.ndarray,
) -> dict[int, GroupVelocityProfile]:
    """Compute group velocity profiles for all boundary rows.

    Uses RBF/tension boundary weights from :func:`uniform_boundary_weights_rbf`.

    Parameters
    ----------
    p : int
        Interior half-bandwidth.
    q : int
        Polynomial degree for boundary RBF augmentation.
    nextra : int
        Extra boundary rows/columns.
    nu : int
        Derivative order (1 or 2).
    sigma : float
        RBF shape / tension parameter.
    kernel : str
        RBF kernel type (``"tension"``, ``"gaussian"``, ``"multiquadric"``).
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].

    Returns
    -------
    dict[int, GroupVelocityProfile]
        Keyed by boundary row index (0 to r-1).
    """
    from stencil_gen.phs import uniform_boundary_weights_rbf
    from stencil_gen.temo import compute_dimensions

    dims = compute_dimensions(p, q, 0, nextra, nu)
    r, t = dims.r, dims.t
    nodes = np.arange(t)

    profiles: dict[int, GroupVelocityProfile] = {}
    for i in range(r):
        w = uniform_boundary_weights_rbf(i, t, nu, q, sigma, kernel=kernel)
        w_float = [float(c) for c in w]
        profiles[i] = _build_profile(w_float, nodes - i, xi_array, order=q)

    return profiles


def boundary_group_velocity_classical(
    boundary_rows,
    alpha_values: dict,
    order: int,
    xi_array: np.ndarray,
) -> dict[int, GroupVelocityProfile]:
    """Compute group velocity profiles for classical (non-RBF) boundary rows.

    Takes the symbolic ``BoundaryRow`` list from :func:`derive_boundary` (or the
    conservation-updated rows from :func:`solve_conservation`) and substitutes
    concrete alpha values to obtain numerical stencil coefficients.

    Parameters
    ----------
    boundary_rows : list[BoundaryRow]
        Boundary rows with symbolic coefficients in alpha parameters.
    alpha_values : dict
        Mapping from alpha symbols to numeric values (e.g.,
        ``{alpha_0: -0.77, alpha_1: 0.16}``).
    order : int
        Polynomial accuracy order of the boundary scheme (q = 2*(p+s) - 1
        for the classical Brady & Livescu stencils).
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].

    Returns
    -------
    dict[int, GroupVelocityProfile]
        Keyed by boundary row index.
    """
    t = len(boundary_rows[0].coefficients)
    nodes = np.arange(t)

    profiles: dict[int, GroupVelocityProfile] = {}
    for row in boundary_rows:
        i = row.row_index
        w_float = [float(c.xreplace(alpha_values))
                   if hasattr(c, 'xreplace') else float(c)
                   for c in row.coefficients]
        profiles[i] = _build_profile(w_float, nodes - i, xi_array, order=order)

    return profiles


def cut_cell_group_velocity(
    cut_cell_result,
    psi_sym,
    psi_val: float,
    alpha_values: dict,
    xi_array: np.ndarray,
    order: int | None = None,
) -> dict[int, GroupVelocityProfile]:
    """Compute group velocity profiles for all rows of a cut-cell stencil.

    Evaluates the symbolic psi-dependent stencil coefficients at a specific
    ``psi_val`` and ``alpha_values``, then computes group velocity profiles
    using the non-uniform offsets from the cut-cell grid geometry.

    Parameters
    ----------
    cut_cell_result : CutCellResult
        Precomputed symbolic cut-cell stencil (from ``derive_cut_cell_mathematica``
        or ``derive_cut_cell_scheme``).
    psi_sym : Symbol
        The SymPy symbol for psi used in ``cut_cell_result``.
    psi_val : float
        Numeric psi value in [0, 1].
    alpha_values : dict
        Mapping from alpha symbols to numeric values.
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].
    order : int, optional
        Polynomial accuracy order.  Defaults to the scheme's boundary
        accuracy ``q`` inferred from the dimensions.

    Returns
    -------
    dict[int, GroupVelocityProfile]
        Keyed by row index (0 to R-1) of the floating stencil.
    """
    F = cut_cell_result.floating
    dims = cut_cell_result.dims
    R, T = F.rows, F.cols

    if order is None:
        # dims.r = q + 1 + nextra, but nextra is not recoverable from dims
        # alone, so this overestimates q when nextra > 0.  Callers with
        # nextra > 0 should pass order explicitly.
        order = max(1, dims.r - 1)

    subs = {psi_sym: psi_val, **alpha_values}

    profiles: dict[int, GroupVelocityProfile] = {}
    for i in range(R):
        # Evaluate symbolic coefficients numerically
        w = [float(F[i, j].xreplace(subs)) for j in range(T)]

        # Non-uniform offsets: wall at -(psi_val + i), grid points at j - i
        offsets = [-(psi_val + i)] + [j - i for j in range(T - 1)]

        profiles[i] = _build_profile(w, offsets, xi_array, order=order)

    return profiles


@dataclass
class PsiSweepResult:
    """Results from sweeping psi across the cut-cell parameter range.

    Attributes
    ----------
    psi_values : np.ndarray
        Array of psi values swept.
    profiles : dict[float, dict[int, GroupVelocityProfile]]
        Nested dict: ``{psi_val: {row_index: GroupVelocityProfile}}``.
    worst_row : int
        Row index with the largest group velocity error across all psi values.
    worst_psi : float
        Psi value with the largest group velocity error.
    min_C : float
        Most negative group velocity across all psi values and rows.
    has_sign_reversal : bool
        True if any boundary row has C > 0 at wavenumbers where the interior
        stencil has C < 0 (parasitic energy reversal).
    """

    psi_values: np.ndarray
    profiles: dict[float, dict[int, GroupVelocityProfile]]
    worst_row: int
    worst_psi: float
    min_C: float
    has_sign_reversal: bool


def psi_sweep_group_velocity(
    scheme_params,
    psi_values: np.ndarray,
    alpha_values: dict,
    xi_array: np.ndarray,
    order: int | None = None,
) -> PsiSweepResult:
    """Sweep psi across the cut-cell parameter range, computing group velocity.

    Derives the symbolic cut-cell stencil once from ``scheme_params``, then
    evaluates group velocity profiles at each psi value.  Also computes the
    interior group velocity to detect sign reversals.

    Parameters
    ----------
    scheme_params : SchemeParams
        Scheme parameters (p, q, s, nextra, nu).
    psi_values : np.ndarray
        Array of psi values in [0, 1] to sweep.
    alpha_values : dict
        Mapping from alpha symbols to numeric values.
    xi_array : np.ndarray
        Wavenumber values xi in [0, pi].
    order : int, optional
        Polynomial accuracy order passed to :func:`cut_cell_group_velocity`.

    Returns
    -------
    PsiSweepResult
    """
    from sympy import Symbol

    from stencil_gen.temo import derive_cut_cell_mathematica, derive_cut_cell_scheme

    psi_sym = Symbol("psi")
    # Use singularity-free Mathematica workflow for schemes with zeros;
    # otherwise use the standard pipeline.
    if scheme_params.zeros:
        cc_result = derive_cut_cell_mathematica(scheme_params, psi_sym)
    else:
        cc_result = derive_cut_cell_scheme(scheme_params, psi_sym)

    # Map caller's alpha_values (which may use their own symbols) to the
    # result's alpha_symbols.  If the caller passes an empty dict or uses
    # symbols that already match, this is a no-op.
    if not alpha_values:
        alpha_values = {s: 0 for s in cc_result.alpha_symbols}

    # Compute interior group velocity for sign reversal detection
    interior = interior_group_velocity(
        p=scheme_params.p, nu=scheme_params.nu, xi_array=xi_array,
    )
    C_int = interior.group_velocity

    profiles: dict[float, dict[int, GroupVelocityProfile]] = {}
    worst_row = 0
    worst_psi = float(psi_values[0])
    worst_err = 0.0
    global_min_C = float("inf")
    has_sign_reversal = False

    for pv in psi_values:
        pv_f = float(pv)
        profs = cut_cell_group_velocity(
            cc_result, psi_sym, pv_f, alpha_values, xi_array, order=order,
        )
        profiles[pv_f] = profs

        for row_idx, prof in profs.items():
            C = prof.group_velocity

            # Track worst group velocity error
            max_err = float(np.max(np.abs(prof.gv_error)))
            if max_err > worst_err:
                worst_err = max_err
                worst_row = row_idx
                worst_psi = pv_f

            # Track most negative C
            row_min = float(np.min(C))
            if row_min < global_min_C:
                global_min_C = row_min

            # Detect sign reversal: boundary C > 0 where interior C < 0
            reversal_mask = (C > 0) & (C_int < 0)
            if np.any(reversal_mask):
                has_sign_reversal = True

    return PsiSweepResult(
        psi_values=np.asarray(psi_values),
        profiles=profiles,
        worst_row=worst_row,
        worst_psi=worst_psi,
        min_C=global_min_C,
        has_sign_reversal=has_sign_reversal,
    )


def local_group_velocity(
    weights_func,
    x: np.ndarray,
    xi_array: np.ndarray,
) -> np.ndarray:
    """Compute local group velocity for a varying-coefficient problem.

    For a varying-coefficient problem ``u_t + a(x)*u_x = 0``, the stencil
    coefficients may be x-dependent (e.g., through ``a(x)`` scaling or
    through ``psi(x)`` for cut cells).  At each grid point,
    ``weights_func(x)`` returns the local stencil weights and offsets, and
    the group velocity is computed from those.

    Parameters
    ----------
    weights_func : callable
        ``weights_func(x_val) -> (weights, offsets)`` where *weights* is a
        1-D array of stencil coefficients and *offsets* is a 1-D array of
        (possibly non-integer) grid offsets from the evaluation point.
    x : np.ndarray
        Grid point coordinates, shape ``(N_x,)``.
    xi_array : np.ndarray
        Wavenumber values xi in ``[0, pi]``, shape ``(N_xi,)``.

    Returns
    -------
    np.ndarray, shape ``(N_x, N_xi)``
        Local group velocity ``C[i_x, i_xi]`` at each grid point and
        wavenumber.
    """
    N_x = len(x)
    N_xi = len(xi_array)
    C = np.empty((N_x, N_xi))

    for i in range(N_x):
        weights, offsets = weights_func(x[i])
        C[i, :] = group_velocity_exact_nonuniform(weights, offsets, xi_array)

    return C


@dataclass
class RayTraceResult:
    """Trajectory of a ray traced through a group velocity field.

    Attributes
    ----------
    t : np.ndarray
        Time array, shape ``(N_t,)``.
    x : np.ndarray
        Position trajectory, shape ``(N_t,)``.
    xi : np.ndarray
        Wavenumber trajectory, shape ``(N_t,)``.
    """

    t: np.ndarray
    x: np.ndarray
    xi: np.ndarray


def ray_trace_group_velocity(
    C_field: np.ndarray,
    x_grid: np.ndarray,
    xi_array: np.ndarray,
    xi_0: float,
    x_0: float,
    t_final: float,
    dt: float,
) -> RayTraceResult:
    """Trace a ray through a spatially varying group velocity field.

    Integrates the ray equations:

    - ``dx/dt = C(x, xi)``   (group velocity)
    - ``dxi/dt = -dC/dx``    (refraction; Trefethen 1982, Eq. 4.9b)

    using classical RK4, with bilinear interpolation of the group velocity
    field ``C_field`` (from :func:`local_group_velocity`).

    Parameters
    ----------
    C_field : np.ndarray, shape ``(N_x, N_xi)``
        Local group velocity field ``C[i_x, i_xi]``.
    x_grid : np.ndarray, shape ``(N_x,)``
        Grid point coordinates corresponding to axis 0 of *C_field*.
    xi_array : np.ndarray, shape ``(N_xi,)``
        Wavenumber values corresponding to axis 1 of *C_field*.
    xi_0 : float
        Initial wavenumber.
    x_0 : float
        Initial position.
    t_final : float
        Final integration time.
    dt : float
        Time step.

    Returns
    -------
    RayTraceResult
    """
    from scipy.interpolate import RegularGridInterpolator

    # Build interpolators for C and dC/dx
    interp_C = RegularGridInterpolator(
        (x_grid, xi_array), C_field, method="linear", bounds_error=False,
        fill_value=None,
    )

    # dC/dx via finite differences along the x-axis
    dCdx = np.gradient(C_field, x_grid, axis=0)
    interp_dCdx = RegularGridInterpolator(
        (x_grid, xi_array), dCdx, method="linear", bounds_error=False,
        fill_value=None,
    )

    # RK4 integration
    n_steps = max(1, int(np.ceil(t_final / dt)))
    dt_actual = t_final / n_steps

    t_arr = np.empty(n_steps + 1)
    x_arr = np.empty(n_steps + 1)
    xi_arr = np.empty(n_steps + 1)

    t_arr[0] = 0.0
    x_arr[0] = x_0
    xi_arr[0] = xi_0

    def rhs(x_val, xi_val):
        pt = np.array([[x_val, xi_val]])
        dx_dt = float(interp_C(pt)[0])
        dxi_dt = -float(interp_dCdx(pt)[0])
        return dx_dt, dxi_dt

    for n in range(n_steps):
        x_n, xi_n = x_arr[n], xi_arr[n]

        k1x, k1xi = rhs(x_n, xi_n)
        k2x, k2xi = rhs(x_n + 0.5 * dt_actual * k1x, xi_n + 0.5 * dt_actual * k1xi)
        k3x, k3xi = rhs(x_n + 0.5 * dt_actual * k2x, xi_n + 0.5 * dt_actual * k2xi)
        k4x, k4xi = rhs(x_n + dt_actual * k3x, xi_n + dt_actual * k3xi)

        x_arr[n + 1] = x_n + (dt_actual / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        xi_arr[n + 1] = xi_n + (dt_actual / 6.0) * (k1xi + 2 * k2xi + 2 * k3xi + k4xi)
        t_arr[n + 1] = t_arr[n] + dt_actual

    return RayTraceResult(t=t_arr, x=x_arr, xi=xi_arr)


def gks_group_velocity_check(
    D: np.ndarray,
    xi_array: np.ndarray,
    neutral_tol: float = 0.1,
    localization_tol: float = 0.3,
    side: str = "left",
) -> list[GKSModeInfo]:
    """Identify boundary modes whose group velocity indicates GKS-type instability.

    For the advection equation u_t + u_x = 0 semi-discretized as du/dt = -Du,
    computes eigenvalues and eigenvectors of -D_bc (D with the inflow row/column
    removed).  Identifies boundary-localized, nearly-neutral eigenmodes and
    checks whether the interior stencil's group velocity at each mode's dominant
    wavenumber directs energy from the boundary into the domain — the hallmark
    of GKS instability (Trefethen 1983).

    Parameters
    ----------
    D : np.ndarray
        Full N×N differentiation matrix (approximating d/dx).
    xi_array : np.ndarray
        Wavenumber array in [0, pi] for group velocity evaluation.
    neutral_tol : float
        Fraction of max|Re(lambda)| below which an eigenvalue is considered
        nearly-neutral.  Default 0.1.
    localization_tol : float
        Minimum fraction of eigenvector energy in the boundary region
        required to classify a mode as boundary-localized.  Default 0.3.
    side : str
        Which boundary to analyse.  ``"left"`` (default) removes row/col 0
        (Dirichlet at x=0).  ``"right"`` removes the last row/col (Dirichlet
        at x=L) and flips the outgoing-mode sign convention.  ``"bottom"``
        and ``"top"`` are reserved for 2D operators (deferred to phase 41.6)
        and raise ``NotImplementedError``.

    Returns
    -------
    list[GKSModeInfo]
        One entry per boundary-localized, nearly-neutral eigenmode.
    """
    _valid_sides = ("left", "right", "bottom", "top")
    if side not in _valid_sides:
        raise ValueError(f"side must be one of {_valid_sides}, got {side!r}")
    if side in ("bottom", "top"):
        raise NotImplementedError(
            f"side={side!r} requires a 2D differentiation matrix; "
            "deferred to phase 41.6"
        )

    n = D.shape[0]
    if side == "left":
        D_bc = D[1:, 1:]  # remove inflow row/column (Dirichlet at x=0)
    else:  # side == "right"
        D_bc = D[:-1, :-1]  # remove outflow row/column (Dirichlet at x=L)
    m = D_bc.shape[0]

    eigenvalues, eigenvectors = np.linalg.eig(-D_bc)

    # Nearly-neutral threshold
    max_abs_real = np.max(np.abs(eigenvalues.real))
    threshold = max(neutral_tol * max_abs_real, 1e-10)

    # Interior group velocity profile from D's middle row
    mid = n // 2
    row = D[mid, :]
    cols = np.nonzero(np.abs(row) > 1e-15)[0]
    C_profile = group_velocity_exact(row[cols], mid, cols, xi_array)

    # Boundary region width: ~1/4 of domain, at least 4 points
    bw = max(4, m // 4)

    results: list[GKSModeInfo] = []
    for idx in range(len(eigenvalues)):
        lam = eigenvalues[idx]

        # Skip conjugate duplicates: keep the positive-imaginary member
        if lam.imag < -1e-10:
            continue

        if abs(lam.real) > threshold:
            continue

        v = eigenvectors[:, idx]
        energy = np.abs(v) ** 2
        total = np.sum(energy)
        if total < 1e-30:
            continue

        # Check boundary localization
        left_frac = np.sum(energy[:bw]) / total
        right_frac = np.sum(energy[-bw:]) / total

        if max(left_frac, right_frac) < localization_tol:
            continue  # interior mode, not boundary-localized

        # Use the side where more energy is concentrated
        if left_frac >= right_frac:
            portion = v[:bw]
            mode_side = "left"
        else:
            portion = v[-bw:]
            mode_side = "right"

        # Estimate dominant wavenumber via zero-padded FFT
        pad_len = max(256, 4 * len(portion))
        spec = np.abs(np.fft.fft(portion, n=pad_len))
        nyq = pad_len // 2
        # Skip DC (k=0); search bins 1..nyq for the peak
        peak = int(np.argmax(spec[1 : nyq + 1])) + 1
        dom_xi = float(peak * 2 * np.pi / pad_len)

        # Interpolate interior group velocity at the dominant wavenumber
        C_val = float(np.interp(dom_xi, xi_array, C_profile))

        # Outgoing = energy radiating from boundary into domain interior.
        # Left boundary: rightward (C > 0) enters the domain.
        # Right boundary: leftward (C < 0) enters the domain.
        # When side="right", the sign convention flips: the boundary is at x=L,
        # so C < 0 (leftward into the domain from the right boundary) is outgoing.
        if mode_side == "left":
            is_out = C_val > 0
        else:
            is_out = C_val < 0

        results.append(
            GKSModeInfo(
                eigenvalue=lam,
                boundary_wavenumber=dom_xi,
                group_velocity=C_val,
                is_outgoing=is_out,
            )
        )

    return results


def local_group_velocity_2d_varying(
    interior_stencil_x: tuple[np.ndarray, np.ndarray],
    interior_stencil_y: tuple[np.ndarray, np.ndarray],
    c_x_field: np.ndarray,
    c_y_field: np.ndarray,
    xi_array: np.ndarray,
) -> dict:
    """Compute local group velocity error on a 2D varying-coefficient field.

    For the PDE ``u_t + c_x(x,y) u_x + c_y(x,y) u_y = 0`` discretized with
    tensor-product interior stencils, the local (frozen-coefficient) group
    velocity at grid point ``(i,j)`` is ``(c_x[i,j]*C_x(xi), c_y[i,j]*C_y(xi))``
    where ``C_x`` and ``C_y`` are the 1D interior group velocity profiles.

    For smooth ``(c_x, c_y)`` fields, this local-frozen-coefficient analysis
    is the first-order WKB approximation to the varying-coefficient dispersion
    relation (see e.g. Trefethen 1982, Vichnevetsky & Bowles 1982).

    Parameters
    ----------
    interior_stencil_x : (weights, offsets)
        Interior stencil for the x-direction.
    interior_stencil_y : (weights, offsets)
        Interior stencil for the y-direction.
    c_x_field : np.ndarray, shape (Ny, Nx)
        x-component of the coefficient field.
    c_y_field : np.ndarray, shape (Ny, Nx)
        y-component of the coefficient field.
    xi_array : np.ndarray, shape (N_xi,)
        Wavenumber values in [0, pi].

    Returns
    -------
    dict with keys:
        C_x_field : np.ndarray, shape (Ny, Nx, N_xi)
            Local group velocity in x at each grid point.
        C_y_field : np.ndarray, shape (Ny, Nx, N_xi)
            Local group velocity in y at each grid point.
        gv_error_x_field : np.ndarray, shape (Ny, Nx, N_xi)
            ``c_x[i,j] * gv_error_x(xi)`` — absolute GV error in x.
        gv_error_y_field : np.ndarray, shape (Ny, Nx, N_xi)
            ``c_y[i,j] * gv_error_y(xi)`` — absolute GV error in y.
    """
    w_x, off_x = np.asarray(interior_stencil_x[0]), np.asarray(interior_stencil_x[1])
    w_y, off_y = np.asarray(interior_stencil_y[0]), np.asarray(interior_stencil_y[1])

    # 1D GV profiles (same for all grid points)
    C_x_1d = group_velocity_exact_nonuniform(w_x, off_x, xi_array)  # shape (N_xi,)
    C_y_1d = group_velocity_exact_nonuniform(w_y, off_y, xi_array)

    # 1D GV error: (C - 1) / 1
    gv_err_x_1d = group_velocity_error(C_x_1d)  # shape (N_xi,)
    gv_err_y_1d = group_velocity_error(C_y_1d)

    # Broadcast: c_field[i,j] * profile[xi] → (Ny, Nx, N_xi)
    C_x_out = c_x_field[:, :, np.newaxis] * C_x_1d[np.newaxis, np.newaxis, :]
    C_y_out = c_y_field[:, :, np.newaxis] * C_y_1d[np.newaxis, np.newaxis, :]

    gv_err_x_out = c_x_field[:, :, np.newaxis] * gv_err_x_1d[np.newaxis, np.newaxis, :]
    gv_err_y_out = c_y_field[:, :, np.newaxis] * gv_err_y_1d[np.newaxis, np.newaxis, :]

    return {
        "C_x_field": C_x_out,
        "C_y_field": C_y_out,
        "gv_error_x_field": gv_err_x_out,
        "gv_error_y_field": gv_err_y_out,
    }


def max_local_gv_error_2d(result: dict) -> float:
    """Scalar reduction: max absolute local GV error over all points and wavenumbers.

    Parameters
    ----------
    result : dict
        Output of :func:`local_group_velocity_2d_varying`.

    Returns
    -------
    float
        ``max(|gv_error_x_field|, |gv_error_y_field|)`` over all (i, j, xi).
    """
    max_x = float(np.max(np.abs(result["gv_error_x_field"])))
    max_y = float(np.max(np.abs(result["gv_error_y_field"])))
    return max(max_x, max_y)


# Scheme → half-bandwidth mapping (duplicated from brady2d_stability to avoid
# circular dependency; only E2 and E4 are in scope).
_SCHEME_P = {"E2": 1, "E4": 2}


def anisotropy_over_coefficient_field(
    scheme: str,
    c_x_field: np.ndarray,
    c_y_field: np.ndarray,
    theta_array: np.ndarray,
    xi_mag: float,
) -> dict:
    """Evaluate 2D anisotropy error projected onto a spatially varying advection field.

    For each grid point ``(i, j)`` the local propagation direction is
    ``(c_x[i,j], c_y[i,j]) / |(c_x, c_y)|``.  The scheme's
    :func:`anisotropy_profile` — computed once at wavenumber magnitude
    *xi_mag* — gives the numerical group velocity error as a function of
    propagation angle.  This function interpolates that error at each
    grid point's local angle and returns the maximum over the field.

    The aligned error at a point is the Euclidean norm of the group
    velocity error vector ``|C_numerical - C_exact|`` at the local
    propagation angle, which captures both speed and angular deviations.

    Parameters
    ----------
    scheme : str
        Scheme name (``"E2"`` or ``"E4"``).
    c_x_field, c_y_field : np.ndarray, shape (Ny, Nx)
        Spatially varying advection velocity components.
    theta_array : np.ndarray
        Propagation angles (radians) at which the anisotropy profile
        is evaluated.  Should cover at least the range of local angles
        in ``arctan2(c_y_field, c_x_field)``.
    xi_mag : float
        Wavenumber magnitude in ``[0, pi]``.

    Returns
    -------
    dict
        ``max_aligned_error`` : float
            Maximum ``|C_numerical - C_exact|`` over all grid points.
        ``worst_point`` : tuple[int, int]
            ``(i, j)`` index of the grid point with the largest error.
        ``worst_theta`` : float
            Local propagation angle at the worst point.
    """
    p = _SCHEME_P[scheme]
    anis = anisotropy_profile(p, nu=1, theta_array=theta_array, xi_mag=xi_mag)

    # Error vector magnitude at each theta in theta_array:
    # exact GV = (cos(theta), sin(theta)), numerical = (C_x, C_y)
    err_mag = np.sqrt(
        (anis.C_x - np.cos(anis.theta)) ** 2
        + (anis.C_y - np.sin(anis.theta)) ** 2
    )

    # Local propagation angle at each grid point
    theta_local = np.arctan2(c_y_field, c_x_field)  # shape (Ny, Nx)

    # Interpolate err_mag onto local angles.  np.interp requires sorted
    # xp, so sort theta_array and err_mag together.
    sort_idx = np.argsort(theta_array)
    theta_sorted = theta_array[sort_idx]
    err_sorted = err_mag[sort_idx]
    aligned_error = np.interp(theta_local, theta_sorted, err_sorted)

    # Scalar reduction
    flat_idx = int(np.argmax(aligned_error))
    worst_ij = np.unravel_index(flat_idx, aligned_error.shape)

    return {
        "max_aligned_error": float(np.max(aligned_error)),
        "worst_point": (int(worst_ij[0]), int(worst_ij[1])),
        "worst_theta": float(theta_local[worst_ij]),
    }
