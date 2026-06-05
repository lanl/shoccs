"""Group velocity objectives for the sweep optimization pipeline.

Thin scalar wrappers around the :mod:`stencil_gen.group_velocity` API that
return single numbers suitable for use as secondary objectives in sweep
scripts.

Contract: these helpers are *secondary* objectives only.  Stability
(``stability_eigenvalue < STABILITY_TOL``) is always the hard feasibility
gate; GV error is minimized only among the feasible set.  This mirrors the
feasible-then-minimize pattern already used in ``tension_penalty_sweep.py``.
"""

from __future__ import annotations

import numpy as np

from stencil_gen.group_velocity import (
    boundary_group_velocity,
    gks_group_velocity_check,
    group_velocity_error,
    group_velocity_exact_nonuniform,
    interior_group_velocity,
    psi_sweep_group_velocity,
)


def _default_xi(n_xi: int) -> np.ndarray:
    """Dense xi grid on (0, pi] used by all GV objectives."""
    return np.linspace(1e-6, np.pi, n_xi)


def interior_gv_error_max(p: int, nu: int = 1, n_xi: int = 200) -> float:
    """Worst-case interior group velocity error for the explicit (s=0) scheme."""
    xi = _default_xi(n_xi)
    profile = interior_group_velocity(p=p, nu=nu, xi_array=xi)
    return float(np.max(np.abs(profile.gv_error)))


def interior_cutoff_fraction(p: int, nu: int = 1, n_xi: int = 200) -> float:
    """Interior cutoff xi as a fraction of pi (1.0 = ideal, lower = earlier onset)."""
    xi = _default_xi(n_xi)
    profile = interior_group_velocity(p=p, nu=nu, xi_array=xi)
    return float(profile.cutoff_xi / np.pi)


def boundary_gv_error_max(
    p: int,
    q: int,
    nextra: int,
    nu: int,
    sigma: float,
    kernel: str,
    n_xi: int = 200,
) -> float:
    """Worst-case GV error across all RBF boundary rows at a given sigma/kernel."""
    xi = _default_xi(n_xi)
    profiles = boundary_group_velocity(
        p=p, q=q, nextra=nextra, nu=nu, sigma=sigma, kernel=kernel, xi_array=xi,
    )
    return float(max(np.max(np.abs(prof.gv_error)) for prof in profiles.values()))


def cutcell_gv_min_C(
    scheme_params,
    psi_values: np.ndarray,
    alpha_values: dict,
    n_xi: int = 200,
) -> tuple[float, bool]:
    """Most negative cut-cell group velocity and sign-reversal flag (parasitic)."""
    xi = _default_xi(n_xi)
    result = psi_sweep_group_velocity(
        scheme_params=scheme_params,
        psi_values=np.asarray(psi_values),
        alpha_values=alpha_values,
        xi_array=xi,
    )
    return float(result.min_C), bool(result.has_sign_reversal)


def gv_score_from_matrix(D: np.ndarray, n_xi: int = 200) -> dict:
    """GV summary of a precomputed differentiation matrix without rebuilding it.

    Scans leading rows whose leftmost nonzero is column 0 (boundary rows) and
    computes the per-row group velocity profile from the row's own coefficients.
    Returns ``{"max_gv_error": float, "min_cutoff_xi": float}`` aggregated over
    those rows.
    """
    xi = _default_xi(n_xi)
    n = D.shape[0]
    nz_tol = 1e-15

    max_err = 0.0
    min_cutoff = float(np.pi)
    any_row = False
    for i in range(n):
        row = D[i, :]
        nz_cols = np.nonzero(np.abs(row) > nz_tol)[0]
        if nz_cols.size == 0 or nz_cols[0] != 0:
            break
        any_row = True
        weights = row[nz_cols]
        offsets = nz_cols.astype(float) - float(i)
        C = group_velocity_exact_nonuniform(weights, offsets, xi)
        gv_err = group_velocity_error(C)
        max_err = max(max_err, float(np.max(np.abs(gv_err))))

        last_positive_idx = 0
        for idx in range(1, len(xi)):
            if C[idx] > 0.0:
                last_positive_idx = idx
        if last_positive_idx + 1 < len(xi):
            cutoff = float(xi[last_positive_idx + 1])
        else:
            cutoff = float(xi[-1])
        min_cutoff = min(min_cutoff, cutoff)

    if not any_row:
        return {"max_gv_error": float("nan"), "min_cutoff_xi": float("nan")}
    return {"max_gv_error": max_err, "min_cutoff_xi": min_cutoff}


def print_gks_advisory(
    D: np.ndarray,
    *,
    label: str,
    n_xi: int = 200,
) -> int:
    """Run ``gks_group_velocity_check`` on ``D`` and print outgoing-mode warnings.

    Advisory only — never mutates caller state.  Prints a single ``no outgoing
    boundary modes`` line when clean, or one ``WARNING`` line per outgoing mode.
    Returns the count of outgoing modes for callers that want to assert on it.
    """
    xi = _default_xi(n_xi)
    modes = gks_group_velocity_check(D, xi)
    outgoing = [m for m in modes if m.is_outgoing]
    print(f"\n  GKS advisory ({label}):")
    if not outgoing:
        print(
            f"    no outgoing boundary modes detected "
            f"({len(modes)} boundary mode(s) inspected)"
        )
        return 0
    for m in outgoing:
        print(
            f"    WARNING: outgoing boundary mode at xi={m.boundary_wavenumber:.4f}, "
            f"lambda={m.eigenvalue.real:+.3e}{m.eigenvalue.imag:+.3e}j, "
            f"C={m.group_velocity:+.4e}"
        )
    return len(outgoing)
