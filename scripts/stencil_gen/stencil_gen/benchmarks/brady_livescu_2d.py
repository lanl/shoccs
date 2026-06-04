"""Brady & Livescu 2019 §4.3 two-dimensional varying-coefficient benchmark.

Reference problem from:
  Brady, P.T. and Livescu, D. (2019). "High-order, stable, and conservative
  boundary schemes for central and compact finite differences."
  Computers & Fluids, 183, pp. 84-101.

Section 4.3 (pp. 92-94) defines the 2D scalar advection test:

    u_t + grad(psi) . grad(u) = 0    on [0, sqrt(2)]^2, t in [0, 1000]

    psi(x, y) = sqrt((x + 0.25)^2 + (y + 0.25)^2)
    c_x(x, y) = (x + 0.25) / psi(x, y)
    c_y(x, y) = (y + 0.25) / psi(x, y)

    u(x, y, 0) = sin(2*pi*psi(x, y))
    u(0, y, t) = sin(2*pi*(psi(0, y) - t))    (inflow, Dirichlet)
    u(x, 0, t) = sin(2*pi*(psi(x, 0) - t))    (inflow, Dirichlet)

    Exact: u(x, y, t) = sin(2*pi*(psi(x, y) - t))

The coefficient field (c_x, c_y) is a unit radial vector pointing away from
(-0.25, -0.25), giving a smooth, spatially-varying advection velocity that
tests stability and accuracy of boundary closures under non-constant
coefficients.
"""

import math

import numpy as np

L_DOMAIN: float = math.sqrt(2.0)
"""Domain side length: sqrt(2)."""

PSI_OFFSET: float = 0.25
"""Offset in the stream function: psi = sqrt((x + 0.25)^2 + (y + 0.25)^2)."""


def psi(x, y):
    """Stream function psi(x, y) = sqrt((x + 0.25)^2 + (y + 0.25)^2).

    Vectorized: accepts scalar or array inputs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.sqrt((x + PSI_OFFSET) ** 2 + (y + PSI_OFFSET) ** 2)


def c_x(x, y):
    """x-component of the advection velocity: (x + 0.25) / psi(x, y).

    Vectorized: accepts scalar or array inputs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (x + PSI_OFFSET) / psi(x, y)


def c_y(x, y):
    """y-component of the advection velocity: (y + 0.25) / psi(x, y).

    Vectorized: accepts scalar or array inputs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (y + PSI_OFFSET) / psi(x, y)


def exact_solution(x, y, t):
    """Exact solution u(x, y, t) = sin(2*pi*(psi(x, y) - t)).

    Vectorized: accepts scalar or array inputs.
    """
    return np.sin(2.0 * np.pi * (psi(x, y) - t))


def initial_condition(x, y):
    """Initial condition u(x, y, 0) = sin(2*pi*psi(x, y)).

    Equivalent to exact_solution(x, y, 0).
    """
    return exact_solution(x, y, 0.0)


def inflow_bc_x(y, t):
    """Inflow Dirichlet BC at x=0: u(0, y, t) = sin(2*pi*(psi(0, y) - t))."""
    return exact_solution(0.0, y, t)


def inflow_bc_y(x, t):
    """Inflow Dirichlet BC at y=0: u(x, 0, t) = sin(2*pi*(psi(x, 0) - t))."""
    return exact_solution(x, 0.0, t)


def make_coefficient_field(N: int):
    """Build the 2D coefficient field on a uniform N x N grid.

    Returns (x, y, c_x_field, c_y_field) on [0, L_DOMAIN]^2.
    The grid includes all N points in each direction (indices 0..N-1),
    with inflow boundaries at i=0 (x=0) and j=0 (y=0).

    Parameters
    ----------
    N : int
        Number of grid points in each direction.

    Returns
    -------
    x : np.ndarray, shape (N, N)
        x-coordinates on the 2D grid.
    y : np.ndarray, shape (N, N)
        y-coordinates on the 2D grid.
    c_x_field : np.ndarray, shape (N, N)
        x-component of advection velocity at each grid point.
    c_y_field : np.ndarray, shape (N, N)
        y-component of advection velocity at each grid point.
    """
    x1d = np.linspace(0.0, L_DOMAIN, N)
    y1d = np.linspace(0.0, L_DOMAIN, N)
    xg, yg = np.meshgrid(x1d, y1d, indexing="ij")
    return xg, yg, c_x(xg, yg), c_y(xg, yg)
