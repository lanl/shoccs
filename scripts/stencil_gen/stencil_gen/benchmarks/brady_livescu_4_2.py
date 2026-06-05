"""Brady & Livescu 2019 §4.2 reflecting-hyperbolic eigenvalue benchmark.

Reference problem from:
  Brady, P.T. and Livescu, D. (2019). "High-order, stable, and conservative
  boundary schemes for central and compact finite differences."
  Computers & Fluids, 183, pp. 84-101.

Section 4.2 (pp. 91-92) defines the 1D coupled hyperbolic system:

    u_t = v_x
    v_t = u_x          on [0, 1],  t in [0, 500]

    u(0, t) = 0         (Dirichlet, reflecting)
    v(1, t) = 0         (Dirichlet, reflecting)

    u(x, 0) = -(3*pi/2) * sin(3*pi*x/2)
    v(x, 0) = 0

The system is a linearised wave equation with constant unit wave speeds
(+1 and -1).  The reflecting BCs are energy-conserving, so the continuous
operator has a purely imaginary spectrum: eigenvalues lambda_k = +/- i*w_k
where w_k = (2k - 1)*pi/2 for positive integer k.

Exact solution (standing wave from d'Alembert decomposition, Eqs. 50-51):

    u(x, t) = -(3*pi/2) * sin(3*pi*x/2) * cos(3*pi*t/2)
    v(x, t) = -(3*pi/2) * cos(3*pi*x/2) * sin(3*pi*t/2)

The initial condition excites a single eigenmode (k=2, w = 3*pi/2).
"""

import numpy as np

L_DOMAIN: float = 1.0
"""Domain length: [0, 1]."""


def initial_u(x: np.ndarray) -> np.ndarray:
    """Initial condition u(x, 0) = -(3*pi/2) * sin(3*pi*x/2)."""
    x = np.asarray(x, dtype=float)
    return -1.5 * np.pi * np.sin(1.5 * np.pi * x)


def initial_v(x: np.ndarray) -> np.ndarray:
    """Initial condition v(x, 0) = 0."""
    x = np.asarray(x, dtype=float)
    return np.zeros_like(x)


def exact_solution(x: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact solution (u, v) at time t (Eqs. 50-51 of Brady & Livescu 2019).

    Standing-wave superposition from d'Alembert decomposition:
        u(x, t) = -(3*pi/2) * sin(3*pi*x/2) * cos(3*pi*t/2)
        v(x, t) = -(3*pi/2) * cos(3*pi*x/2) * sin(3*pi*t/2)

    Vectorized: accepts scalar or array x.
    """
    x = np.asarray(x, dtype=float)
    omega = 1.5 * np.pi
    amp = -1.5 * np.pi
    u = amp * np.sin(omega * x) * np.cos(omega * t)
    v = amp * np.cos(omega * x) * np.sin(omega * t)
    return u, v


def continuous_eigenvalues(k_max: int = 20) -> np.ndarray:
    """Continuous eigenvalues of the BL 4.2 operator with reflecting BCs.

    The eigenproblem on [0, 1] with u(0) = 0, v(1) = 0 yields:
        lambda_k = +/- i * (2k - 1) * pi / 2,   k = 1, 2, ..., k_max

    Returns an array of 2*k_max complex eigenvalues (dtype complex128).
    """
    ks = np.arange(1, k_max + 1)
    omega = (2 * ks - 1) * np.pi / 2.0
    return np.concatenate([1j * omega, -1j * omega])
