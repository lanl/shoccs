"""Shared utility functions for the stencil_gen package."""

from sympy import cancel, linsolve


def solve_linear(A, b, unknowns):
    """Solve A @ unknowns = b via linsolve and return {symbol: value}.

    Each value is passed through ``cancel()`` to simplify rational expressions.
    """
    sol_set = linsolve((A, b), *unknowns)
    sol_tuple = list(sol_set)[0]
    return {sym: cancel(val) for sym, val in zip(unknowns, sol_tuple)}
