"""Shared fixtures for stencil_gen tests."""

import pytest
from math import factorial


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def _check_taylor_accuracy(B_u, q, nu):
    """Assert each row of B_u satisfies Taylor moment conditions.

    For a nu-th derivative operator accurate to order q, each row i must satisfy:
        sum_j c_j * (j - i)^m = m! * delta_{m, nu}   for m = 0, ..., max(q, nu).
    """
    from sympy import simplify

    r, t = B_u.shape
    n_moments = max(q + 1, nu + 1)
    for i in range(r):
        for m in range(n_moments):
            moment = sum(B_u[i, j] * (j - i) ** m for j in range(t))
            expected = factorial(nu) if m == nu else 0
            assert simplify(moment - expected) == 0, (
                f"Row {i}, moment {m}: got {simplify(moment)}, expected {expected}"
            )


@pytest.fixture(scope="session")
def assert_taylor_accuracy():
    """Fixture providing Taylor accuracy assertion for uniform B_u matrices."""
    return _check_taylor_accuracy


def run_pipeline(p, nu=1, s=0):
    """Run derive_boundary + conservation, return full pipeline results."""
    from stencil_gen.boundary import derive_boundary
    from stencil_gen.conservation import build_conservation_system, solve_conservation

    result = derive_boundary(p=p, nu=nu, s=s)
    equations, w_syms, last_free = build_conservation_system(
        result.r, result.t, p, result.rows, result.interior_coeffs
    )
    solution_dict, updated_rows = solve_conservation(
        equations, w_syms, last_free, result.all_free_params, result.rows
    )
    return updated_rows, solution_dict, w_syms, result


@pytest.fixture(scope="module")
def e2_1_uniform():
    """Cache derive_e2_uniform_boundary(nu=1) once per module."""
    from stencil_gen.temo import derive_e2_uniform_boundary

    return derive_e2_uniform_boundary(nu=1)


@pytest.fixture(scope="module")
def e2_2_uniform():
    """Cache derive_e2_uniform_boundary(nu=2) once per module."""
    from stencil_gen.temo import derive_e2_uniform_boundary

    return derive_e2_uniform_boundary(nu=2)


@pytest.fixture(scope="module")
def e4u_pipeline():
    """Run E4u pipeline once per module."""
    return run_pipeline(p=2)


@pytest.fixture(scope="module")
def e6u_pipeline():
    """Run E6u pipeline once per module."""
    return run_pipeline(p=3)


@pytest.fixture(scope="module")
def e8u_pipeline():
    """Run E8u pipeline once per module."""
    return run_pipeline(p=4)
