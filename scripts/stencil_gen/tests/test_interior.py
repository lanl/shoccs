"""Tests for interior stencil derivation."""

import pytest
from sympy import Rational as R
from sympy import factorial

from stencil_gen.interior import (
    InteriorCoefficients,
    StencilSpec,
    derive_interior,
    full_delta_array,
    full_gamma_array,
)


# ── Test group 1: StencilSpec properties ──────────────────────────────


def test_spec_order():
    assert StencilSpec(0, 2, 1).order == 4
    assert StencilSpec(1, 1, 1).order == 4
    assert StencilSpec(0, 1, 2).order == 2
    assert StencilSpec(0, 4, 1).order == 8


def test_spec_n_unknowns():
    assert StencilSpec(0, 2, 1).n_unknowns == 2
    assert StencilSpec(1, 1, 1).n_unknowns == 2
    assert StencilSpec(1, 3, 1).n_unknowns == 4


def test_spec_scheme_name():
    assert StencilSpec(0, 2, 1).scheme_name == "E4_1"
    assert StencilSpec(1, 1, 1).scheme_name == "T4_1"
    assert StencilSpec(0, 1, 2).scheme_name == "E2_2"
    assert StencilSpec(0, 4, 1).scheme_name == "E8_1"
    assert StencilSpec(2, 2, 1).scheme_name == "P8_1"


# ── Test group 2: Known coefficient values ────────────────────────────


@pytest.mark.parametrize(
    "s, p, nu, expected_gamma, expected_delta",
    [
        # E2: s=0, p=1, nu=1
        (0, 1, 1, {1: R(1, 2)}, {}),
        # E4: s=0, p=2, nu=1
        (0, 2, 1, {1: R(2, 3), 2: R(-1, 12)}, {}),
        # E6: s=0, p=3, nu=1
        (0, 3, 1, {1: R(3, 4), 2: R(-3, 20), 3: R(1, 60)}, {}),
        # E8: s=0, p=4, nu=1
        (0, 4, 1, {1: R(4, 5), 2: R(-1, 5), 3: R(4, 105), 4: R(-1, 280)}, {}),
        # T4: s=1, p=1, nu=1
        (1, 1, 1, {1: R(3, 4)}, {1: R(1, 4)}),
        # T6: s=1, p=2, nu=1
        (1, 2, 1, {1: R(7, 9), 2: R(1, 36)}, {1: R(1, 3)}),
        # T8: s=1, p=3, nu=1
        (1, 3, 1, {1: R(25, 32), 2: R(1, 20), 3: R(-1, 480)}, {1: R(3, 8)}),
        # E2_2: s=0, p=1, nu=2
        (0, 1, 2, {1: R(1)}, {}),
        # E4_2: s=0, p=2, nu=2
        (0, 2, 2, {1: R(4, 3), 2: R(-1, 12)}, {}),
    ],
    ids=["E2_1", "E4_1", "E6_1", "E8_1", "T4_1", "T6_1", "T8_1", "E2_2", "E4_2"],
)
def test_known_values(s, p, nu, expected_gamma, expected_delta):
    coeffs = derive_interior(s, p, nu)
    assert coeffs.gamma == expected_gamma
    assert coeffs.delta == expected_delta


# ── Test group 3: full_gamma_array / full_delta_array ─────────────────


def test_full_gamma_array_E4_1():
    coeffs = derive_interior(0, 2, 1)
    gamma = full_gamma_array(coeffs)
    assert gamma == [R(1, 12), R(-2, 3), R(0), R(2, 3), R(-1, 12)]


def test_full_gamma_array_E2_2():
    coeffs = derive_interior(0, 1, 2)
    gamma = full_gamma_array(coeffs)
    assert gamma == [R(1), R(-2), R(1)]


def test_full_gamma_array_E4_2():
    coeffs = derive_interior(0, 2, 2)
    gamma = full_gamma_array(coeffs)
    assert gamma == [R(-1, 12), R(4, 3), R(-5, 2), R(4, 3), R(-1, 12)]


def test_full_delta_array_explicit():
    coeffs = derive_interior(0, 2, 1)
    delta = full_delta_array(coeffs)
    assert delta == [R(1)]


def test_full_delta_array_T4():
    coeffs = derive_interior(1, 1, 1)
    delta = full_delta_array(coeffs)
    assert delta == [R(1, 4), R(1), R(1, 4)]


# ── Test group 4: Polynomial exactness ────────────────────────────────


@pytest.mark.parametrize(
    "s, p, nu",
    [
        (0, 1, 1),
        (0, 2, 1),
        (0, 3, 1),
        (0, 4, 1),  # E2, E4, E6, E8
        (1, 1, 1),
        (1, 2, 1),
        (1, 3, 1),  # T4, T6, T8
        (0, 1, 2),
        (0, 2, 2),  # E2_2, E4_2
    ],
    ids=["E2_1", "E4_1", "E6_1", "E8_1", "T4_1", "T6_1", "T8_1", "E2_2", "E4_2"],
)
def test_polynomial_exactness(s, p, nu):
    coeffs = derive_interior(s, p, nu)
    gamma = full_gamma_array(coeffs)
    delta = full_delta_array(coeffs)
    order = 2 * (p + s)
    max_degree = order + nu - 1

    for d in range(0, max_degree + 1):
        # RHS: (1/h^nu) * sum gamma_j * j^d  (h=1)
        rhs = sum(
            gamma[j_idx] * R(j**d)
            for j_idx, j in enumerate(range(-p, p + 1))
        )
        # LHS: sum delta_k * d!/(d-nu)! * k^(d-nu)  if d >= nu, else 0
        if d >= nu:
            exact_factor = R(factorial(d), factorial(d - nu))
            lhs = sum(
                delta[k_idx] * exact_factor * R(k ** (d - nu))
                for k_idx, k in enumerate(range(-s, s + 1))
            )
        else:
            lhs = R(0)
        assert rhs == lhs, f"Failed for degree {d}: RHS={rhs}, LHS={lhs}"

    # Verify NOT exact at degree max_degree + 1 (confirms order is tight)
    d = max_degree + 1
    rhs = sum(
        gamma[j_idx] * R(j**d)
        for j_idx, j in enumerate(range(-p, p + 1))
    )
    if d >= nu:
        exact_factor = R(factorial(d), factorial(d - nu))
        lhs = sum(
            delta[k_idx] * exact_factor * R(k ** (d - nu))
            for k_idx, k in enumerate(range(-s, s + 1))
        )
    else:
        lhs = R(0)
    assert rhs != lhs, f"Stencil is unexpectedly exact at degree {d}"


# ── Test group 6: Input validation ────────────────────────────────────


def test_invalid_negative_s():
    with pytest.raises(ValueError):
        derive_interior(-1, 2, 1)


def test_invalid_zero_p():
    with pytest.raises(ValueError):
        derive_interior(0, 0, 1)


def test_invalid_nu():
    with pytest.raises(ValueError):
        derive_interior(0, 2, 3)
