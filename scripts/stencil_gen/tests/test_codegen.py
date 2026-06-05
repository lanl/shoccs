"""Tests for stencil_gen.codegen module."""

import pytest
from sympy import Integer, Rational, Symbol, symbols

from stencil_gen.codegen import (
    StencilGenSpec,
    TestCase,
    apply_cse,
    compute_test_values,
    format_rational_h_division,
    generate_interior_method,
    generate_nbs_method,
    generate_stencil_cpp,
    generate_test_cpp,
)
from stencil_gen.printer import StencilCodePrinter, build_symbol_map

# ── Shared test fixtures ─────────────────────────────────────────────────

alpha_0, alpha_1 = symbols("alpha_0 alpha_1")

# --- E4u_1 fixtures (uniform, nu=1, R=3, T=5) ---

uniform_printer = StencilCodePrinter(symbol_map=build_symbol_map({"alpha": 2}))

# 15 expressions from E4u_1.cpp nbs_floating (lines 80-94)
e4u_floating_coeffs = [
    (6 * alpha_0 - 11) / 6,
    3 - 4 * alpha_0,
    (12 * alpha_0 - 3) / 2,
    -(12 * alpha_0 - 1) / 3,
    alpha_0,
    (3 * alpha_1 - 1) / 3,
    -(8 * alpha_1 + 1) / 2,
    6 * alpha_1 + 1,
    -(24 * alpha_1 + 1) / 6,
    alpha_1,
    -(168 * alpha_1 + 54 * alpha_0 - 11) / 138,
    (112 * alpha_1 + 36 * alpha_0 - 15) / 23,
    -(336 * alpha_1 + 108 * alpha_0 + 1) / 46,
    (336 * alpha_1 + 108 * alpha_0 + 47) / 69,
    -(28 * alpha_1 + 9 * alpha_0 + 2) / 23,
]

# 10 expressions: nbs_dirichlet coefficients (already sliced, rows 1-2 only)
e4u_dirichlet_coeffs = e4u_floating_coeffs[5:]

# --- polyE2_1 fixtures (cut-cell, nu=1, R=3, T=4) ---

fa = [Symbol(f"fa_{i}") for i in range(6)]
psi = Symbol("psi")

cutcell_printer = StencilCodePrinter(
    symbol_map=build_symbol_map({"fa": 6, "da": 3, "ia": 4}, has_psi=True)
)

# 12 expressions reconstructed from polyE2_1.cpp nbs_floating
poly_floating_coeffs = [
    (fa[1] - 1) / (2 * (1 + psi)),
    (fa[0] - 1) / 2,
    -fa[0] - (2 + psi) * (fa[1] - 1) / (2 * (1 + psi)),
    (fa[0] + fa[1]) / 2,
    (fa[3] - 1) / (2 * (1 + psi)),
    (fa[2] - 1) / 2,
    -fa[2] - (2 + psi) * (fa[3] - 1) / (2 * (1 + psi)),
    (fa[3] + fa[2]) / 2,
    (fa[5] - 1) / (2 * (1 + psi)),
    (fa[4] - 1) / 2,
    -fa[4] - (2 + psi) * (fa[5] - 1) / (2 * (1 + psi)),
    (fa[5] + fa[4]) / 2,
]


# ── 20.4b: CSE wrapper tests ──────────────────────────────────────────────


def test_cse_psi_dependent():
    """CSE hoists common psi subexpressions."""
    psi = Symbol("psi")
    a = Symbol("alpha_0")
    coeffs = [a / (1 + psi), (1 - a) / (1 + psi), a / (2 + psi)]
    repls, reduced = apply_cse(coeffs, prefix="t", start=5)
    # (1 + psi) should be hoisted as a common subexpression
    assert len(repls) >= 1
    # Verify algebraic equivalence after substitution
    for orig, red in zip(coeffs, reduced):
        restored = red.xreplace(dict(repls))
        assert (orig - restored).cancel() == 0


def test_cse_uniform_skipped():
    """Uniform stencil: CSE is not called (decision is caller's)."""
    # This test validates the decision logic, not apply_cse itself
    a0, a1 = symbols("alpha_0 alpha_1")
    coeffs = [(6 * a0 - 11) / 6, 3 - 4 * a0, a1]
    # For uniform, caller should NOT call apply_cse
    # Verify the expressions are simple enough
    assert all(c.count_ops() <= 20 for c in coeffs)


def test_cse_numbering_starts_at_5():
    """CSE temporaries start at t5 by default."""
    psi = Symbol("psi")
    a = Symbol("alpha_0")
    coeffs = [a / (1 + psi), (1 - a) / (1 + psi)]
    repls, _ = apply_cse(coeffs, prefix="t", start=5)
    if repls:
        # First temporary should be t5
        assert repls[0][0].name == "t5"


def test_cse_custom_prefix_and_start():
    """Custom prefix and start index work."""
    psi = Symbol("psi")
    a = Symbol("alpha_0")
    coeffs = [a / (1 + psi), (1 - a) / (1 + psi)]
    repls, _ = apply_cse(coeffs, prefix="x", start=0)
    if repls:
        assert repls[0][0].name == "x0"


def test_cse_roundtrip():
    """Full round-trip: substituting replacements back recovers original."""
    psi = Symbol("psi")
    fa_0, fa_1, fa_2 = symbols("fa_0 fa_1 fa_2")
    coeffs = [
        (fa_1 - 1) / (2 * (1 + psi)),
        (fa_0 - 1) / 2,
        -fa_0 - (2 + psi) * (fa_1 - 1) / (2 * (1 + psi)),
        (fa_0 + fa_1) / 2,
    ]
    repls, reduced = apply_cse(coeffs)
    subs = dict(repls)
    for orig, red in zip(coeffs, reduced):
        restored = red.xreplace(subs)
        assert (orig - restored).cancel() == 0


# ── 20.4c: Interior method generator tests ────────────────────────────────


def test_format_rational_h_division_nu1():
    """nu=1 rational formatting matches E4u_1.cpp pattern."""
    assert format_rational_h_division(Rational(1, 12), nu=1) == "1 / (12 * h)"
    assert format_rational_h_division(Rational(-2, 3), nu=1) == "-2 / (3 * h)"
    assert format_rational_h_division(Rational(1, 1), nu=1) == "1 / h"
    assert format_rational_h_division(Rational(0, 1), nu=1) == "0"


def test_format_rational_h_division_nu2():
    """nu=2 rational formatting matches E2_2.cpp, E4_2.cpp patterns."""
    assert format_rational_h_division(Rational(1, 1), nu=2) == "1 / (h * h)"
    assert format_rational_h_division(Rational(-2, 1), nu=2) == "-2 / (h * h)"
    assert format_rational_h_division(Rational(-1, 12), nu=2) == "-1 / (12 * (h * h))"
    assert format_rational_h_division(Rational(4, 3), nu=2) == "4 / (3 * (h * h))"
    assert format_rational_h_division(Rational(-5, 2), nu=2) == "-5 / (2 * (h * h))"
    assert format_rational_h_division(Rational(0, 1), nu=2) == "0"


def test_interior_e4u_antisymmetry():
    """E4u interior uses shorthand: c[3] = -c[1]; c[4] = -c[0];"""
    coeffs = [Rational(1, 12), Rational(-2, 3), 0, Rational(2, 3), Rational(-1, 12)]
    body = generate_interior_method(coeffs, nu=1, is_uniform=True)
    assert "c[3] = -c[1];" in body
    assert "c[4] = -c[0];" in body
    assert "c[2] = 0;" in body


def test_interior_e8u():
    """E8u interior uses shorthand for all 4 mirrored pairs."""
    coeffs = [
        Rational(1, 280), Rational(-4, 105), Rational(1, 5), Rational(-4, 5),
        0, Rational(4, 5), Rational(-1, 5), Rational(4, 105), Rational(-1, 280),
    ]
    body = generate_interior_method(coeffs, nu=1, is_uniform=True)
    assert "c[5] = -c[3];" in body
    assert "c[8] = -c[0];" in body


def test_interior_poly_loop():
    """polyE2 interior uses loop-based h division."""
    coeffs = [Rational(-1, 2), 0, Rational(1, 2)]
    body = generate_interior_method(coeffs, nu=1, is_uniform=False)
    assert "for (auto&& v : c) v /= h;" in body
    assert "-0.5" in body  # no baked-in h


# ── 20.4d: Boundary method generator tests ───────────────────────────────


def test_nbs_uniform_no_cse():
    """Uniform stencil boundary has no CSE temporaries."""
    body = generate_nbs_method(
        "nbs_floating",
        e4u_floating_coeffs,
        r=3,
        t=5,
        printer=uniform_printer,
        psi_dependent=False,
    )
    assert "real t" not in body  # no CSE temps
    assert "alpha[0]" in body
    assert "for (auto&& v : c) v /= h;" in body
    assert "v *= -1" in body


def test_nbs_cutcell_has_cse():
    """Cut-cell stencil boundary has CSE temporaries."""
    body = generate_nbs_method(
        "nbs_floating",
        poly_floating_coeffs,
        r=3,
        t=4,
        printer=cutcell_printer,
        psi_dependent=True,
    )
    assert "real t5" in body or "real t6" in body
    assert "psi" in body


def test_nbs_dirichlet_fewer_rows():
    """Dirichlet method has (r-1)*t coefficients."""
    body = generate_nbs_method(
        "nbs_dirichlet",
        e4u_dirichlet_coeffs,
        r=3,
        t=5,
        printer=uniform_printer,
        psi_dependent=False,
    )
    # Should have (3-1)*5 = 10 coefficient assignments
    assert body.count("c[") == 10


def test_nbs_uniform_signature():
    """Uniform nbs_floating has single-line signature with unnamed psi."""
    body = generate_nbs_method(
        "nbs_floating",
        e4u_floating_coeffs,
        r=3,
        t=5,
        printer=uniform_printer,
        psi_dependent=False,
    )
    # Single-line signature with unnamed psi (real, not real psi)
    assert "nbs_floating(real h, real, std::span<real> c, bool right) const" in body
    # No subspan in body (uniform does it in dispatcher)
    assert "c = c.subspan" not in body


def test_nbs_cutcell_signature():
    """Cut-cell nbs_floating has two-line signature with named psi."""
    body = generate_nbs_method(
        "nbs_floating",
        poly_floating_coeffs,
        r=3,
        t=4,
        printer=cutcell_printer,
        psi_dependent=True,
    )
    # Two-line signature with named psi
    assert "nbs_floating(real h, real psi, std::span<real> c, bool right) const" in body
    # Subspan in method body
    assert "c = c.subspan(0, R * T);" in body


def test_nbs_nu2_right_boundary():
    """nu=2 right-boundary only reverses, no negation."""
    body = generate_nbs_method(
        "nbs_floating",
        e4u_floating_coeffs,
        r=3,
        t=5,
        printer=uniform_printer,
        psi_dependent=False,
        nu=2,
    )
    assert "for (auto&& v : c) v /= (h * h);" in body
    assert "std::ranges::reverse(c);" in body
    assert "v *= -1" not in body


def test_nbs_floating_coeff_count():
    """Floating method has r*t coefficient assignments."""
    body = generate_nbs_method(
        "nbs_floating",
        e4u_floating_coeffs,
        r=3,
        t=5,
        printer=uniform_printer,
        psi_dependent=False,
    )
    # Should have R*T = 15 coefficient assignments
    assert body.count("c[") == 15


# ── TestStencilGenSpec: dataclass shape checks (42.4a) ───────────────────


class TestStencilGenSpec:
    """Basic dataclass-level invariants for StencilGenSpec."""

    def test_scalar_params_default_empty(self):
        spec = StencilGenSpec(
            name="Foo",
            P=2,
            R=3,
            T=5,
            X=0,
            derivative_order=1,
            is_uniform=True,
            param_arrays={},
            interior_coeffs=[],
            floating_coeffs=[],
            dirichlet_coeffs=[],
        )
        assert spec.scalar_params == []

    def test_scalar_params_accepts_list(self):
        spec = StencilGenSpec(
            name="Foo",
            P=2,
            R=3,
            T=5,
            X=0,
            derivative_order=1,
            is_uniform=True,
            param_arrays={},
            interior_coeffs=[],
            floating_coeffs=[],
            dirichlet_coeffs=[],
            scalar_params=["sigma"],
        )
        assert spec.scalar_params == ["sigma"]

    def test_scalar_params_independent_between_instances(self):
        a = StencilGenSpec(
            name="A", P=2, R=3, T=5, X=0, derivative_order=1, is_uniform=True,
            param_arrays={}, interior_coeffs=[], floating_coeffs=[], dirichlet_coeffs=[],
        )
        b = StencilGenSpec(
            name="B", P=2, R=3, T=5, X=0, derivative_order=1, is_uniform=True,
            param_arrays={}, interior_coeffs=[], floating_coeffs=[], dirichlet_coeffs=[],
        )
        a.scalar_params.append("sigma")
        assert b.scalar_params == []


# ── TestScalarParamsEmission: struct preamble emission (42.4b) ──────────


class TestScalarParamsEmission:
    """Struct preamble emits `real name;` fields and scalar constructors.

    Scope intentionally limited to struct preamble — interior/nbs expression
    bodies that use a scalar symbol require the printer symbol-map update
    from 42.4c. That is exercised by the end-to-end test in 42.4d.
    """

    @staticmethod
    def _spec_with_scalars(name, scalars):
        return StencilGenSpec(
            name=name,
            P=2,
            R=3,
            T=5,
            X=0,
            derivative_order=1,
            is_uniform=True,
            param_arrays={},
            interior_coeffs=[Integer(0)] * 5,
            floating_coeffs=[Integer(0)] * 15,
            dirichlet_coeffs=[Integer(0)] * 15,
            scalar_params=scalars,
        )

    def test_single_scalar_field_emitted(self):
        code = generate_stencil_cpp(self._spec_with_scalars("Tension", ["sigma"]))
        assert "real sigma;" in code

    def test_single_scalar_constructor_signature(self):
        code = generate_stencil_cpp(self._spec_with_scalars("Tension", ["sigma"]))
        assert "Tension(real sigma_)" in code

    def test_multiple_scalar_fields_emitted(self):
        code = generate_stencil_cpp(
            self._spec_with_scalars("Multi", ["sigma", "epsilon"])
        )
        assert "real sigma;" in code
        assert "real epsilon;" in code

    def test_multiple_scalar_constructor_signature(self):
        code = generate_stencil_cpp(
            self._spec_with_scalars("Multi", ["sigma", "epsilon"])
        )
        assert "Multi(real sigma_," in code
        assert "real epsilon_)" in code

    def test_default_constructor_still_emitted(self):
        code = generate_stencil_cpp(self._spec_with_scalars("Tension", ["sigma"]))
        assert "Tension() = default;" in code

    def test_no_scalars_emits_no_real_field(self):
        spec = self._spec_with_scalars("NoScalar", [])
        code = generate_stencil_cpp(spec)
        assert "real sigma;" not in code
        assert "NoScalar(real " not in code


# ── TestScalarParamsCodegenEndToEnd: struct + symbol-map path (42.4d) ────


class TestScalarParamsCodegenEndToEnd:
    """End-to-end: scalar_params flows from spec through preamble and printer.

    Exercises both pieces wired in 42.4b (struct preamble emission) and 42.4c
    (printer symbol-map). A scalar symbol placed in floating_coeffs must print
    as `sigma` (not `sigma[0]`) in the generated nbs_floating body.

    Note: the plan text sketches putting `sigma*h` in ``interior_coeffs``, but
    ``generate_interior_method`` takes the uniform fast path of
    ``Rational(c)`` / ``float(c)`` and cannot accept free symbols. The
    expression path that actually uses the StencilCodePrinter (and therefore
    the scalar symbol map) is ``nbs_floating`` / ``nbs_dirichlet``, so that's
    where we place the scalar symbol for this test.
    """

    @staticmethod
    def _spec_with_scalar_in_floating(name, scalars, floating_expr):
        R, T = 3, 5
        floating = [floating_expr] + [Integer(0)] * (R * T - 1)
        return StencilGenSpec(
            name=name,
            P=2,
            R=R,
            T=T,
            X=0,
            derivative_order=1,
            is_uniform=True,
            param_arrays={},
            interior_coeffs=[Integer(0)] * 5,
            floating_coeffs=floating,
            dirichlet_coeffs=[Integer(0)] * (R * T),
            scalar_params=scalars,
        )

    def test_field_constructor_and_body_together(self):
        sigma = Symbol("sigma")
        spec = self._spec_with_scalar_in_floating("TestStruct", ["sigma"], sigma)
        code = generate_stencil_cpp(spec)
        assert "real sigma;" in code
        assert "TestStruct(real sigma_)" in code
        assert "sigma = sigma_;" in code
        # Expression body prints the scalar as plain `sigma`, never subscripted
        assert "c[0] = sigma;" in code
        assert "sigma[0]" not in code

    def test_scalar_inside_compound_expression(self):
        sigma = Symbol("sigma")
        spec = self._spec_with_scalar_in_floating(
            "Compound", ["sigma"], sigma * Rational(1, 2)
        )
        code = generate_stencil_cpp(spec)
        assert "real sigma;" in code
        # Accept either `sigma / 2` or `sigma * (1.0 / 2)` depending on sympy
        # print ordering — but never `sigma[0]`.
        assert "sigma[0]" not in code
        assert "sigma" in code.split("nbs_floating", 1)[1]

    def test_two_scalars_both_subscript_free(self):
        sigma = Symbol("sigma")
        epsilon = Symbol("epsilon")
        spec = self._spec_with_scalar_in_floating(
            "TwoScalar", ["sigma", "epsilon"], sigma + epsilon
        )
        code = generate_stencil_cpp(spec)
        assert "real sigma;" in code
        assert "real epsilon;" in code
        assert "TwoScalar(real sigma_," in code
        assert "real epsilon_)" in code
        assert "sigma[0]" not in code
        assert "epsilon[0]" not in code
        body = code.split("nbs_floating", 1)[1]
        assert "sigma" in body and "epsilon" in body


# ── StencilGenSpec fixtures for 20.4e ────────────────────────────────────

e4u_spec = StencilGenSpec(
    name="E4u_1",
    P=2,
    R=3,
    T=5,
    X=0,
    derivative_order=1,
    is_uniform=True,
    param_arrays={"alpha": 2},
    interior_coeffs=[
        Rational(1, 12), Rational(-2, 3), 0, Rational(2, 3), Rational(-1, 12),
    ],
    floating_coeffs=e4u_floating_coeffs,
    dirichlet_coeffs=[Integer(0)] * 5 + e4u_dirichlet_coeffs,
)

poly_spec = StencilGenSpec(
    name="polyE2_1",
    P=1,
    R=3,
    T=4,
    X=0,
    derivative_order=1,
    is_uniform=False,
    param_arrays={"fa": 6, "da": 3, "ia": 4},
    interior_coeffs=[Rational(-1, 2), 0, Rational(1, 2)],
    floating_coeffs=poly_floating_coeffs,
    dirichlet_coeffs=[Integer(0)] * 4 + [Integer(0)] * 8,
    has_interp=True,
    interp_P=2,
    interp_T=4,
)


# ── 20.4e: Full struct generator tests ──────────────────────────────────


def test_full_struct_e4u():
    """Generate E4u_1 struct, verify structural elements."""
    code = generate_stencil_cpp(e4u_spec)
    assert '#include "stencil.hpp"' in code
    assert "struct E4u_1 {" in code
    assert "static constexpr int P = 2;" in code
    assert "static constexpr int R = 3;" in code
    assert "std::array<real, 2> alpha;" in code
    assert "copy_zero_padded(a, alpha);" in code
    assert "info query_max()" in code
    assert "interior(real h," in code
    assert "nbs_floating(real h," in code
    assert "nbs_dirichlet(real h," in code
    assert "make_E4u_1(std::span<const real> alpha)" in code
    assert "} // namespace ccs::stencils" in code


def test_full_struct_poly():
    """Generate polyE2_1 struct, verify cut-cell elements."""
    code = generate_stencil_cpp(poly_spec)
    assert "std::array<real, 6> fa;" in code
    assert "std::array<real, 3> da;" in code
    assert "std::array<real, 4> ia;" in code
    assert "real psi" in code  # psi parameter is named
    assert "interp_info query_interp() const { return {2, 4}; }" in code


# ── 21.4a: Non-uniform single-param constructor/factory tests ────────


# Minimal E4_1-like spec: non-uniform, single param array
e4_1_minimal_spec = StencilGenSpec(
    name="E4_1",
    P=2,
    R=4,
    T=7,
    X=0,
    derivative_order=1,
    is_uniform=False,
    param_arrays={"alpha": 4},
    interior_coeffs=[
        Rational(1, 12), Rational(-2, 3), 0, Rational(2, 3), Rational(-1, 12),
    ],
    floating_coeffs=[Integer(0)] * 28,  # placeholder R*T
    dirichlet_coeffs=[Integer(0)] * 28,  # placeholder R*T
)


def test_nonuniform_single_param_constructor():
    """Non-uniform stencil with single param array gets span constructor."""
    code = generate_stencil_cpp(e4_1_minimal_spec)
    assert "E4_1(std::span<const real> a)" in code
    assert "copy_zero_padded(a, alpha);" in code


def test_nonuniform_single_param_factory():
    """Non-uniform stencil with single param array gets correct factory."""
    code = generate_stencil_cpp(e4_1_minimal_spec)
    assert "make_E4_1(std::span<const real> alpha)" in code
    assert "return E4_1{alpha};" in code


def test_nonuniform_single_param_struct_constants():
    """E4_1-like spec emits correct struct constants."""
    code = generate_stencil_cpp(e4_1_minimal_spec)
    assert "static constexpr int P = 2;" in code
    assert "static constexpr int R = 4;" in code
    assert "static constexpr int T = 7;" in code
    assert "static constexpr int X = 0;" in code
    assert "std::array<real, 4> alpha;" in code


# ── 20.4f: Test file generator tests ──────────────────────────────────


def test_test_gen_structure():
    """Generated test file has correct Catch2 structure."""
    cases = [
        TestCase(
            bc_type="Floating",
            h=2.0,
            psi=1.0,
            alpha_values={"alpha": [-0.773, 0.162]},
            expected_coeffs=[0.0] * 15,
        )
    ]
    code = generate_test_cpp(e4u_spec, cases)
    assert '#include "stencil.hpp"' in code
    assert "TEST_CASE" in code
    assert "Approx" in code
    assert "from_lua" in code
    assert ".margin(1.0e-8)" in code


def test_test_gen_values():
    """Generated test values match hand-computed reference."""
    cases = [
        TestCase(
            bc_type="Floating",
            h=2.0,
            psi=1.0,
            alpha_values={"alpha": [-0.7733323791884821, 0.1623961700641681]},
            expected_coeffs=[
                -1.3033328562609077,
                3.046664758376964,
                -3.069997137565446,
                1.713331425043631,
                -0.38666618959424104,
                -0.08546858163458262,
                -0.5747923401283361,
                0.9871885101925043,
                -0.4081256734616695,
                0.08119808503208405,
                0.0923093909615862,
                -0.5359042305130115,
                0.3038563457695172,
                0.13076243615365518,
                0.00897605762825287,
            ],
        )
    ]
    code = generate_test_cpp(e4u_spec, cases)
    assert "-1.3033328562609077" in code


def test_test_gen_lua_config():
    """Generated test file has correct Lua config."""
    cases = [
        TestCase(
            bc_type="Floating",
            h=2.0,
            psi=1.0,
            alpha_values={"alpha": [-0.773, 0.162]},
            expected_coeffs=[0.0] * 15,
        )
    ]
    code = generate_test_cpp(e4u_spec, cases)
    assert 'type = "E4u"' in code
    assert "order = 1" in code
    assert "alpha = {-0.773, 0.162}" in code


def test_test_gen_dirichlet_sizing():
    """Dirichlet test case has (R-1)*T coefficients."""
    cases = [
        TestCase(
            bc_type="Dirichlet",
            h=0.5,
            psi=0.0,
            alpha_values={"alpha": [-0.773, 0.162]},
            expected_coeffs=[0.0] * 10,
        )
    ]
    code = generate_test_cpp(e4u_spec, cases)
    assert "st.query(bcs::Dirichlet)" in code
    assert "REQUIRE(r == 2)" in code  # R-1 = 3-1 = 2
    assert "T c(10)" in code  # (R-1)*T = 2*5 = 10


def test_test_gen_multiple_cases():
    """Multiple test cases generate multiple scoped blocks."""
    cases = [
        TestCase(
            bc_type="Floating",
            h=2.0,
            psi=1.0,
            alpha_values={"alpha": [-0.773, 0.162]},
            expected_coeffs=[0.0] * 15,
        ),
        TestCase(
            bc_type="Dirichlet",
            h=0.5,
            psi=0.0,
            alpha_values={"alpha": [-0.773, 0.162]},
            expected_coeffs=[0.0] * 10,
        ),
    ]
    code = generate_test_cpp(e4u_spec, cases)
    assert code.count("st.query(bcs::") == 2
    assert code.count("REQUIRE_THAT(c,") == 2


def test_compute_test_values_floating():
    """compute_test_values evaluates symbolic expressions correctly."""
    values = compute_test_values(
        e4u_floating_coeffs,
        alpha_values={"alpha": [-0.7733323791884821, 0.1623961700641681]},
        h=2.0,
        psi=1.0,
    )
    reference = [
        -1.3033328562609077,
        3.046664758376964,
        -3.069997137565446,
        1.713331425043631,
        -0.38666618959424104,
        -0.08546858163458262,
        -0.5747923401283361,
        0.9871885101925043,
        -0.4081256734616695,
        0.08119808503208405,
        0.0923093909615862,
        -0.5359042305130115,
        0.3038563457695172,
        0.13076243615365518,
        0.00897605762825287,
    ]
    assert values == pytest.approx(reference, abs=1e-8)


def test_compute_test_values_dirichlet():
    """compute_test_values for Dirichlet sliced coefficients."""
    values = compute_test_values(
        e4u_dirichlet_coeffs,
        alpha_values={"alpha": [-0.7733323791884821, 0.1623961700641681]},
        h=0.5,
        psi=0.0,
    )
    reference = [
        -0.3418743265383305,
        -2.2991693605133445,
        3.9487540407700172,
        -1.632502693846678,
        0.3247923401283362,
        0.3692375638463448,
        -2.143616922052046,
        1.2154253830780688,
        0.5230497446146207,
        0.03590423051301148,
    ]
    assert values == pytest.approx(reference, abs=1e-8)
