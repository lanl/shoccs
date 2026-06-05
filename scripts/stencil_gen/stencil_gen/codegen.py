"""C++ code generation for stencil files.

Generates .cpp and .t.cpp files matching the patterns in src/stencils/.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sympy import Expr, Rational, Symbol, cse, numbered_symbols

from stencil_gen.printer import StencilCodePrinter, build_symbol_map


def apply_cse(
    coeffs: list[Expr],
    prefix: str = "t",
    start: int = 5,
) -> tuple[list[tuple[Symbol, Expr]], list[Expr]]:
    """Apply common subexpression elimination with project conventions.

    The start=5 default matches the existing naming convention in polyE2_1.cpp
    where CSE temporaries begin at t5.

    Args:
        coeffs: List of symbolic expressions to optimize.
        prefix: Prefix for generated temporary symbols.
        start: Starting index for temporary numbering.

    Returns:
        Tuple of (replacements, reduced) where replacements is a list of
        (symbol, expression) pairs and reduced is the simplified expression list.
    """
    replacements, reduced = cse(
        coeffs,
        symbols=numbered_symbols(prefix, start=start),
    )
    return replacements, reduced


def format_rational_h_division(r: Rational, nu: int) -> str:
    """Format a rational coefficient with baked-in h-division for interior methods.

    Returns a C++ expression string. Not using StencilCodePrinter because these
    are exact Rational constants, not general symbolic expressions.
    """
    if r == 0:
        return "0"
    p, q = int(r.p), int(r.q)
    h_div = "h" if nu == 1 else "(h * h)"
    if q == 1:
        return f"{p} / {h_div}"
    return f"{p} / ({q} * {h_div})"


def generate_interior_method(
    coeffs: list,
    nu: int,
    is_uniform: bool,
) -> str:
    """Generate the body of the interior() method.

    Returns the lines between { and } of the interior method, each indented
    with 8 spaces (matching 2-level indent inside struct + method).

    Args:
        coeffs: Length 2P+1 exact Rational coefficients from interior derivation.
        nu: Derivative order (1 or 2).
        is_uniform: True for uniform stencils (bake h into expressions).
    """
    P = (len(coeffs) - 1) // 2
    lines: list[str] = []
    indent = "        "

    if is_uniform:
        # Verify symmetry/antisymmetry
        if nu == 1:
            # Antisymmetric: c[2P-k] = -c[k], c[P] = 0
            use_shorthand = all(
                coeffs[2 * P - k] == -coeffs[k] for k in range(P)
            ) and coeffs[P] == 0
        else:
            # Symmetric: c[2P-k] = c[k]
            use_shorthand = all(
                coeffs[2 * P - k] == coeffs[k] for k in range(P)
            )

        # Emit first P+1 coefficients with baked-in h
        for i in range(P + 1):
            c = coeffs[i]
            if c == 0:
                lines.append(f"{indent}c[{i}] = 0;")
            else:
                lines.append(f"{indent}c[{i}] = {format_rational_h_division(Rational(c), nu)};")

        # Emit shorthand tail or explicit assignments
        if use_shorthand:
            sign = "-" if nu == 1 else ""
            for i in range(P + 1, 2 * P + 1):
                mirror = 2 * P - i
                lines.append(f"{indent}c[{i}] = {sign}c[{mirror}];")
        else:
            # Fallback: explicit assignments for remaining coefficients
            for i in range(P + 1, 2 * P + 1):
                c = coeffs[i]
                if c == 0:
                    lines.append(f"{indent}c[{i}] = 0;")
                else:
                    lines.append(f"{indent}c[{i}] = {format_rational_h_division(Rational(c), nu)};")

        lines.append("")
        lines.append(f"{indent}return c.subspan(0, 2 * P + 1);")
    else:
        # Cut-cell: loop-based h division
        lines.append(f"{indent}c = c.subspan(0, 2 * {P} + 1);")
        for i in range(2 * P + 1):
            c = coeffs[i]
            if c == 0:
                lines.append(f"{indent}c[{i}] = 0;")
            else:
                # Print as float literal with 17 significant digits
                val = float(c)
                # Use repr-style formatting but clean up
                formatted = f"{val:.17g}"
                lines.append(f"{indent}c[{i}] = {formatted};")

        if nu == 1:
            lines.append(f"{indent}for (auto&& v : c) v /= h;")
        else:
            lines.append(f"{indent}for (auto&& v : c) v /= (h * h);")
        lines.append(f"{indent}return c;")

    return "\n".join(lines)


def generate_nbs_method(
    method_name: str,
    coeffs: list[Expr],
    r: int,
    t: int,
    printer: StencilCodePrinter,
    psi_dependent: bool,
    nu: int = 1,
) -> str:
    """Generate a complete nbs_floating or nbs_dirichlet method.

    Returns the complete method including signature and braces, indented with
    4 spaces (one level inside the struct). Body lines use 8 spaces.

    Args:
        method_name: "nbs_floating" or "nbs_dirichlet".
        coeffs: r*t (floating) or (r-1)*t (Dirichlet) symbolic expressions.
        r: Number of boundary rows.
        t: Boundary stencil width.
        printer: Configured StencilCodePrinter instance.
        psi_dependent: True for cut-cell stencils (psi is named, CSE applied).
        nu: Derivative order (1 or 2).
    """
    indent4 = "    "
    indent8 = "        "

    # Step 1: Determine subspan size
    if method_name == "nbs_floating":
        span_expr = "R * T"
    else:
        span_expr = "(R - 1) * T"

    # Step 2: Optionally apply CSE
    if psi_dependent:
        replacements, reduced = apply_cse(coeffs)
    else:
        replacements = []
        reduced = coeffs

    # Step 3: Emit method signature
    lines: list[str] = []
    if psi_dependent:
        # Cut-cell: two-line signature, named psi parameter
        lines.append(f"{indent4}std::span<const real>")
        lines.append(f"{indent4}{method_name}(real h, real psi, std::span<real> c, bool right) const")
    else:
        # Uniform: single-line signature, unnamed psi parameter
        lines.append(f"{indent4}std::span<const real> {method_name}(real h, real, std::span<real> c, bool right) const")
    lines.append(f"{indent4}{{")

    # For cut-cell, emit subspan inside the method body
    if psi_dependent:
        lines.append(f"{indent8}c = c.subspan(0, {span_expr});")

    # Step 4: Emit CSE temporaries (if any)
    if replacements:
        lines.append("")
        for sym, expr in replacements:
            lines.append(f"{indent8}real {sym.name} = {printer.doprint(expr)};")

    # Blank line between temporaries and coefficients
    if replacements:
        lines.append("")

    # Step 5: Emit coefficient assignments
    for i, expr in enumerate(reduced):
        lines.append(f"{indent8}c[{i}] = {printer.doprint(expr)};")

    # Step 6: Emit h-division loop
    lines.append("")
    if nu == 1:
        lines.append(f"{indent8}for (auto&& v : c) v /= h;")
    else:
        lines.append(f"{indent8}for (auto&& v : c) v /= (h * h);")

    # Step 7: Emit right-boundary logic
    if nu == 1:
        lines.append(f"{indent8}if (right) {{")
        lines.append(f"{indent8}    for (auto&& v : c) v *= -1;")
        lines.append(f"{indent8}    std::ranges::reverse(c);")
        lines.append(f"{indent8}}}")
    else:
        lines.append(f"{indent8}if (right) {{")
        lines.append(f"{indent8}    std::ranges::reverse(c);")
        lines.append(f"{indent8}}}")

    # Step 8: Return statement
    lines.append("")
    lines.append(f"{indent8}return c;")
    lines.append(f"{indent4}}}")

    return "\n".join(lines)


# ── 20.4e: Full struct generator ─────────────────────────────────────────


@dataclass
class StencilGenSpec:
    """Specification for generating a complete stencil .cpp file.

    Runtime parameters come in two shapes:

    - ``param_arrays`` maps an array parameter name to its length (e.g.
      ``{"alpha": 2}`` emits ``std::array<real, 2> alpha;`` and references to
      ``alpha[0]``/``alpha[1]`` in expression bodies).
    - ``scalar_params`` lists plain scalar parameter names (e.g. ``["sigma"]``
      emits ``real sigma;`` and references to ``sigma`` — no subscript — in
      expression bodies). Used by spline families whose boundary closure takes
      a single tension/shape parameter from Lua.
    """

    name: str
    P: int
    R: int
    T: int
    X: int
    derivative_order: int
    is_uniform: bool
    param_arrays: dict[str, int]
    interior_coeffs: list
    floating_coeffs: list
    dirichlet_coeffs: list
    has_interp: bool = False
    interp_P: int = 0
    interp_T: int = 0
    scalar_params: list[str] = field(default_factory=list)


def _emit_header(spec: StencilGenSpec) -> str:
    """Emit includes, optional comment, and namespace open."""
    lines = [
        '#include "stencil.hpp"',
        "",
        "#include <algorithm>",
        "",
        "#include <cmath>",
        "",
    ]
    if spec.is_uniform:
        family = spec.name.rsplit("_", 1)[0]
        lines.append(f"/// {family} - uniform mesh {family} stencil")
        lines.append("")
    lines.append("namespace ccs::stencils")
    lines.append("{")
    return "\n".join(lines) + "\n"


def _emit_struct_preamble(spec: StencilGenSpec) -> str:
    """Emit struct open, constants, member arrays, and constructors."""
    indent = "    "
    lines = [
        f"struct {spec.name} {{",
        "",
        f"{indent}static constexpr int P = {spec.P};",
        f"{indent}static constexpr int R = {spec.R};",
        f"{indent}static constexpr int T = {spec.T};",
        f"{indent}static constexpr int X = {spec.X};",
    ]

    if spec.param_arrays:
        lines.append("")
        for name, count in spec.param_arrays.items():
            lines.append(f"{indent}std::array<real, {count}> {name};")

    if spec.scalar_params:
        lines.append("")
        for name in spec.scalar_params:
            lines.append(f"{indent}real {name};")

    lines.append("")
    lines.append(f"{indent}{spec.name}() = default;")

    if len(spec.param_arrays) == 1:
        array_name = list(spec.param_arrays.keys())[0]
        lines.append(f"{indent}{spec.name}(std::span<const real> a)")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}    copy_zero_padded(a, {array_name});")
        lines.append(f"{indent}}}")
    elif len(spec.param_arrays) > 1:
        names = list(spec.param_arrays.keys())
        ctor_indent = " " * (4 + len(spec.name) + 1)
        lines.append(f"{indent}{spec.name}(std::span<const real> {names[0]}_,")
        for n in names[1:-1]:
            lines.append(f"{ctor_indent}std::span<const real> {n}_,")
        lines.append(f"{ctor_indent}std::span<const real> {names[-1]}_)")
        lines.append(f"{indent}{{")
        for n in names:
            lines.append(f"{indent}    copy_zero_padded({n}_, {n});")
        lines.append(f"{indent}}}")

    if spec.scalar_params:
        scalar_names = spec.scalar_params
        ctor_indent = " " * (4 + len(spec.name) + 1)
        if len(scalar_names) == 1:
            n = scalar_names[0]
            lines.append(f"{indent}{spec.name}(real {n}_)")
            lines.append(f"{indent}{{")
            lines.append(f"{indent}    {n} = {n}_;")
            lines.append(f"{indent}}}")
        else:
            lines.append(f"{indent}{spec.name}(real {scalar_names[0]}_,")
            for n in scalar_names[1:-1]:
                lines.append(f"{ctor_indent}real {n}_,")
            lines.append(f"{ctor_indent}real {scalar_names[-1]}_)")
            lines.append(f"{indent}{{")
            for n in scalar_names:
                lines.append(f"{indent}    {n} = {n}_;")
            lines.append(f"{indent}}}")

    return "\n".join(lines) + "\n"


def _emit_query_methods(spec: StencilGenSpec) -> str:
    """Emit query_max, query, query_interp, and interpolation stubs."""
    indent = "    "
    indent8 = "        "
    lines = [
        "",
        f"{indent}info query_max() const {{ return {{P, R, T, X}}; }}",
        f"{indent}info query(bcs::type b) const",
        f"{indent}{{",
        f"{indent8}switch (b) {{",
        f"{indent8}case bcs::Dirichlet:",
        f"{indent8}    return {{P, R - 1, T, 0}};",
        f"{indent8}case bcs::Floating:",
        f"{indent8}    return {{P, R, T, 0}};",
        f"{indent8}case bcs::Neumann:",
        f"{indent8}    return {{}};",
        f"{indent8}default:",
        f"{indent8}    return {{}};",
        f"{indent8}}}",
        f"{indent}}}",
    ]

    if spec.has_interp:
        lines.append(
            f"{indent}interp_info query_interp() const "
            f"{{ return {{{spec.interp_P}, {spec.interp_T}}}; }}"
        )
    else:
        lines.append(f"{indent}interp_info query_interp() const {{ return {{}}; }}")

    lines.extend(
        [
            "",
            f"{indent}std::span<const real> interp_interior(real, std::span<real> c) const {{ return c; }}",
            "",
            f"{indent}std::span<const real> interp_wall(int, real, real, std::span<real> c, bool) const",
            f"{indent}{{",
            f"{indent8}return c;",
            f"{indent}}}",
        ]
    )

    return "\n".join(lines) + "\n"


def _emit_interior_method(spec: StencilGenSpec) -> str:
    """Emit the interior() method, wrapping generate_interior_method."""
    indent = "    "
    body = generate_interior_method(
        spec.interior_coeffs, spec.derivative_order, spec.is_uniform
    )
    lines = [
        "",
        f"{indent}std::span<const real> interior(real h, std::span<real> c) const",
        f"{indent}{{",
        body,
        f"{indent}}}",
    ]
    return "\n".join(lines) + "\n"


def _emit_nbs_dispatcher(spec: StencilGenSpec) -> str:
    """Emit the nbs() switch dispatcher method."""
    indent = "    "
    indent8 = "        "
    # "    std::span<const real> nbs(" = 30 chars
    cont = " " * 30

    x_param = "std::span<real> x" if spec.is_uniform else "std::span<real>"

    lines = [
        "",
        f"{indent}std::span<const real> nbs(real h,",
        f"{cont}bcs::type b,",
        f"{cont}real psi,",
        f"{cont}bool right,",
        f"{cont}std::span<real> c,",
        f"{cont}{x_param}) const",
        f"{indent}{{",
        f"{indent8}switch (b) {{",
    ]

    if spec.is_uniform:
        lines.extend(
            [
                f"{indent8}case bcs::Floating:",
                f"{indent8}    return nbs_floating(h, psi, c.subspan(0, R * T), right);",
                f"{indent8}case bcs::Dirichlet:",
                f"{indent8}    return nbs_dirichlet(h, psi, c.subspan(0, (R - 1) * T), right);",
            ]
        )
    else:
        lines.extend(
            [
                f"{indent8}case bcs::Floating:",
                f"{indent8}    return nbs_floating(h, psi, c, right);",
                f"{indent8}case bcs::Dirichlet:",
                f"{indent8}    return nbs_dirichlet(h, psi, c, right);",
            ]
        )

    lines.extend(
        [
            f"{indent8}default:",
            f"{indent8}    return c;",
            f"{indent8}}}",
            f"{indent}}}",
        ]
    )

    return "\n".join(lines) + "\n"


def _emit_nbs_methods(spec: StencilGenSpec, printer: StencilCodePrinter) -> str:
    """Emit nbs_floating, nbs_dirichlet, and nbs_neumann methods."""
    indent = "    "
    indent8 = "        "
    parts: list[str] = []

    # nbs_floating
    floating_method = generate_nbs_method(
        "nbs_floating",
        spec.floating_coeffs,
        spec.R,
        spec.T,
        printer,
        psi_dependent=not spec.is_uniform,
        nu=spec.derivative_order,
    )
    parts.append("")
    parts.append(floating_method)

    # nbs_dirichlet — skip row 0
    dirichlet_coeffs = spec.dirichlet_coeffs[spec.T :]
    dirichlet_method = generate_nbs_method(
        "nbs_dirichlet",
        dirichlet_coeffs,
        spec.R,
        spec.T,
        printer,
        psi_dependent=not spec.is_uniform,
        nu=spec.derivative_order,
    )
    parts.append("")
    parts.append(dirichlet_method)

    # nbs_neumann stub
    parts.append("")
    if spec.is_uniform and spec.X == 0:
        parts.append(
            f"{indent}void nbs_neumann(real, real, std::span<real>, std::span<real>, bool) const {{}}"
        )
    else:
        parts.append(f"{indent}std::span<const real>")
        parts.append(
            f"{indent}nbs_neumann(real, real, std::span<real> c, std::span<real>, bool) const"
        )
        parts.append(f"{indent}{{")
        parts.append(f"{indent8}return c;")
        parts.append(f"{indent}}}")

    return "\n".join(parts) + "\n"


def _emit_factory(spec: StencilGenSpec) -> str:
    """Emit struct close, factory function, and namespace close."""
    lines = ["};", ""]

    if not spec.param_arrays:
        lines.append(
            f"stencil make_{spec.name}() {{ return {spec.name}{{}}; }}"
        )
    elif len(spec.param_arrays) == 1:
        array_name = list(spec.param_arrays.keys())[0]
        lines.append(
            f"stencil make_{spec.name}(std::span<const real> {array_name})"
            f" {{ return {spec.name}{{{array_name}}}; }}"
        )
    else:
        names = list(spec.param_arrays.keys())
        prefix = f"stencil make_{spec.name}("
        cont = " " * len(prefix)
        lines.append(f"{prefix}std::span<const real> {names[0]},")
        for n in names[1:-1]:
            lines.append(f"{cont}std::span<const real> {n},")
        lines.append(f"{cont}std::span<const real> {names[-1]})")
        lines.append("{")
        args = ", ".join(names)
        lines.append(f"    return {spec.name}{{{args}}};")
        lines.append("}")

    lines.extend(["", "} // namespace ccs::stencils", ""])

    return "\n".join(lines)


def generate_stencil_cpp(spec: StencilGenSpec) -> str:
    """Generate a complete stencil .cpp file from a StencilGenSpec."""
    smap = build_symbol_map(
        spec.param_arrays,
        has_psi=not spec.is_uniform,
        scalar_params=spec.scalar_params,
    )
    printer = StencilCodePrinter(symbol_map=smap)
    return "".join(
        [
            _emit_header(spec),
            _emit_struct_preamble(spec),
            _emit_query_methods(spec),
            _emit_interior_method(spec),
            _emit_nbs_dispatcher(spec),
            _emit_nbs_methods(spec, printer),
            _emit_factory(spec),
        ]
    )


# ── 20.4f: Test file generator ─────────────────────────────────────────


@dataclass
class TestCase:
    """A single test case for a generated .t.cpp file."""

    __test__ = False  # prevent pytest collection

    bc_type: str  # "Floating" or "Dirichlet"
    h: float
    psi: float
    alpha_values: dict[str, list[float]]
    expected_coeffs: list[float]
    margin: float = 1.0e-8


LUA_KEY_MAP = {
    "alpha": "alpha",
    "fa": "floating_alpha",
    "da": "dirichlet_alpha",
    "ia": "interpolant_alpha",
}


def _format_margin(v: float) -> str:
    """Format a margin value in C++ scientific notation style (e.g. 1.0e-8)."""
    return f"{v:.1e}".replace("e-0", "e-").replace("e+0", "e+")


def compute_test_values(
    coeffs: list[Expr],
    alpha_values: dict[str, list[float]],
    h: float,
    psi: float,
    right: bool = False,
    nu: int = 1,
) -> list[float]:
    """Evaluate symbolic coefficients at given parameter values."""
    subs: dict[Symbol, float] = {}
    for name, values in alpha_values.items():
        for i, v in enumerate(values):
            subs[Symbol(f"{name}_{i}")] = v
    subs[Symbol("psi")] = psi
    subs[Symbol("h")] = h

    result = [float(c.xreplace(subs)) for c in coeffs]

    h_divisor = h if nu == 1 else h * h
    result = [v / h_divisor for v in result]

    if right and nu == 1:
        result = [-v for v in reversed(result)]
    elif right and nu == 2:
        result = list(reversed(result))

    return result


def generate_test_cpp(
    spec: StencilGenSpec,
    test_cases: list[TestCase],
) -> str:
    """Generate a complete .t.cpp test file from a StencilGenSpec and test cases."""
    scheme_type = spec.name.rsplit("_", 1)[0]
    order = spec.derivative_order

    # Build Lua alpha table string
    lua_lines: list[str] = []
    for key, count in spec.param_arrays.items():
        lua_key = LUA_KEY_MAP.get(key, key)
        # Use the alpha values from the first test case for the Lua config
        if test_cases and key in test_cases[0].alpha_values:
            vals = test_cases[0].alpha_values[key]
        else:
            vals = [0.0] * count
        val_str = ", ".join(f"{v!r}" for v in vals)
        lua_lines.append(f"                {lua_key} = {{{val_str}}}")
    alpha_lua_table = ",\n".join(lua_lines)

    lines: list[str] = []

    # Includes
    lines.append('#include "stencil.hpp"')
    lines.append("")
    lines.append("#include <catch2/catch_approx.hpp>")
    lines.append("#include <catch2/catch_test_macros.hpp>")
    lines.append("#include <catch2/matchers/catch_matchers_vector.hpp>")
    lines.append("")
    lines.append("#include <vector>")
    lines.append("")
    lines.append("#include <sol/sol.hpp>")
    lines.append("#include <spdlog/spdlog.h>")
    lines.append("")
    lines.append("using Catch::Matchers::Approx;")
    lines.append("using namespace ccs;")
    lines.append("")

    # TEST_CASE open
    lines.append(f'TEST_CASE("{spec.name}")')
    lines.append("{")
    lines.append("    using T = std::vector<real>;")
    lines.append("    sol::state lua;")
    lines.append("    lua.open_libraries(sol::lib::base, sol::lib::math);")
    lines.append('    lua.script(R"(')
    lines.append("        simulation = {")
    lines.append("            scheme = {")
    lines.append(f"                order = {order},")
    lines.append(f'                type = "{scheme_type}",')
    lines.append(alpha_lua_table)
    lines.append("            }")
    lines.append("        }")
    lines.append('    )");')
    lines.append("")
    lines.append('    auto st_opt = stencil::from_lua(lua["simulation"]);')
    lines.append("    REQUIRE(!!st_opt);")
    lines.append("    const auto& st = *st_opt;")

    # Each test case as a scoped block
    for tc in test_cases:
        expected_r = spec.R if tc.bc_type == "Floating" else spec.R - 1
        n_coeffs = expected_r * spec.T

        lines.append("")
        lines.append("    {")
        lines.append(f"        auto [p, r, t, x] = st.query(bcs::{tc.bc_type});")
        lines.append(f"        REQUIRE(p == {spec.P});")
        lines.append(f"        REQUIRE(r == {expected_r});")
        lines.append(f"        REQUIRE(t == {spec.T});")
        lines.append("        REQUIRE(x == 0);")
        lines.append("")
        lines.append(f"        T c({n_coeffs});")
        lines.append("        T ex{};")
        lines.append("")
        lines.append(
            f"        st.nbs({tc.h!r}, bcs::{tc.bc_type}, {tc.psi!r}, false, c, ex);"
        )

        # Format expected values
        val_strs = [repr(v) for v in tc.expected_coeffs]
        lines.append("        REQUIRE_THAT(c,")
        lines.append(f"                     Approx(T{{{val_strs[0]},")
        for v in val_strs[1:-1]:
            lines.append(f"                              {v},")
        lines.append(f"                              {val_strs[-1]}}})")
        lines.append(f"                         .margin({_format_margin(tc.margin)}));")
        lines.append("    }")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)
