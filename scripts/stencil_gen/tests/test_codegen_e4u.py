"""End-to-end E4u_1 validation: derive + generate C++ code.

Tests the full pipeline from symbolic boundary derivation (20.2 + 20.3)
through C++ code generation (20.4), validating numerical equivalence with
the hand-written E4u_1 stencil.
"""

import subprocess
from pathlib import Path

import pytest
from sympy import Rational

from stencil_gen.interior import derive_interior, full_gamma_array
from stencil_gen.codegen import (
    StencilGenSpec,
    compute_test_values,
    generate_stencil_cpp,
)

# Known E4u_1 alpha values (from E4u_1.t.cpp)
ALPHA = [-0.7733323791884821, 0.1623961700641681]


@pytest.fixture(scope="module")
def e4u_data(e4u_pipeline):
    """Run pipeline once, reuse across tests."""
    updated_rows, _solution_dict, _w_syms, result = e4u_pipeline
    # Flatten row coefficients into a single list
    floating_coeffs = []
    for row in updated_rows:
        floating_coeffs.extend(row.coefficients)
    return {
        "updated_rows": updated_rows,
        "result": result,
        "floating_coeffs": floating_coeffs,
        "r": result.r,
        "t": result.t,
    }


def test_e4u_interior_coefficients():
    """Interior coefficients match the known E4u values."""
    result = derive_interior(s=0, p=2, nu=1)
    gamma = full_gamma_array(result)
    expected = [Rational(1, 12), Rational(-2, 3), 0, Rational(2, 3), Rational(-1, 12)]
    for got, want in zip(gamma, expected):
        assert got == want


def test_e4u_floating_numerical(e4u_data):
    """Floating BC coefficients match E4u_1.t.cpp reference values."""
    floating_coeffs = e4u_data["floating_coeffs"]

    values = compute_test_values(
        floating_coeffs,
        alpha_values={"alpha": ALPHA},
        h=2.0,
        psi=1.0,
    )

    reference = [
        -1.3033328562609077, 3.046664758376964, -3.069997137565446,
        1.713331425043631, -0.38666618959424104,
        -0.08546858163458262, -0.5747923401283361, 0.9871885101925043,
        -0.4081256734616695, 0.08119808503208405,
        0.0923093909615862, -0.5359042305130115, 0.3038563457695172,
        0.13076243615365518, 0.00897605762825287,
    ]

    assert values == pytest.approx(reference, abs=1e-8)


def test_e4u_dirichlet_numerical(e4u_data):
    """Dirichlet BC coefficients match E4u_1.t.cpp reference values."""
    floating_coeffs = e4u_data["floating_coeffs"]
    t = e4u_data["t"]
    # Skip row 0 for Dirichlet
    dirichlet_coeffs = floating_coeffs[t:]

    values = compute_test_values(
        dirichlet_coeffs,
        alpha_values={"alpha": ALPHA},
        h=0.5,
        psi=0.0,
    )

    reference = [
        -0.3418743265383305, -2.2991693605133445, 3.9487540407700172,
        -1.632502693846678, 0.3247923401283362,
        0.3692375638463448, -2.143616922052046, 1.2154253830780688,
        0.5230497446146207, 0.03590423051301148,
    ]

    assert values == pytest.approx(reference, abs=1e-8)


def test_e4u_full_pipeline(e4u_data):
    """Full pipeline: derive + generate C++ for E4u_1."""
    interior = derive_interior(s=0, p=2, nu=1)
    gamma = full_gamma_array(interior)
    floating_coeffs = e4u_data["floating_coeffs"]

    spec = StencilGenSpec(
        name="E4u_1",
        P=2,
        R=3,
        T=5,
        X=0,
        derivative_order=1,
        is_uniform=True,
        param_arrays={"alpha": 2},
        interior_coeffs=gamma,
        floating_coeffs=floating_coeffs,
        dirichlet_coeffs=floating_coeffs,  # row 0 gets sliced away by generator
    )

    code = generate_stencil_cpp(spec)

    # Structural checks
    assert "struct E4u_1" in code
    assert "static constexpr int P = 2;" in code
    assert "std::array<real, 2> alpha;" in code
    assert "interior(real h," in code
    assert "nbs_floating(real h," in code
    assert "nbs_dirichlet(real h," in code
    assert "make_E4u_1" in code
    assert "namespace ccs::stencils" in code

    # Find the nbs method definitions by searching for the signature pattern
    # The dispatcher uses "nbs_floating(h," while definitions use "nbs_floating(real h,"
    floating_start = code.index("nbs_floating(real h,")
    dirichlet_start = code.index("nbs_dirichlet(real h,")
    neumann_start = code.index("nbs_neumann")

    floating_section = code[floating_start:dirichlet_start]
    assert floating_section.count("c[") == 15, (
        f"Expected 15 c[] assignments in floating, got {floating_section.count('c[')}"
    )

    dirichlet_section = code[dirichlet_start:neumann_start]
    assert dirichlet_section.count("c[") == 10, (
        f"Expected 10 c[] assignments in dirichlet, got {dirichlet_section.count('c[')}"
    )


def test_e4u_generated_code_compiles(e4u_data):
    """Generated E4u_1 code compiles successfully.

    Skipped if no C++ compiler is available.
    """
    import shutil
    import tempfile

    if not shutil.which("g++") and not shutil.which("clang++"):
        pytest.skip("No C++ compiler available")

    interior = derive_interior(s=0, p=2, nu=1)
    gamma = full_gamma_array(interior)
    floating_coeffs = e4u_data["floating_coeffs"]

    spec = StencilGenSpec(
        name="E4u_1_gen",
        P=2,
        R=3,
        T=5,
        X=0,
        derivative_order=1,
        is_uniform=True,
        param_arrays={"alpha": 2},
        interior_coeffs=gamma,
        floating_coeffs=floating_coeffs,
        dirichlet_coeffs=floating_coeffs,
    )

    code = generate_stencil_cpp(spec)

    driver = """\
#include <array>
#include <span>
#include <algorithm>
#include <ranges>
#include <cmath>
#include <cstdio>

namespace ccs {
using real = double;
using integer = long;
}
namespace ccs::bcs {
enum type { Floating, Dirichlet, Neumann };
}
namespace ccs::stencils {
struct info { int p, r, t, nextra; };
struct interp_info { int p, t; };
struct stencil {
    template<typename T> stencil(T&&) {}
};
inline void copy_zero_padded(std::span<const double> src, std::span<double> dst) {
    auto n = std::min(src.size(), dst.size());
    std::copy_n(src.begin(), n, dst.begin());
    std::fill(dst.begin() + n, dst.end(), 0.0);
}
"""

    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
        f.write(driver)
        struct_start = code.index("struct ")
        ns_close = code.rindex("} // namespace")
        f.write(code[struct_start:ns_close])
        f.write("} // namespace ccs::stencils\n")
        f.write(
            """
int main() {
    printf("compilation ok\\n");
    return 0;
}
"""
        )
        tmpfile = f.name

    compiler = shutil.which("g++") or shutil.which("clang++")
    result = subprocess.run(
        [compiler, "-std=c++20", "-fsyntax-only", tmpfile],
        capture_output=True,
        text=True,
    )
    Path(tmpfile).unlink()
    assert result.returncode == 0, f"Compilation failed:\n{result.stderr}"
