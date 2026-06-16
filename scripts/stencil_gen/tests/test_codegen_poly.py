"""Codegen emission-shape + capstone numeric-equivalence tests for polyE2_1.

Covers design §4.2 (emission-shape substring assertions) and §4.4 (the capstone:
regenerated output/polyE2_1.cpp ≡ committed src/stencils/polyE2_1.cpp to 1e-12 on
a sampled (y, psi, fa, ia, da) grid for ALL FOUR methods, including the exact
hard-coded dirichlet array from polyE2_1.t.cpp).
"""

import os
import random
import sys

import pytest

from stencil_gen.codegen import generate_stencil_cpp
from stencil_gen.interp import build_polyE2_1_spec

_tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
sys.path.insert(0, _tools_dir)

from eval_poly import make_all  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_COMMITTED = os.path.join(_REPO, "src", "stencils", "polyE2_1.cpp")
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
_OUTPUT = os.path.join(_OUTPUT_DIR, "polyE2_1.cpp")


@pytest.fixture(scope="module")
def generated_code():
    spec = build_polyE2_1_spec()
    code = generate_stencil_cpp(spec)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    with open(_OUTPUT, "w") as f:
        f.write(code)
    return code


# ── §4.2 emission-shape ──────────────────────────────────────────────────
def test_query_interp_shape(generated_code):
    assert "interp_info query_interp() const { return {2, 4}; }" in generated_code


def test_interp_interior_shape(generated_code):
    assert (
        "std::span<const real> interp_interior(real y, std::span<real> c) const"
        in generated_code
    )
    assert "if (y > 0) {" in generated_code
    assert "return c.subspan(0, 2);" in generated_code


def test_interp_wall_shape(generated_code):
    assert (
        "interp_wall(int i, real y, real psi, std::span<real> c, bool right) const"
        in generated_code
    )
    assert "if (right) {" in generated_code
    assert "switch (i) {" in generated_code
    assert "case 0:" in generated_code
    assert "case 1:" in generated_code
    assert "return c.subspan(0, 4);" in generated_code


def test_interp_wall_has_cse(generated_code):
    # CSE temporaries present in the interp_wall body
    body = generated_code.split("interp_wall(int i", 1)[1].split("interior(real h", 1)[0]
    assert "const real t5 =" in body


def test_interp_wall_no_h_no_reverse(generated_code):
    """interp_wall is value-based: no /h, no reverse (negative assertion)."""
    body = generated_code.split("interp_wall(int i", 1)[1].split("interior(real h", 1)[0]
    assert "/= h" not in body
    assert "std::ranges::reverse" not in body
    assert "*= -1" not in body


def test_interp_struct_arrays(generated_code):
    assert "std::array<real, 6> fa;" in generated_code
    assert "std::array<real, 3> da;" in generated_code
    assert "std::array<real, 4> ia;" in generated_code


# ── §4.4 capstone: numeric equivalence ───────────────────────────────────
@pytest.fixture(scope="module")
def evaluators(generated_code):
    return {
        "committed": make_all(_COMMITTED),
        "regen": make_all(_OUTPUT),
    }


def test_capstone_numeric_equivalence(evaluators):
    """Regenerated ≡ committed to 1e-12 across a sampled grid, all four methods."""
    committed = evaluators["committed"]
    regen = evaluators["regen"]
    random.seed(20240615)
    maxerr = {k: 0.0 for k in committed}

    for _ in range(200):
        y = round(random.uniform(-0.49, 0.49), 5)
        psi = round(random.uniform(0.001, 0.999), 5)
        fa = [round(random.uniform(-1, 1), 4) for _ in range(6)]
        da = [round(random.uniform(-1, 1), 4) for _ in range(3)]
        ia = [round(random.uniform(-1, 1), 4) for _ in range(4)]
        h = round(random.uniform(0.5, 2.0), 4)

        for yv in (y, -y if y != 0 else 0.1):
            a = committed["interp_interior"](yv)
            b = regen["interp_interior"](yv)
            maxerr["interp_interior"] = max(
                maxerr["interp_interior"], *(abs(x - z) for x, z in zip(a, b))
            )
        for i in (0, 1):
            for right in (False, True):
                a = committed["interp_wall"](i, y, psi, fa, da, ia, right)
                b = regen["interp_wall"](i, y, psi, fa, da, ia, right)
                maxerr["interp_wall"] = max(
                    maxerr["interp_wall"], *(abs(x - z) for x, z in zip(a, b))
                )
        for right in (False, True):
            a = committed["nbs_floating"](h, psi, fa, da, ia, right)
            b = regen["nbs_floating"](h, psi, fa, da, ia, right)
            maxerr["nbs_floating"] = max(
                maxerr["nbs_floating"], *(abs(x - z) for x, z in zip(a, b))
            )
            a = committed["nbs_dirichlet"](h, psi, fa, da, ia, right)
            b = regen["nbs_dirichlet"](h, psi, fa, da, ia, right)
            maxerr["nbs_dirichlet"] = max(
                maxerr["nbs_dirichlet"], *(abs(x - z) for x, z in zip(a, b))
            )

    for method, err in maxerr.items():
        assert err < 1e-12, f"{method}: max abs error {err:.3e} >= 1e-12"


def test_capstone_dirichlet_hardcoded(evaluators):
    """Regenerated nbs_dirichlet reproduces the exact polyE2_1.t.cpp:63-70 array
    (psi=0.001, h=1, da=[3/25, 13/100, 7/50])."""
    regen = evaluators["regen"]
    expected = [
        -0.4395604395604396,
        -0.4166369491239419,
        0.7128343378083233,
        0.14336305087605816,
        -0.4295704295704296,
        -0.435,
        0.7295704295704296,
        0.135,
    ]
    got = regen["nbs_dirichlet"](
        1.0, 0.001, [0, 0, 0, 0, 0, 0], [3 / 25, 13 / 100, 7 / 50], [0, 0, 0, 0]
    )
    for g, e in zip(got, expected):
        assert abs(g - e) < 1e-12, f"got {g!r}, expected {e!r}"


def test_capstone_interp_interior_oracle(evaluators):
    """interp_interior reproduces the bf oracle from polyE2_1.t.cpp:249-265:
    Σ v·bf(mesh) == bf(mesh[center] + y·h), center = 1-(y>0), uniform h."""
    regen = evaluators["regen"]
    bf = lambda x: 2 * x + 1  # noqa: E731
    ymin, ymax = -1.0, 5.0
    # mesh = linear_distribute(ymin, ymax, p=2): 2 points
    mesh = [ymin, ymax]
    h = mesh[1] - mesh[0]
    for k in range(11):
        y = -0.45 + k * 0.9 / 10
        center = 1 - int(y > 0)
        v = regen["interp_interior"](y)
        # interior interp uses a local 2-node bracketing window around `center`.
        # The test mesh has 2 nodes; the inner-product is over those two nodes.
        yi = sum(v[j] * bf(mesh[j]) for j in range(2))
        assert abs(yi - bf(mesh[center] + y * h)) < 1e-9
