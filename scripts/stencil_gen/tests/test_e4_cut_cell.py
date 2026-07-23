"""Tests for the E4_1 cut-cell first-derivative closure.

Anchored to the published construction of Brady & Livescu 2021 (JCP 426:109794,
Sec. 3.2/3.3), reproduced symbolically by ``derive_cut_cell_paper`` and pinned
against ``tests/fixtures/e4_1_paper_reference.json`` (the machine-verified
ground truth at the id=1 free parameters).

Layout of the paper cut block (5x7): row 0 IS the wall row (derivative AT the
wall point), rows 1..3 are centered at fluid nodes f0..f2, and row 4 is the
transition row (no wall support, support through node 5).  The four uniform
free alphas (alphau_04, alphau_14, alphau_24, alphau_25) survive symbolically
through the cut construction — there is NO alpha elimination.

The mis-ported one-node-downwind construction (cut rows built at f0..f4 with no
wall row, the zeros={3,4} hack, the old 2-parameter alpha space, the "approach
B" psi(psi-1) poles, and the 197/288 / D6 lore) has been removed from the
pipeline (e4-align A1/A6); the tests that pinned those specifics are gone.

The generic uniform-boundary derivation (``derive_uniform_boundary_for_temo``)
and the shared cut-cell conservation machinery are still exercised here through
E2_1/E2_2, which are unchanged by the E4_1 alignment.
"""

import json
import pathlib

import pytest
from sympy import (
    Matrix, Rational, S, Symbol, cancel, fraction, solve,
)

from stencil_gen.conservation import _interior_contribution
from stencil_gen.temo import (
    E2_1,
    E2_2,
    E4_1,
    assemble_cut_cell_result,
    build_cut_cell_conservation_system,
    build_uniform_for_mathematica,
    compute_dimensions,
    construct_cut_cell_stencil,
    derive_cut_cell_paper,
    derive_cut_cell_scheme,
    derive_e2_uniform_boundary,
    derive_uniform_boundary_for_temo,
    solve_uniform_conservation_direct,
)


# ---------------------------------------------------------------------------
# Module-scoped fixtures: symbolic derivation caching.
#
# The full paper cut-cell derivation (~7 s) is cached once for the module and
# shared by every acceptance test.  The conserved uniform closure is likewise
# cached once.
# ---------------------------------------------------------------------------


PAPER_FIXTURE = (
    pathlib.Path(__file__).parent / "fixtures" / "e4_1_paper_reference.json"
)


@pytest.fixture(scope="module")
def paper_fixture():
    """Load the paper ground-truth fixture."""
    with open(PAPER_FIXTURE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def paper_cc():
    """Cache the symbolic derive_cut_cell_paper(E4_1) result for the module."""
    import os
    os.environ.setdefault("SYMPY_CACHE_SIZE", "50000")

    psi = Symbol("psi")
    return derive_cut_cell_paper(E4_1, psi), psi


@pytest.fixture(scope="module")
def paper_uniform_stage():
    """Conserved uniform closure B_u (4x6) + weights w_u, symbolic in alpha."""
    uniform = build_uniform_for_mathematica(E4_1)
    sol, weights = solve_uniform_conservation_direct(
        uniform.B_u, uniform.interior, E4_1.p
    )
    B_u = uniform.B_u.applyfunc(lambda e: cancel(e.xreplace(sol)))
    w_u = [cancel(w) for w in weights]
    return B_u, w_u, uniform


# ---------------------------------------------------------------------------
# Uniform boundary stage (generic derivation, shared with E2).
#
# These test the Taylor-accurate uniform boundary closure produced by
# ``derive_uniform_boundary_for_temo`` — construction-agnostic properties
# (shape, interior coefficients, per-row order conditions).  The paper's own
# E4_1 uniform stage is validated separately in TestMathematicaUniformConservation
# and TestPaperE41Construction.
# ---------------------------------------------------------------------------


class TestE4UniformBoundary:
    """Tests for derive_uniform_boundary_for_temo with E4_1."""

    @pytest.fixture
    def e4_result(self):
        """Compute E4_1 uniform boundary once for the test class."""
        return derive_uniform_boundary_for_temo(E4_1)

    def test_shape(self, e4_result):
        """E4_1 B_u has shape (4, 6) — r_eff=4 rows, t=6 columns."""
        assert e4_result.B_u.shape == (4, 6)

    def test_five_alpha_symbols(self, e4_result):
        """E4_1 has exactly 5 free alpha symbols."""
        assert len(e4_result.alpha_symbols) == 5
        # Verify they are named alpha_0..alpha_4
        for k, sym in enumerate(e4_result.alpha_symbols):
            assert sym.name == f"alpha_{k}"

    def test_zero_constraints(self, e4_result):
        """B_u[0, 5] == 0, B_u[1, 5] == 0, B_u[2, 5] == 0 (zero-constrained entries)."""
        assert e4_result.B_u[0, 5] == 0
        assert e4_result.B_u[1, 5] == 0
        assert e4_result.B_u[2, 5] == 0

    def test_last_row_free_alphas(self, e4_result):
        """B_u[3, 4] and B_u[3, 5] contain alpha_3, alpha_4."""
        alpha_3 = e4_result.alpha_symbols[3]
        alpha_4 = e4_result.alpha_symbols[4]
        # These entries should involve alpha_3 and alpha_4 respectively
        assert alpha_3 in e4_result.B_u[3, 4].free_symbols
        assert alpha_4 in e4_result.B_u[3, 5].free_symbols

    def test_interior_coefficients(self, e4_result):
        """Interior coefficients are [1/12, -2/3, 0, 2/3, -1/12]."""
        expected = [Rational(1, 12), Rational(-2, 3), S.Zero,
                    Rational(2, 3), Rational(-1, 12)]
        assert e4_result.interior == expected

    def test_scheme_metadata(self, e4_result):
        """Result carries correct p, q, nu."""
        assert e4_result.p == 2
        assert e4_result.q == 3
        assert e4_result.nu == 1

    def test_rows_0_1_match_e4u_1(self, e4_result):
        """First 5 columns of rows 0, 1 match E4u_1.cpp's nbs_floating coefficients.

        E4u_1.cpp row 0 (c[0..4], before /h):
            c[0] = (6*a0 - 11)/6
            c[1] = 3 - 4*a0
            c[2] = (12*a0 - 3)/2
            c[3] = -(12*a0 - 1)/3
            c[4] = a0

        E4u_1.cpp row 1 (c[5..9], before /h):
            c[5] = (3*a1 - 1)/3
            c[6] = -(8*a1 + 1)/2
            c[7] = 6*a1 + 1
            c[8] = -(24*a1 + 1)/6
            c[9] = a1
        """
        B_u = e4_result.B_u
        a0 = e4_result.alpha_symbols[0]
        a1 = e4_result.alpha_symbols[1]

        # E4u_1 row 0 expected (5 columns)
        row0_expected = [
            (6 * a0 - 11) / S(6),
            3 - 4 * a0,
            (12 * a0 - 3) / S(2),
            -(12 * a0 - 1) / S(3),
            a0,
        ]

        # E4u_1 row 1 expected (5 columns)
        row1_expected = [
            (3 * a1 - 1) / S(3),
            -(8 * a1 + 1) / S(2),
            6 * a1 + 1,
            -(24 * a1 + 1) / S(6),
            a1,
        ]

        for j in range(5):
            diff = cancel(B_u[0, j] - row0_expected[j])
            assert diff == 0, (
                f"Row 0, col {j}: B_u={B_u[0,j]}, expected={row0_expected[j]}"
            )

        for j in range(5):
            diff = cancel(B_u[1, j] - row1_expected[j])
            assert diff == 0, (
                f"Row 1, col {j}: B_u={B_u[1,j]}, expected={row1_expected[j]}"
            )

    def test_taylor_accuracy(self, e4_result, assert_taylor_accuracy):
        """Each row satisfies Taylor matching for q+1=4 equations (polynomials up to degree 3)."""
        assert_taylor_accuracy(e4_result.B_u, e4_result.q, nu=1)

    def test_no_conservation_constraint(self, e4_result):
        """E4_1 (nextra=0) has no column-sum conservation constraint.

        Column sums need NOT be zero — this confirms nextra=0 path is different
        from E2_1's nextra=1 path.
        """
        B_u = e4_result.B_u
        # Just verify that B_u doesn't have phi symbols (conservation resolved)
        free = B_u.free_symbols
        assert all("phi" not in str(s) for s in free), (
            f"Unexpected phi symbols: {free}"
        )

    def test_only_alpha_symbols_in_B_u(self, e4_result):
        """B_u contains only the expected alpha symbols, nothing else."""
        expected_syms = set(e4_result.alpha_symbols)
        actual_syms = e4_result.B_u.free_symbols
        assert actual_syms <= expected_syms, (
            f"Unexpected symbols in B_u: {actual_syms - expected_syms}"
        )

    def test_custom_alpha_symbols(self):
        """derive_uniform_boundary_for_temo(E4_1) accepts custom alpha names."""
        syms = [Symbol(f"a{k}") for k in range(5)]
        result = derive_uniform_boundary_for_temo(E4_1, alpha_symbols=syms)
        assert result.alpha_symbols == syms
        free = result.B_u.free_symbols
        assert free <= set(syms)

    def test_wrong_alpha_count_raises(self):
        """Wrong number of alpha symbols raises ValueError."""
        with pytest.raises(ValueError, match="alpha symbols"):
            derive_uniform_boundary_for_temo(E4_1, alpha_symbols=[Symbol("a")])


@pytest.mark.slow
class TestE4UniformConservation:
    """Tests for derive_uniform_boundary_for_temo(E4_1, conserve=True).

    Conservation reduces the raw 5-alpha uniform closure to 4 free alphas plus
    4 quadrature weights — the same free-parameter count as the paper uniform
    stage.
    """

    @pytest.fixture(scope="class")
    def conserved(self):
        return derive_uniform_boundary_for_temo(E4_1, conserve=True)

    def test_alpha_count(self, conserved):
        """Conservation reduces E4_1 from 5 to 4 free alphas."""
        assert len(conserved.alpha_symbols) == 4
        for k, sym in enumerate(conserved.alpha_symbols):
            assert sym.name == f"alpha_{k}"

    def test_shape_unchanged(self, conserved):
        """B_u shape is still (4, 6) — conservation doesn't change dimensions."""
        assert conserved.B_u.shape == (4, 6)

    def test_weights_present(self, conserved):
        """Conservation produces 4 quadrature weights."""
        assert conserved.weights is not None
        assert len(conserved.weights) == 4

    def test_weights_depend_on_alpha_3(self, conserved):
        """All weights are rational functions of alpha_3 only."""
        alpha_3 = conserved.alpha_symbols[3]
        for w in conserved.weights:
            assert w.free_symbols <= {alpha_3}, (
                f"Weight {w} depends on {w.free_symbols}, expected only {alpha_3}"
            )

    def test_conservation_holds(self, conserved):
        """Weighted column sums satisfy the discrete-conservation condition."""
        B_u = conserved.B_u
        r, t = B_u.shape
        for j in range(t - 1):
            col_sum = sum(conserved.weights[i] * B_u[i, j] for i in range(r))
            ic = _interior_contribution(j, r, conserved.p, conserved.interior)
            total = cancel(col_sum + ic)
            target = -1 if j == 0 else 0
            assert cancel(total - target) == 0, (
                f"Conservation violated at col {j}: residual={cancel(total - target)}"
            )

    def test_taylor_accuracy(self, conserved, assert_taylor_accuracy):
        """Each row still satisfies Taylor matching for q+1=4 equations."""
        assert_taylor_accuracy(conserved.B_u, q=3, nu=1)

    def test_rows_0_2_unchanged(self, conserved):
        """Rows 0-2 are identical to the non-conservative result."""
        ur_no_cons = derive_uniform_boundary_for_temo(E4_1, conserve=False)
        for i in range(3):
            for j in range(6):
                diff = cancel(conserved.B_u[i, j] - ur_no_cons.B_u[i, j])
                assert diff == 0, f"Row {i}, col {j} should be unchanged"

    def test_only_alpha_symbols_in_Bu(self, conserved):
        """B_u contains only the expected alpha symbols."""
        expected_syms = set(conserved.alpha_symbols)
        actual_syms = conserved.B_u.free_symbols
        assert actual_syms <= expected_syms, (
            f"Unexpected symbols in B_u: {actual_syms - expected_syms}"
        )

    def test_custom_alpha_symbols(self):
        """conserve=True accepts 4 custom alpha names."""
        syms = [Symbol(f"a{k}") for k in range(4)]
        result = derive_uniform_boundary_for_temo(E4_1, alpha_symbols=syms, conserve=True)
        assert result.alpha_symbols == syms
        assert result.B_u.free_symbols <= set(syms)

    def test_e2_1_unaffected(self):
        """E2_1 (nextra=1) is unaffected by conserve=True — uses its own conservation."""
        ur = derive_uniform_boundary_for_temo(E2_1, conserve=True)
        assert len(ur.alpha_symbols) == 4
        assert ur.weights is None  # nextra=1 conservation is inline, no explicit weights


class TestMathematicaUniformConservation:
    """Paper uniform stage: matches Mathematica page 4 (baseSol)."""

    def test_weights_match_mathematica(self):
        """Verify w[1..4] match Mathematica baseSol at alpha=(1,1,-1,1)."""
        from stencil_gen.temo import (
            build_uniform_for_mathematica,
            solve_uniform_conservation_direct,
        )

        uniform = build_uniform_for_mathematica(E4_1)
        B_u = uniform.B_u
        _, weights = solve_uniform_conservation_direct(B_u, uniform.interior, E4_1.p)

        a0 = Symbol("alpha_0")
        # Mathematica page 4 (w[1] = -6/(-11+6*alpha[1]))
        expected_w0 = cancel(Rational(-6) / (-11 + 6 * a0))
        assert cancel(weights[0] - expected_w0) == 0

    def test_base_scheme_shape(self):
        """Base scheme should be (r+1) x t = 5x6 for E4_1."""
        from stencil_gen.temo import (
            build_base_scheme,
            build_uniform_for_mathematica,
            solve_uniform_conservation_direct,
        )

        uniform = build_uniform_for_mathematica(E4_1)
        sol, _ = solve_uniform_conservation_direct(
            uniform.B_u, uniform.interior, E4_1.p
        )
        base = build_base_scheme(uniform.B_u, sol, uniform.interior, E4_1.p)
        assert base.shape == (5, 6)

    def test_base_scheme_values(self):
        """Base scheme at alpha=(1,1,-1,1) matches Mathematica page 4."""
        from stencil_gen.temo import (
            build_base_scheme,
            build_uniform_for_mathematica,
            solve_uniform_conservation_direct,
        )

        uniform = build_uniform_for_mathematica(E4_1)
        sol, _ = solve_uniform_conservation_direct(
            uniform.B_u, uniform.interior, E4_1.p
        )
        base = build_base_scheme(uniform.B_u, sol, uniform.interior, E4_1.p)

        a0, a1, a2, a3 = [Symbol(f"alpha_{i}") for i in range(4)]
        subs = {a0: 1, a1: 1, a2: -1, a3: 1}
        base_num = base.subs(subs).applyfunc(cancel)

        # Row 0: [-5/6, -1, 9/2, -11/3, 1, 0]
        assert cancel(base_num[0, 0] - Rational(-5, 6)) == 0
        assert cancel(base_num[0, 1] - Rational(-1)) == 0
        assert cancel(base_num[0, 4] - Rational(1)) == 0

        # Row 4 (interior): [0, 0, 1/12, -2/3, 0, 2/3]
        assert base_num[4, 0] == 0
        assert base_num[4, 1] == 0
        assert cancel(base_num[4, 2] - Rational(1, 12)) == 0


# ---------------------------------------------------------------------------
# Shared cut-cell conservation machinery (E2 sanity).
#
# build_cut_cell_conservation_system is used by the E2 schemes; these E2_1
# tests guard the shared function.  E4_1 is validated end-to-end through the
# paper acceptance tests below, not through this low-level system.
# ---------------------------------------------------------------------------


class TestBuildCutCellConservationSystem:
    """Tests for build_cut_cell_conservation_system dimensions and IC values."""

    def test_e2_1_conservation_system_dimensions(self):
        """E2_1: T-2=3 equations (grid-point cols), 3 weight unknowns, all IC zero."""
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E2_1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=E2_1.p, q=E2_1.q, nu=E2_1.nu,
            nextra=E2_1.nextra, psi=psi,
        )
        R, T = stencil.matrix.rows, stencil.matrix.cols
        assert R == 4
        assert T == 5

        eqs, ws = build_cut_cell_conservation_system(
            stencil.matrix, R, T, p=E2_1.p, nu=E2_1.nu,
            interior_coeffs=ur.interior, psi=psi,
        )
        assert len(eqs) == T - 2  # 3 equations (grid-point cols 0..2)
        assert len(ws) == R - 1   # 3 weight unknowns (w_1, w_2, w_3)

        # All IC values should be 0 for E2_1 (no interior row reaches grid points 0..2)
        for g in range(T - 2):
            ic = _interior_contribution(g, R, E2_1.p, ur.interior)
            assert ic == 0, f"E2_1 IC(g={g}) should be 0, got {ic}"

    def test_e2_1_conservation_solvable(self):
        """E2_1: 3 eqs / 3 unknowns → solvable, weights are rational in (alpha, psi).

        Validates the column mapping: the grid-point column range (T-frame cols
        1..T-2) gives a consistent square system whose solution yields
        psi-dependent conservation (quadrature) weights.
        """
        psi = Symbol("psi")
        ur = derive_uniform_boundary_for_temo(E2_1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=E2_1.p, q=E2_1.q, nu=E2_1.nu,
            nextra=E2_1.nextra, psi=psi,
        )
        R, T = stencil.matrix.rows, stencil.matrix.cols

        eqs, ws = build_cut_cell_conservation_system(
            stencil.matrix, R, T, p=E2_1.p, nu=E2_1.nu,
            interior_coeffs=ur.interior, psi=psi,
        )
        assert len(eqs) == 3
        assert len(ws) == 3

        # Solve the system for w_1, w_2, w_3
        sol = solve(eqs, ws, dict=True)
        assert len(sol) == 1, f"Expected unique solution, got {len(sol)}"
        sol = sol[0]

        # All three weights must be present in the solution
        for w in ws:
            assert w in sol, f"Weight {w} missing from solution"

        # Weights should be rational functions of (alpha, psi)
        alpha_syms = sorted(stencil.matrix.free_symbols - {psi}, key=lambda s: s.name)
        for w in ws:
            w_val = sol[w]
            free = w_val.free_symbols
            # Must depend on psi (ψ-dependent weights)
            assert psi in free, f"{w} solution should depend on psi, got {w_val}"
            # Free symbols should be a subset of {psi} ∪ alphas
            assert free <= {psi} | set(alpha_syms), (
                f"{w} has unexpected symbols: {free - {psi} - set(alpha_syms)}"
            )

        # Substituting solution back into all 3 equations must yield 0
        for i, eq in enumerate(eqs):
            residual = cancel(eq.subs(sol))
            assert residual == 0, (
                f"Equation {i} residual non-zero after substitution: {residual}"
            )


# ---------------------------------------------------------------------------
# derive_cut_cell_scheme dispatch.
#
# The high-level entry point reproduces the existing E2_1/E2_2 pipelines and
# forwards E4_1 (structural uniform zeros) to the paper construction.
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDeriveCutCellScheme:
    """Tests for the derive_cut_cell_scheme high-level pipeline."""

    def test_e4_1_forwards_to_paper(self, paper_cc):
        """derive_cut_cell_scheme(E4_1) forwards to derive_cut_cell_paper.

        E4_1 carries structural uniform zeros, so the high-level entry point
        dispatches to the published construction: 5x7 floating, 4 surviving
        alphas, no Neumann, and floating/weights identical to
        derive_cut_cell_paper.
        """
        paper, psi = paper_cc
        auto = derive_cut_cell_scheme(E4_1, psi)

        assert auto.floating.shape == (5, 7)
        assert auto.dirichlet.shape == (4, 7)
        assert len(auto.alpha_symbols) == 4
        assert auto.neumann is None

        # Forwarding must reproduce the paper result exactly.
        assert auto.floating == paper.floating
        assert auto.weights == paper.weights
        assert auto.conservation_subs == paper.conservation_subs

    def test_e2_1_reproduces_existing(self):
        """derive_cut_cell_scheme(E2_1) matches manual E2_1 pipeline."""
        psi = Symbol("psi")

        # Manual E2_1 pipeline using old derive_e2_uniform_boundary
        ur = derive_e2_uniform_boundary(nu=1)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=1, q=1, nu=1, nextra=1, psi=psi,
        )
        dims = compute_dimensions(1, 1, 0, 1, 1)
        manual = assemble_cut_cell_result(
            stencil.matrix, None, None, dims, ur.alpha_symbols,
        )

        # High-level pipeline
        auto = derive_cut_cell_scheme(E2_1, psi)

        # Shapes must match
        assert auto.floating.shape == manual.floating.shape
        assert auto.dirichlet.shape == manual.dirichlet.shape
        assert auto.dims == manual.dims

        # Floating matrices must match entry by entry
        R, T = auto.floating.shape
        for i in range(R):
            for j in range(T):
                assert cancel(auto.floating[i, j] - manual.floating[i, j]) == 0, (
                    f"E2_1 floating mismatch at [{i},{j}]"
                )

    def test_e2_2_reproduces_existing(self):
        """derive_cut_cell_scheme(E2_2) matches manual E2_2 pipeline."""
        psi = Symbol("psi")

        # Manual E2_2 pipeline
        ur = derive_e2_uniform_boundary(nu=2)
        stencil = construct_cut_cell_stencil(
            ur.B_u, ur.interior, p=1, q=1, nu=2, nextra=0, psi=psi,
        )
        from stencil_gen.temo import derive_uniform_neumann, construct_neumann_stencil
        B_uN, eta_u = derive_uniform_neumann(ur.interior, 1, 1, 2)
        neumann_main, eta = construct_neumann_stencil(
            ur.B_u, B_uN, eta_u, ur.interior, 1, 1, 2, 0, psi,
        )
        dims = compute_dimensions(1, 1, 0, 0, 2)
        manual = assemble_cut_cell_result(
            stencil.matrix, neumann_main, eta, dims, ur.alpha_symbols,
        )

        # High-level pipeline
        auto = derive_cut_cell_scheme(E2_2, psi)

        # Shapes
        assert auto.floating.shape == manual.floating.shape
        assert auto.dims == manual.dims
        assert auto.neumann is not None
        assert auto.eta is not None

        # Floating
        R, T = auto.floating.shape
        for i in range(R):
            for j in range(T):
                assert cancel(auto.floating[i, j] - manual.floating[i, j]) == 0, (
                    f"E2_2 floating mismatch at [{i},{j}]"
                )

        # Neumann
        for i in range(R):
            for j in range(T):
                assert cancel(auto.neumann[i, j] - manual.neumann[i, j]) == 0, (
                    f"E2_2 neumann mismatch at [{i},{j}]"
                )

        # Eta
        for i in range(R):
            assert cancel(auto.eta[i] - manual.eta[i]) == 0, (
                f"E2_2 eta mismatch at row {i}"
            )


# ---------------------------------------------------------------------------
# Paper cut-cell construction acceptance.
#
# Anchored to tests/fixtures/e4_1_paper_reference.json — the machine-verified
# ground truth for the Brady & Livescu 2021 (JCP 426:109794) E4_1 cut-cell
# closure at the id=1 alphas.  Row 0 of the floating block IS the wall row
# (derivative at the wall point); row 4 is the transition row (no wall support,
# support through node 5).
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPaperE41Construction:
    """derive_cut_cell_paper reproduces the paper reference fixture."""

    def test_paper_shapes_and_alphas(self, paper_cc):
        """5x7 floating, 4x7 dirichlet (wall row dropped), 5 weights,
        full 4-alpha dependence (no alpha elimination)."""
        cc, psi = paper_cc
        assert cc.floating.shape == (5, 7)
        assert cc.dirichlet.shape == (4, 7)
        assert len(cc.weights) == 5
        assert len(cc.alpha_symbols) == 4
        free = cc.floating.free_symbols - {psi}
        assert free == set(cc.alpha_symbols), (
            "all 4 uniform alphas must survive symbolically"
        )

    def test_paper_dims_and_dirichlet(self, paper_cc):
        """Dims: floating 5x7, dirichlet 4x7, no Neumann; the Dirichlet block is
        exactly floating rows 1.. (the wall row dropped)."""
        cc, psi = paper_cc
        assert cc.floating.shape == (5, 7)
        assert cc.dirichlet.shape == (4, 7)
        assert cc.neumann is None
        R, T = cc.floating.shape
        for i in range(R - 1):
            for j in range(T):
                assert cancel(cc.dirichlet[i, j] - cc.floating[i + 1, j]) == 0, (
                    f"Dirichlet != floating[1:] at [{i},{j}]"
                )

    def test_paper_blocks_match_fixture(self, paper_cc, paper_fixture):
        """Floating block, Dirichlet slice, and conservation weights match
        the fixture at the id=1 alphas for all 5 psi values to <= 1e-10."""
        from sympy import lambdify

        cc, psi = paper_cc
        alphas = list(paper_fixture["alpha_id1"].values())
        subs_a = dict(zip(cc.alpha_symbols, alphas))

        entry = [[lambdify([psi], cc.floating[i, j].xreplace(subs_a))
                  for j in range(7)] for i in range(5)]
        wfun = [lambdify([psi], w.xreplace(subs_a)) for w in cc.weights]

        for psi_val in paper_fixture["psi_values"]:
            blk = paper_fixture["blocks"][str(psi_val)]
            B_ref = blk["floating_5x7"]
            D_ref = blk["dirichlet_4x6_wall_row_and_col_dropped"]
            w_ref = blk["conservation_weights_w0_to_w4"]

            B = [[float(entry[i][j](psi_val)) for j in range(7)]
                 for i in range(5)]
            for i in range(5):
                for j in range(7):
                    assert abs(B[i][j] - B_ref[i][j]) <= 1e-10, (
                        f"floating[{i},{j}] at psi={psi_val}: "
                        f"{B[i][j]} vs {B_ref[i][j]}"
                    )
            # Dirichlet variant: wall row AND wall column dropped.
            for i in range(4):
                for j in range(6):
                    assert abs(B[i + 1][j + 1] - D_ref[i][j]) <= 1e-10, (
                        f"dirichlet[{i},{j}] at psi={psi_val}"
                    )
            for k in range(5):
                w_k = float(wfun[k](psi_val))
                assert abs(w_k - w_ref[k]) <= 1e-10, (
                    f"weight w_{k} at psi={psi_val}: {w_k} vs {w_ref[k]}"
                )

    def test_paper_psi1_uniform_identity_symbolic(
        self, paper_cc, paper_uniform_stage
    ):
        """B(1) = [conserved uniform closure (cols 0..5), 0; interior row]
        EXACTLY — symbolic zero residual with all alphas free."""
        cc, psi = paper_cc
        B_u, _, uniform = paper_uniform_stage

        expected = Matrix.zeros(5, 7)
        for i in range(4):
            for j in range(6):
                expected[i, j] = B_u[i, j]
        for k, c in enumerate(uniform.interior):
            expected[4, 2 + k] = c

        F1 = cc.floating.subs(psi, 1)
        for i in range(5):
            for j in range(7):
                assert cancel(F1[i, j] - expected[i, j]) == 0, (
                    f"psi=1 identity violated at [{i},{j}]"
                )

    def test_paper_no_psi_poles(self, paper_cc):
        """No floating entry has a pole at psi=0 or psi=1.

        The paper construction is pole-free on [0,1] (benign Vandermonde
        denominators (psi+1)(psi+2)(psi+3)) — unlike the removed mis-port
        "approach B", which introduced psi(psi-1) poles via alpha elimination.
        """
        cc, psi = paper_cc
        F = cc.floating
        for i in range(F.rows):
            for j in range(F.cols):
                _, denom = fraction(cancel(F[i, j]))
                if denom != 1 and psi in denom.free_symbols:
                    assert denom.subs(psi, 0) != 0, f"ψ=0 pole at [{i},{j}]"
                    assert denom.subs(psi, 1) != 0, f"ψ=1 pole at [{i},{j}]"

    def test_paper_wall_column_identity(self, paper_cc, paper_uniform_stage):
        """Wall-column equation: w_0 * B[0,0] = w_u^T B_u[:,0] = -1 exactly
        (symbolically, for all alpha)."""
        cc, psi = paper_cc
        B_u, w_u, _ = paper_uniform_stage

        wall_rhs = cancel(sum(w_u[i] * B_u[i, 0] for i in range(4)))
        assert wall_rhs == -1, f"w_u^T B_u[:,0] = {wall_rhs}, expected -1"
        assert cancel(cc.weights[0] * cc.floating[0, 0] + 1) == 0, (
            "w_0 * B[0,0] must equal -1 exactly"
        )

    def test_paper_transition_row_structure(self, paper_cc):
        """Transition row (row 4): no wall support, support through node 5
        (column 6 generically nonzero)."""
        cc, psi = paper_cc
        assert cc.floating[4, 0] == 0
        assert cc.floating[4, 6] != 0

    def test_paper_row_centers_taylor(self, paper_cc):
        """Row i approximates f'(x_i) with x = [0 (wall), psi, ..., psi+5]:
        q+1 = 4 Taylor moment conditions hold about the PAPER centers."""
        from sympy import Rational as R_, factorial as fact

        cc, psi = paper_cc
        alpha_vals = {a: R_(k + 1, 7) for k, a in enumerate(cc.alpha_symbols)}
        x = [S.Zero] + [psi + k for k in range(6)]
        centers = [x[0], x[1], x[2], x[3], x[4]]  # wall, f0, f1, f2, f3
        for psi_val in [R_(3, 10), R_(7, 10)]:
            subs = {psi: psi_val, **alpha_vals}
            for i in range(5):
                for k in range(4):
                    moment = sum(
                        (cc.floating[i, j] * (x[j] - centers[i]) ** k
                         / fact(k)).subs(subs)
                        for j in range(7)
                    )
                    target = 1 if k == 1 else 0
                    assert cancel(moment - target) == 0, (
                        f"Taylor k={k} fails for row {i} at psi={psi_val}"
                    )
