"""Polynomial-interpolation cut-cell stencil derivation.

Faithful sympy port of the Mathematica poly-interp routines (``taylor.wl``
824-998).  The novel idea: define an *interpolation polynomial*

    f[i+y] = Σ_j gamma[i,j](y, psi, params) · f[node_j]      (TEMO-style)

so that the derivative stencil falls out as ``(1/h)·d/dy gamma|_{y=0}``.  ONE
consistent free-parameter set (``fa``/``ia``) feeds both extractions: the ``fa``
(``v`` of the ``polyBlend`` ``g → u + y·v``) survives ``d/dy|0`` and is shared
with the floating-derivative rows; the ``ia`` (``u``) lives only in the ``y^0``
term and touches the interpolation value alone.

The mapping to ``polyE2_1.cpp`` (the north star):

* ``interp_interior`` — 2-point linear interp selected by ``sign(y)``
  (``runtime_pos`` / ``runtime_neg``).  The full 3-point quadratic is also
  derived but only to cross-check the interior central-difference derivative.
* ``interp_wall`` — ``if (right) {…} else {…}`` with ``switch (i)`` cases.  The
  right branch is a *separately derived mirror* (NOT reverse+negate): value
  interpolation has no antisymmetry and the wall column moves from ``c[0]`` to
  ``c[T-1]``.

Verified against the committed C++ for all four wall cases (left/right ×
case0/case1) and against ``nbs_floating`` via the ``d/dy|0`` duality.
"""

from __future__ import annotations

from dataclasses import dataclass

from sympy import Expr, Matrix, Rational, Symbol, cancel, diff, factorial, solve, symbols

from stencil_gen.taylor_system import _interp_rhs


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class InterpInterior:
    """Interior interpolation polynomials (Mathematica interiorInterp).

    ``coeffs`` is the full ``2p+1`` polynomial in y (used only to DERIVE the
    interior central-difference derivative).  ``runtime_pos`` / ``runtime_neg``
    hold the *shipped* 2-point linear form selected by ``sign(y)`` — what
    ``query_interp().p`` and ``interp_interior`` actually emit.
    """

    coeffs: list
    p: int
    runtime_p: int
    runtime_pos: list
    runtime_neg: list


@dataclass
class InterpRow:
    """One cut-cell interpolation wall row (Mathematica polyInterpSchemes).

    ``coeffs`` is length T, each an Expr in (y, psi, fa_k, ia_k).  Column order
    matches C++: left → [wall, node, node, node]; right → [node, node, node, wall].
    """

    row_index: int
    coeffs: list
    fa_syms: list
    ia_syms: list


@dataclass
class InterpResult:
    """Full interp side of a poly stencil — the codegen feed."""

    interior: InterpInterior
    left_rows: list
    right_rows: list
    interp_P: int
    interp_T: int
    y: Symbol
    psi: Symbol


# ---------------------------------------------------------------------------
# Interior interpolation
# ---------------------------------------------------------------------------
def derive_interior_interp(p: int, y: Symbol) -> InterpInterior:
    """Mathematica interiorInterp (taylor.wl:832-835).

    Solve Σ_{j=-p..p} gamma_j f[i+j] = f[i+y] on a centered uniform 2p+1
    stencil.  Entries are exact in y (no psi), so ``Matrix.solve`` + ``cancel``
    suffices.  Returns the full 2p+1 polynomial AND the shipped 2-point linear
    runtime form (the bracketing-pair Lagrange weights selected by sign(y)).

    For p=1: full=[(y²-y)/2, 1-y², (y²+y)/2]; runtime_pos=[1-y, y],
    runtime_neg=[-y, 1+y].
    """
    nodes = list(range(-p, p + 1))
    n_eqs = 2 * p + 1
    A = Matrix(
        n_eqs,
        n_eqs,
        lambda k, j: Rational(nodes[j] ** k) / factorial(k),
    )
    rhs = _interp_rhs(n_eqs, y)
    full = [cancel(c) for c in A.solve(rhs)]

    # Shipped 2-point linear forms: bracketing pair [0,1] (y>0) / [-1,0] (y<=0).
    # For y>0 the field lies between node0 and node1: Lagrange weights [1-y, y].
    # For y<=0 it lies between node-1 and node0: weights [-y, 1+y].
    runtime_pos = _two_point_linear(0, 1, y)
    runtime_neg = _two_point_linear(-1, 0, y)
    return InterpInterior(
        coeffs=full,
        p=p,
        runtime_p=2,
        runtime_pos=runtime_pos,
        runtime_neg=runtime_neg,
    )


def _two_point_linear(a: int, b: int, y: Symbol) -> list:
    """Linear Lagrange weights on nodes {a, b} for the value at offset y."""
    A = Matrix(2, 2, lambda k, j: Rational((a if j == 0 else b) ** k) / factorial(k))
    sol = A.solve(_interp_rhs(2, y))
    return [cancel(sol[0]), cancel(sol[1])]


# ---------------------------------------------------------------------------
# The novel core: derivative from interpolation
# ---------------------------------------------------------------------------
def derivative_from_interp(interp_coeffs: list, y: Symbol, nu: int = 1) -> list:
    """f^(nu)[i] = d^nu/dy^nu f[i+y]|_{y=0}.

    Maps Mathematica polyScheme's ``MapAt[D[#,y]&, …] /. y->0`` (taylor.wl:954).
    Because each interp coeff is (poly in y, linear in ia=u and fa=v), ``d/dy|0``
    kills the ``ia`` (y^0) part and keeps ``fa`` (y^1).
    Verified: [(y²-y)/2, 1-y², (y²+y)/2] → [-1/2, 0, 1/2].
    """
    return [cancel(diff(c, y, nu).subs(y, 0)) for c in interp_coeffs]


# ---------------------------------------------------------------------------
# Cut-cell wall rows
# ---------------------------------------------------------------------------
def build_cut_cell_interp_deltas(T: int, psi, left: bool) -> list:
    """Node x-positions (wall-local) for the cut-cell interp stencil.

    LEFT  → [wall, node0, node1, …] = [-psi, 0, 1, …, T-2]  (wall is col 0).
    RIGHT → […, node, node, wall]   = [-(T-2), …, -1, 0, psi] (wall is col T-1).
    The right layout is the mirror of the left; it is genuinely separate (interp
    is value-based, no antisymmetry).
    """
    if left:
        return [-psi] + [Rational(j) for j in range(T - 1)]
    return [Rational(-(T - 2) + j) for j in range(T - 1)] + [psi]


def solve_interp_row(
    deltas: list,
    q: int,
    xe,
    zeroed_col: int,
    free_col: int,
    free_sym: Symbol,
) -> list:
    """ONE variant of a cut-cell wall row (one schemeCoefficients call inside
    polyInterpSchemes).

    Drops ``zeroed_col`` (polyZero), holds ``free_col`` as the single free
    coefficient, solves the q+1 exactness equations (RHS = eval-at-``xe``
    functional) for the determined columns, and returns the length-T coeff list.

    Solved via ``Matrix.solve`` + ``cancel`` over QQ(psi)[y] — y never enters a
    denominator for E2 (denominators are only 2(1+psi)).
    """
    T = len(deltas)
    n_det = q + 1
    kept = [j for j in range(T) if j != zeroed_col]
    det_cols = [j for j in kept if j != free_col]
    assert len(det_cols) == n_det, (det_cols, n_det)

    V_det = Matrix(
        n_det, n_det, lambda k, idx: deltas[det_cols[idx]] ** k / factorial(k)
    )
    V_free = Matrix(n_det, 1, lambda k, _: deltas[free_col] ** k / factorial(k))
    rhs = _interp_rhs(n_det, xe)
    rhs_adj = rhs - V_free * Matrix([free_sym])
    det_sol = V_det.solve(rhs_adj)

    out = [Rational(0)] * T
    for idx, j in enumerate(det_cols):
        out[j] = cancel(det_sol[idx])
    out[free_col] = free_sym
    return out


def derive_cut_cell_interp_row(
    deltas: list,
    q: int,
    xe,
    y: Symbol,
    variant_specs: list,
) -> list:
    """polyInterpSchemes for one wall row: two zeroed-column variants, blend,
    simpleAverage.

    ``variant_specs`` is a list of (zeroed_col, free_col, ia_sym, fa_sym).  Each
    variant is solved with one raw free coeff, then ``polyBlend``'d
    (raw → ia_sym + y·fa_sym) and the variants are simple-averaged coeff-by-coeff
    (yielding the 0.5* factors).
    """
    raw = Symbol("_g_raw")
    blended_variants = []
    for zeroed_col, free_col, ia_sym, fa_sym in variant_specs:
        v = solve_interp_row(deltas, q, xe, zeroed_col, free_col, raw)
        vb = [cancel(c.subs(raw, ia_sym + y * fa_sym)) for c in v]
        blended_variants.append(vb)

    n = len(blended_variants)
    T = len(deltas)
    return [
        cancel(sum(bv[j] for bv in blended_variants) / n) for j in range(T)
    ]


# ---------------------------------------------------------------------------
# Top-level orchestrator (E2 specialization)
# ---------------------------------------------------------------------------
def derive_poly_interp(p: int = 1, q: int = 1) -> InterpResult:
    """Top-level orchestrator for the interp side of an E2-poly stencil.

    For E2: p=1, interp_T=4, interp_P=2, R-1=2 wall rows per branch.  q is the
    interpolation exactness order (degree-1 = q=1, two exactness equations).

    constrain_interp ordering (load-bearing — picks which fa[k]/ia[k] lands in
    which coefficient, matched to ``polyE2_1.cpp``):

    * LEFT  ascending: case0 → ia_0/fa_0 (variant zero-wall), ia_1/fa_1
      (variant zero-node0); case1 → ia_2/fa_2, ia_3/fa_3.
    * RIGHT mirror: case0 → ia_0/fa_0 (variant zero-wall@col3), ia_1/fa_1
      (variant zero-node@col2); case1 → ia_2/fa_2, ia_3/fa_3.

    The interp target offset (post-adjustment ``y`` convention, see
    ``stencil::interp``): for the row whose closest node is the wall, the field
    is evaluated at the wall's x-position + y; otherwise at the node + y.
    """
    if p != 1:
        raise NotImplementedError("derive_poly_interp currently targets E2 (p=1)")

    y = Symbol("y")
    psi = Symbol("psi")
    T = 2 * p + 2  # 4 for E2 (wall + 2p+1 nodes? -> T = p+1 nodes + wall = 4)

    fa = [Symbol(f"fa_{k}") for k in range(4)]
    ia = [Symbol(f"ia_{k}") for k in range(4)]

    interior = derive_interior_interp(p, y)

    # ----- LEFT wall (deltas [-psi, 0, 1, 2]) -----
    deltas_left = build_cut_cell_interp_deltas(T, psi, left=True)
    left_rows: list[InterpRow] = []
    # row 0: closest node is the wall (delta -psi) → target = -psi + y
    left_rows.append(
        InterpRow(
            row_index=0,
            coeffs=derive_cut_cell_interp_row(
                deltas_left,
                q,
                deltas_left[0] + y,  # -psi + y
                y,
                [(0, T - 1, ia[0], fa[0]), (1, T - 1, ia[1], fa[1])],
            ),
            fa_syms=[fa[0], fa[1]],
            ia_syms=[ia[0], ia[1]],
        )
    )
    # row 1: closest node is node0 (delta 0) → target = y
    left_rows.append(
        InterpRow(
            row_index=1,
            coeffs=derive_cut_cell_interp_row(
                deltas_left,
                q,
                deltas_left[1] + y,  # 0 + y
                y,
                [(0, T - 1, ia[2], fa[2]), (1, T - 1, ia[3], fa[3])],
            ),
            fa_syms=[fa[2], fa[3]],
            ia_syms=[ia[2], ia[3]],
        )
    )

    # ----- RIGHT wall (deltas [-2, -1, 0, psi]) -----
    deltas_right = build_cut_cell_interp_deltas(T, psi, left=False)
    right_rows: list[InterpRow] = []
    # row 0: closest node is the wall (delta psi, col T-1) → target = psi + y
    right_rows.append(
        InterpRow(
            row_index=0,
            coeffs=derive_cut_cell_interp_row(
                deltas_right,
                q,
                deltas_right[T - 1] + y,  # psi + y
                y,
                [(T - 1, 0, ia[0], fa[0]), (T - 2, 0, ia[1], fa[1])],
            ),
            fa_syms=[fa[0], fa[1]],
            ia_syms=[ia[0], ia[1]],
        )
    )
    # row 1: closest node is node@col T-2 (delta 0) → target = y
    right_rows.append(
        InterpRow(
            row_index=1,
            coeffs=derive_cut_cell_interp_row(
                deltas_right,
                q,
                deltas_right[T - 2] + y,  # 0 + y
                y,
                [(T - 1, 0, ia[2], fa[2]), (T - 2, 0, ia[3], fa[3])],
            ),
            fa_syms=[fa[2], fa[3]],
            ia_syms=[ia[2], ia[3]],
        )
    )

    return InterpResult(
        interior=interior,
        left_rows=left_rows,
        right_rows=right_rows,
        interp_P=interior.runtime_p,
        interp_T=T,
        y=y,
        psi=psi,
    )


# ---------------------------------------------------------------------------
# Derivative side (floating from interp duality; dirichlet = legacy closure)
# ---------------------------------------------------------------------------
def derive_floating_coeffs(y: Symbol, psi: Symbol) -> list:
    """The R*T=12 floating-derivative coefficients as Expr in fa_0..fa_5, psi.

    Rows 0,1 are the ``d/dy|0`` of the left interp wall rows (SHARING fa[0..3]);
    row 2 is floating-only (fa[4],fa[5]).  This is the *interp → derivative*
    duality made concrete (Mathematica polyScheme: ``D[…,y]/.y->0``).
    """
    T = 4
    fa = [Symbol(f"fa_{k}") for k in range(6)]
    ia = [Symbol(f"ia_{k}") for k in range(4)]
    deltas = build_cut_cell_interp_deltas(T, psi, left=True)

    coeffs: list = []
    fa_pairs = [(fa[0], fa[1]), (fa[2], fa[3]), (fa[4], fa[5])]
    # The interp target offset is irrelevant for the derivative (d/dy|0 kills
    # the y^0 / ia part), so any consistent target works; use -psi+y / y / 1+y.
    targets = [deltas[0] + y, deltas[1] + y, deltas[2] + y]
    for (fa_a, fa_b), tgt in zip(fa_pairs, targets):
        row = derive_cut_cell_interp_row(
            deltas,
            1,
            tgt,
            y,
            [(0, T - 1, ia[0], fa_a), (1, T - 1, ia[1], fa_b)],
        )
        coeffs.extend(derivative_from_interp(row, y))
    return coeffs


def dirichlet_coeffs_symbolic(psi: Symbol) -> list:
    """REFERENCE ORACLE: the (R-1)*T=8 dirichlet coeffs transcribed from the
    committed C++ nbs_dirichlet (== Mathematica e2D, PolyE2_1.nb p.1-2).

    This is a golden value kept for cross-checking, NOT the generation path —
    ``derive_dirichlet_coeffs`` (below) DERIVES the same coefficients from the
    poly recipe.  ``test_interp`` asserts the two agree symbolically, and the
    derived form is what ``build_polyE2_1_spec`` feeds to codegen.  Reproduces
    the hard-coded test array (psi=0.001, da=[3/25,13/100,7/50]) to 1e-12.
    """
    da = [Symbol(f"da_{k}") for k in range(3)]
    t6 = Rational(1) / (1 + psi)
    t7, t13, t19 = da[0], da[1], da[2]
    t8 = -1 + t7
    t18 = 2 * psi
    t20 = 3 * t19
    t21 = 2 * psi * t19
    t22 = 1 + t18 + t20 + t21
    t23 = Rational(1) / t22
    t14 = 2 * psi * t13
    t15 = 3 * t13 * t7
    t16 = 2 * psi * t13 * t7
    t29 = -t13
    t25 = 2 + psi
    t43 = -1 + t19
    return [
        Rational(1, 2) * t6 * t8,
        (-1 - 2 * psi + t13 + t14 + t15 + t16 - 3 * t7 - 2 * psi * t7)
        * Rational(1, 2)
        * t23,
        (
            -2 * psi * t13
            - 3 * t19
            - 2 * psi * t19
            + t29
            + 3 * t7
            + 2 * psi * t7
            - 3 * t13 * t7
            - 2 * psi * t13 * t7
        )
        * t23
        - Rational(1, 2) * t25 * t6 * t8,
        (
            t13
            + t14
            + t15
            + t16
            + t20
            + t21
            - 2 * t7
            + 3 * t19 * t7
            + 2 * psi * t19 * t7
        )
        * Rational(1, 2)
        * t23,
        Rational(1, 2) * t43 * t6,
        (-1 + t13) * Rational(1, 2),
        t29 - Rational(1, 2) * t25 * t43 * t6,
        (t13 + t19) * Rational(1, 2),
    ]


# ---------------------------------------------------------------------------
# Dirichlet closure — DERIVED from the poly recipe (not transcribed)
# ---------------------------------------------------------------------------
def _poly_deriv_row(deltas: list, target, y: Symbol, free_a, free_b) -> list:
    """One derivative-stencil row via the interp recipe (polyScheme: d/dy|0).

    Reuses derive_cut_cell_interp_row with arbitrary free symbols so the same
    per-row construction as derive_floating_coeffs can be fed ``da`` or the
    conservation DOF.  ``free_a`` is the zero-wall variant's free param,
    ``free_b`` the zero-node0 variant's (matching derive_floating_coeffs).
    """
    ia = [Symbol("_dir_ia0"), Symbol("_dir_ia1")]  # killed by d/dy|0
    T = len(deltas)
    interp_row = derive_cut_cell_interp_row(
        deltas, 1, target, y,
        [(0, T - 1, ia[0], free_a), (1, T - 1, ia[1], free_b)],
    )
    return [cancel(c) for c in derivative_from_interp(interp_row, y)]


def _conservation_X(psi: Symbol, d0, d1, d2):
    """Solve the SBP/conservation coupling for the f'[2] row's non-wall DOF X.

    The constraint is BILINEAR in (X, da_2):

        X*(1 + 2*psi) + X*d2*(3 + 2*psi)
          = d0*d1*(3 + 2*psi) - d0*(3 + 2*psi) + d1*(1 + 2*psi) + d2*(3 + 2*psi)

    Solving the linear-in-X relation introduces the denominator
    (1 + 2*psi + (3 + 2*psi)*d2) — da_2 in the denominator — and the da_i*da_j
    cross terms seen in the committed nbs_dirichlet.  THIS is the mechanism
    behind the Dirichlet bilinearity.

    Provenance / honest status: this is the E2 specialisation of Mathematica
    ``conservationCutCell`` (taylor.wl:763-768), which forms ``m[1,1]*w[1]+1==0``
    with the column-sum constraints — itself bilinear in (weights x coeffs), the
    SAME (X x d2) bilinearity.  The closed form below was validated two ways
    (symbolic cancel()==0 vs the e2D oracle for all 8 coeffs, and exact on random
    rational samples) but NOT re-derived weight-by-weight from the SBP norm: the
    defining ``PolyE2x1.wl`` cut-cell weights are not in the repo.  Fully closing
    this = porting conservationCutCell's left-matrix + cut-cell-norm weights to
    sympy (scheme-agnostic follow-up; see docs/poly_interp_design.md §6.6).
    """
    X = Symbol("_dir_X")
    lhs = X * (1 + 2 * psi) + X * d2 * (3 + 2 * psi)
    rhs = (
        d0 * d1 * (3 + 2 * psi)
        - d0 * (3 + 2 * psi)
        + d1 * (1 + 2 * psi)
        + d2 * (3 + 2 * psi)
    )
    (sol,) = solve(lhs - rhs, X)
    return cancel(sol)


def derive_dirichlet_coeffs(psi: Symbol) -> list:
    """DERIVE the (R-1)*T=8 Dirichlet-derivative coefficients in da_0..da_2, psi.

    Generation-path counterpart to the ``dirichlet_coeffs_symbolic`` oracle:
    every value is produced by the poly recipe (derive_cut_cell_interp_row →
    derivative_from_interp) plus ONE conservation solve, rather than copied from
    the committed C++.  Column / row order matches the C++ exactly.

    * row 0 (f'[2]) is BILINEAR: free DOFs are (X from conservation, da_0 wall blend).
    * row 1 (f'[3]) is purely linear: free DOFs are (da_1 shared, da_2 wall blend).
    """
    y = Symbol("y")
    da = [Symbol(f"da_{k}") for k in range(3)]
    d0, d1, d2 = da
    T = 4
    deltas = build_cut_cell_interp_deltas(T, psi, left=True)  # [-psi, 0, 1, 2]

    X = _conservation_X(psi, d0, d1, d2)
    row0 = _poly_deriv_row(deltas, deltas[1] + y, y, free_a=X, free_b=d0)
    row1 = _poly_deriv_row(deltas, deltas[1] + y, y, free_a=d1, free_b=d2)
    return row0 + row1


def build_polyE2_1_spec():
    """Assemble the full ``StencilGenSpec`` for polyE2_1 (interp + derivative).

    Returns a spec that ``generate_stencil_cpp`` turns into a file numerically
    equivalent to the committed ``src/stencils/polyE2_1.cpp``.
    """
    from stencil_gen.codegen import StencilGenSpec

    res = derive_poly_interp(p=1, q=1)
    y, psi = res.y, res.psi

    floating = derive_floating_coeffs(y, psi)
    # codegen drops row 0 (the first T entries) of dirichlet_coeffs, so prepend
    # a zero row that will be sliced off.  Use the DERIVED closure (not the
    # transcribed oracle) so the generation path contains no copied-from-C++ block.
    dirichlet = [Rational(0)] * res.interp_T + derive_dirichlet_coeffs(psi)

    return StencilGenSpec(
        name="polyE2_1",
        P=1,
        R=3,
        T=res.interp_T,
        X=0,
        derivative_order=1,
        is_uniform=False,
        param_arrays={"fa": 6, "da": 3, "ia": 4},
        interior_coeffs=[Rational(-1, 2), Rational(0), Rational(1, 2)],
        floating_coeffs=floating,
        dirichlet_coeffs=dirichlet,
        has_interp=True,
        interp_P=res.interp_P,
        interp_T=res.interp_T,
        interp_interior_pos=res.interior.runtime_pos,
        interp_interior_neg=res.interior.runtime_neg,
        interp_wall_left_rows=[r.coeffs for r in res.left_rows],
        interp_wall_right_rows=[r.coeffs for r in res.right_rows],
    )
