# Polynomial-Interpolation Cut-Cell Stencils — Design & Implementation Spec

**Status:** ready to implement. **North star:** regenerate `src/stencils/polyE2_1.cpp`
from sympy so it is numerically equivalent to the committed hand-port and `t-polyE2_1`
still passes (47/48 → 48/48 contributions unchanged elsewhere).

**Audience:** an engineer implementing the migration directly from this document.

**Scope:** add the polynomial-interpolation derivation to the sympy framework
(`scripts/stencil_gen/`) plus the codegen that emits real `interp_interior`/`interp_wall`
bodies and a real `query_interp`. The derivative side (`nbs_floating`/`nbs_dirichlet`,
`interior`) is *already* produced correctly by `generate_nbs_method`; this work is purely
additive and gated behind `None`-default fields so no existing stencil/test changes.

This spec is grounded in the live tree (file:line are current):
- C++ north star: `src/stencils/polyE2_1.cpp`, test `src/stencils/polyE2_1.t.cpp`.
- Orchestration contract: `src/stencils/stencil.hpp:213-270` (`stencil::interp`).
- Mathematica routines: `taylor.wl:824-998` (+ supporting `40-57`, `401-408`, `511-517`).
- Worked E2 derivation: `PolyE2_1.nb.pdf` (e2D pp.1-2, e2I pp.2-6).
- Validated sympy prototype: `scripts/stencil_gen/_proto_interp.py` (7/7 checks pass).
- Reuse targets: `taylor_system.py`, `interior.py`, `boundary.py`, `codegen.py`,
  `printer.py`, `_util.py:solve_linear`.

---

## 1. Overview — the novel idea

### 1.1 One polynomial, two extractions

Define an **interpolation polynomial** stencil that approximates the field value at a
fractional offset `y` from grid node `i`:

```
f[i+y] = Σ_j  gamma[i,j](y, psi, params) · f[node_j]          (the INTERP polynomial)
```

Each coefficient `gamma[i,j]` is a polynomial in `y` (degree ≤ scheme order) whose
coefficients depend on the cut fraction `psi` and the scheme's free parameters. From the
**same** polynomial set we take two products:

1. **Interpolation stencil** — evaluate `gamma[i,j](y)` at the runtime `y`. Emitted by
   `interp_interior` / `interp_wall`. Pure value interpolation, no `h`, no antisymmetry.
2. **Derivative stencil** — `f'[i] = (1/h) · d/dy gamma[i,j](y)|_{y=0}`, because
   `d/dy f[i+y]|_{y=0} = f'[i]`. Emitted by `nbs_floating` / `nbs_dirichlet`.

### 1.2 Verified E2 example

Interior interpolation on nodes `{-1, 0, 1}` (Taylor-match `f[i+y]`, RHS `y^k/k!`):

```
gamma[-1] = (y² − y)/2     gamma[0] = 1 − y²     gamma[1] = (y² + y)/2
d/dy|_{y=0}:   −1/2                0                   +1/2     ← central difference ✓
```

(`PolyE2_1.nb.pdf` p.6 `int`; reproduced symbolically in `_proto_interp.py` CHECK 1–2:
`d/dy|0` byte-equals `derive_interior(0,1,1)`.)

### 1.3 One consistent free-parameter set: `fa` / `da` / `ia`

The raw poly free parameters (`alpha`, `beta`) are renamed by `constrainInterp`
(`taylor.wl:987-998`) to the scheme parameters. Every free DOF enters via the *blend*
`polyBlend` (`taylor.wl:892`): a raw free coefficient `g → u + y·v`. Therefore:

- **`fa` (floating)** — the `v` part, multiplying `y`. It **survives** `d/dy|_{y=0}`, so the
  SAME `fa` set feeds **both** the interp wall rows and the floating derivative rows.
- **`ia` (interp-only)** — the `u` part, sitting in the `y^0` term. It **vanishes** under
  `d/dy|_{y=0}`, so it touches only the interpolation value, never the derivative.
- **`da` (dirichlet)** — the floating derivative blend renamed `fa → da` for the Dirichlet
  closure (`polyScheme` does `D[…,y]/.y->0`; the dirichlet flavor feeds `nbs_dirichlet`).

Concrete witness (`PolyE2_1.nb.pdf` p.2, left row i=1, interior column):

```
gamma[1,1](y) = (1 + psi − y + ia0 + y·fa0) / (2(1+psi))
   value (y=0)    = (1 + psi + ia0) / (2(1+psi))      ← ia only
   d/dy|0         = (fa0 − 1)       / (2(1+psi))       ← fa only
```

This is the entire migration: build the interp polynomials once; the derivative falls out
by `d/dy|0`; the three array splits (`fa[6]`, `da[3]`, `ia[4]`) are just three renamings of
the same blended free parameters.

### 1.4 The E2 C++ array contract (the target)

| MMA symbol | meaning                      | `d/dy`? | params           | C++ method                     | array            |
|------------|------------------------------|---------|------------------|--------------------------------|------------------|
| `e2I`      | interp polynomials           | no      | `fa[0..5]`,`ia`  | `interp_interior`,`interp_wall`| `ia[4]`; shares `fa` |
| `e2F`      | floating derivative rows     | yes,y=0 | `fa[0..5]`       | `nbs_floating`                 | `fa[6]` (3 rows×2) |
| `e2D`      | dirichlet derivative rows    | yes,y=0 | `da[0..2]`       | `nbs_dirichlet`                | `da[3]`          |

Packed-alpha split in `stencil.cpp` ("E2-poly"): `fa = alpha[0:6]`, `da = alpha[6:9]`,
`ia = alpha[9:13]` — total 13 reals; `make_polyE2_1(fa, da, ia)` is also called with three
separate Lua tables (`floating_alpha`, `dirichlet_alpha`, `interpolant_alpha`). **No C++ or
CMake change is required** — `polyE2_1` is already fully wired.

---

## 2. New sympy module: `stencil_gen/interp.py`

A new file mirroring the style of `boundary.py` / `interior.py`. It reuses the Vandermonde
machinery and adds the interp RHS, the cut-cell wall rows, the `d/dy|0` map, and the
`constrain_interp` renaming.

### 2.1 The single structural change: interp RHS helper (in `taylor_system.py`)

The Vandermonde matrix `V[k,j] = (j−i)^k/k!` (`taylor_system.py:38-44`) is identical for
derivative and interpolation. **Only the RHS differs.** Add to `taylor_system.py`:

```python
def _interp_rhs(n_eqs: int, y) -> Matrix:
    """Column vector rhs[k] = y**k / k! — the Taylor-evaluation functional at offset y.

    Derivation: f[i+y] = Σ_k f^(k)[i] y^k/k!, so to reproduce f[i] (and its moments)
    the coefficients must satisfy Σ_j c_j (j-i)^k = y^k, i.e. rhs[k] = y^k/k!.
    This is the interpolation analogue of _unit_rhs (taylor_system.py:10).
    factorial is already imported (taylor_system.py:7)."""
    return Matrix(n_eqs, 1, lambda k, _: y**k / factorial(k))


def build_interp_system(i: int, t: int, q: int, y) -> tuple[Matrix, Matrix]:
    """Interpolation analogue of build_taylor_system. Reuses V verbatim; swaps the
    RHS to the eval-at-y functional. Returns (V, rhs)."""
    V, _ = build_taylor_system(i, t, q, nu=1)   # reuse the matrix only
    return V, _interp_rhs(q + 1, y)
```

(`build_taylor_system` builds `V` independent of `nu`; reusing it for `V` is exact.)

### 2.2 Dataclasses

```python
@dataclass
class InterpInterior:
    """Interior interpolation polynomials (Mathematica interiorInterp, taylor.wl:832-835).
    coeffs has length 2*p+1, each a polynomial in y. For p=1: [(y²-y)/2, 1-y², (y²+y)/2].
    runtime_p / runtime_coeffs hold the *shipped* low-order form (see §6.1): a 2-point
    linear interp selected by sign(y), which is what query_interp().p and interp_interior
    actually emit."""
    coeffs: list           # full 2p+1 polynomial (used to DERIVE the interior derivative)
    p: int
    runtime_p: int         # = query_interp().p  (2 for E2)
    runtime_pos: list      # length runtime_p, the y>0 branch weights (e.g. [1-y, y])
    runtime_neg: list      # length runtime_p, the y<=0 branch weights (e.g. [-y, 1+y])


@dataclass
class InterpRow:
    """One cut-cell interpolation wall row (Mathematica polyInterpSchemes output,
    taylor.wl:901-933). coeffs is length T, each Expr in (y, psi, fa_k, ia_k).
    Column order matches C++: left -> [wall, node, node, node]; right -> [node,node,node,wall]."""
    row_index: int         # i (0..R-2): cell-distance wall->closest node
    coeffs: list
    fa_syms: list          # the fa (floating) symbols used in this row's y^1 terms
    ia_syms: list          # the ia (interp-only) symbols used in this row's y^0 terms


@dataclass
class InterpResult:
    """Full interp side of a poly stencil — the codegen feed."""
    interior: InterpInterior
    left_rows: list        # R-1 InterpRow for the left/interior wall (cases 0..R-2)
    right_rows: list       # R-1 InterpRow for the right wall (mirror, independently derived)
    interp_P: int          # query_interp().p (= interior.runtime_p)
    interp_T: int          # query_interp().t (= T)
    y: Symbol
    psi: Symbol
```

### 2.3 Functions

```python
def derive_interior_interp(p: int, y: Symbol) -> InterpInterior:
    """Mathematica interiorInterp (taylor.wl:832-835). Solve Σ_{j=-p..p} gamma_j f[i+j]
    = f[i+y] on a centered uniform 2p+1 stencil via build_interp_system over deltas
    {-p..p}. Entries are exact in y (no psi), so Matrix.solve + cancel suffices (no
    solve_in_field). Returns InterpInterior with the full 2p+1 polynomial AND the shipped
    2-point linear runtime form (the bracketing-pair Lagrange weights selected by sign(y);
    see §6.1). For p=1: full=[(y²-y)/2, 1-y², (y²+y)/2]; runtime_pos=[1-y, y],
    runtime_neg=[-y, 1+y]."""


def derivative_from_interp(interp_coeffs: list, y: Symbol, nu: int = 1) -> list:
    """THE NOVEL CORE. f^(nu)[i] = d^nu/dy^nu f[i+y]|_{y=0}. Two lines:
        return [cancel(diff(c, y, nu).subs(y, 0)) for c in interp_coeffs].
    Verified: [(y²-y)/2, 1-y², (y²+y)/2] -> [-1/2, 0, 1/2]. Maps Mathematica polyScheme's
    `MapAt[D[#,y]&, ...] /. y->0` (taylor.wl:954). Because each coeff is (poly in y, linear
    in ia=u and fa=v), d/dy|0 kills the ia (y^0) part and keeps fa (y^1)."""


def solve_interp_row(
    i: int, T: int, q: int, y: Symbol, psi: Symbol,
    zeroed_col: int, free_symbols: list,
) -> list:
    """ONE variant of a cut-cell wall row (one schemeCoefficients call inside
    polyInterpSchemes). Cut-cell analogue of boundary.solve_boundary_row (boundary.py:27),
    but with two differences:
      (1) RHS = _interp_rhs(q+1, y) instead of _unit_rhs;
      (2) deltas are cut-cell positions, not 0..T-1 (see §5.2: [-psi, 0, 1, 2] for E2 left).
    Drops `zeroed_col` (polyZero, taylor.wl:885-890), partitions remaining columns into
    determined (q+1) + free (the rest assigned `free_symbols`), and solves via Matrix.solve
    over QQ(psi)[y] with cancel (y never enters a denominator for E2; if a future scheme
    needs it, route through temo.solve_in_field). Returns length-T list of Expr."""


def derive_cut_cell_interp(
    p: int, q: int, psi: Symbol, y: Symbol,
    fa_syms: list, ia_syms: list,
    left: bool = True,
) -> list:
    """Mathematica polyInterpBoundary (taylor.wl:973-984): map polyInterpSchemes over the
    R-1 wall rows. For each row i:
      - run solve_interp_row TWICE with the two zeroed columns (polyZero variant k=1,2);
      - apply the polyBlend: each raw free param -> (ia_k + y*fa_k) (poly_constrain_free);
      - simpleAverage the two variant results coeff-by-coeff (taylor.wl:899);
    Returns a list of length-T coeff lists (NOT yet renamed). The SAME (ia_k, fa_k) must
    feed both variants so the average stays linear in them. Build right rows by passing
    left=False and the right-anchored cut-cell deltas."""


def constrain_interp(
    rows: list, fa_syms: list, ia_syms: list, left: bool,
) -> list:
    """Mathematica constrainInterp (taylor.wl:987-998). Renames the raw blend symbols to
    the final scheme symbols in the order the C++ arrays expect:
      - LEFT: ascending column order   -> v->fa_k, u->ia_k for k=0,1,2,...
      - RIGHT: descending/right-anchored order (taylor.wl:991-994 SortBy[-First...]).
    THE CRITICAL CORRECTNESS POINT — the assignment order picks which fa[k]/ia[k] lands in
    which coefficient and must match polyE2_1.cpp exactly (see §6.3). Returns rows with
    fa_k/ia_k substituted; symbols already named fa_0, ia_0, ... so printer.build_symbol_map
    emits fa[0], ia[0] with no remap."""


def derive_poly_interp(p: int, q: int) -> InterpResult:
    """Top-level orchestrator for the interp side. Allocates psi, y, the shared fa symbols
    (same names as the derivative scheme's renamed floating params) and new ia symbols, then:
      interior   = derive_interior_interp(p, y)
      left_rows  = constrain_interp(derive_cut_cell_interp(..., left=True),  ..., left=True)
      right_rows = constrain_interp(derive_cut_cell_interp(..., left=False), ..., left=False)
    Wraps everything in InterpResult. For E2: p=1, q=2, interp_T=4, interp_P=2."""
```

### 2.4 Reuse summary

| New thing | Reuses |
|---|---|
| `build_interp_system` | `build_taylor_system` (matrix) + new `_interp_rhs` |
| `derive_interior_interp` | `build_interp_system`, `Matrix.solve`, `cancel`; cross-check vs `derive_interior` |
| `solve_interp_row` | `boundary.solve_boundary_row` structure; `temo.build_cut_cell_deltas` for the cut-cell layout; `_util.solve_linear` / `Matrix.solve` |
| `derivative_from_interp` | `sympy.diff`, `cancel` — output must equal `generate_nbs_method`'s coeffs |
| `constrain_interp` | `printer.build_symbol_map` naming convention (`fa_0` → `fa[0]`) |

### 2.5 Mathematica → sympy routine map (full)

| Mathematica (`taylor.wl`) | sympy (`interp.py`) | note |
|---|---|---|
| `interiorInterp` 825-835 | `derive_interior_interp` | RHS swap only |
| `boundaryInterp` 837-843 | (folded into `solve_interp_row`) | uniform precursor; optional standalone |
| `interpSymbol` 861-863 | — | dropped; pass `y` explicitly |
| `polyZero` 885-890 | `zeroed_col` arg to `solve_interp_row` | 0-based column index |
| `polyBlend` 892-893 | the `u + y*v` substitution inside `derive_cut_cell_interp` | the fa/ia split |
| `polyConstrainFree` 894-896 | per-row blend application | |
| `simpleAverage` 899 | `(a+b)/2` element-wise in `derive_cut_cell_interp` | yields the `0.5 *` factors |
| `polyInterpSchemes` 901-933 | `solve_interp_row` ×2 + blend + average | central routine |
| `polyScheme`/`polyBoundary` 936-971 | `derivative_from_interp` | `D[#,y]/.y->0` |
| `polyInterpBoundary` 973-984 | `derive_cut_cell_interp` | map over rows |
| `constrainInterp` 987-998 | `constrain_interp` | rename + ordering |
| `bIndex`/`rIndex` 401-408, `stencilColRange` 515-517 | cut-cell deltas in `solve_interp_row` | wall = col 0 (left) |

---

## 3. Codegen changes (`codegen.py` + `printer.py`)

All additive; gated on new `None`-default fields so every existing spec keeps emitting the
current no-op interp stubs.

### 3.1 New `StencilGenSpec` fields (after `interp_T`, `codegen.py:261`)

```python
    # Interp bodies. None => emit the existing no-op stubs (back-compat).
    interp_interior_pos: list | None = None   # y>0 branch weights, length interp_P
    interp_interior_neg: list | None = None   # y<=0 branch weights, length interp_P
    interp_wall_left_rows: list | None = None   # (R-1) rows, each length T, Expr in y/psi/fa/ia
    interp_wall_right_rows: list | None = None  # (R-1) rows, mirror
```

`has_interp` continues to gate `query_interp`; the new lists gate the real bodies.

### 3.2 `printer.py` — handle symbol `y`

`Symbol('y')` already prints as `y` via the `_print_Symbol` fallback (`printer.py:62-65`),
so this is defensive only. Add in `build_symbol_map` right before the `h` mapping
(`printer.py:94`):

```python
    smap[Symbol("y")] = "y"
```

No `/h` is applied to interp coeffs, so there is no h-division concern here.

### 3.3 `_emit_query_methods` — thread `printer`, conditionally emit real bodies

Change the signature to `def _emit_query_methods(spec, printer):` and update the call site
in `generate_stencil_cpp` (`codegen.py:557`) to `_emit_query_methods(spec, printer)`.
`query_max`/`query`/`query_interp` (`codegen.py:355-377`) are unchanged. Replace the no-op
block (`codegen.py:379-389`) with:

```python
    lines.append("")
    if spec.interp_interior_pos is not None:
        lines.append(generate_interp_interior_method(
            spec.interp_interior_pos, spec.interp_interior_neg, printer))
    else:
        lines.append(f"{indent}std::span<const real> interp_interior(real, std::span<real> c) const {{ return c; }}")
    lines.append("")
    if spec.interp_wall_left_rows is not None:
        lines.append(generate_interp_wall_method(spec, printer))
    else:
        lines.extend([
            f"{indent}std::span<const real> interp_wall(int, real, real, std::span<real> c, bool) const",
            f"{indent}{{", f"{indent8}return c;", f"{indent}}}",
        ])
```

### 3.4 `generate_interp_interior_method` — new helper

Match `polyE2_1.cpp:46-56` exactly. Two sign branches (the `y>0` branch corresponds to the
`st1/st2` swap at `stencil.hpp:225-227`), `subspan(0, interp_P)`, **no h, no CSE** (literals
direct for p=2):

```python
def generate_interp_interior_method(pos, neg, printer) -> str:
    indent, indent8 = "    ", "        "
    lines = [f"{indent}std::span<const real> interp_interior(real y, std::span<real> c) const",
             f"{indent}{{", f"{indent8}if (y > 0) {{"]
    for j, c in enumerate(pos):
        lines.append(f"{indent8}    c[{j}] = {printer.doprint(c)};")
    lines.append(f"{indent8}}} else {{")
    for j, c in enumerate(neg):
        lines.append(f"{indent8}    c[{j}] = {printer.doprint(c)};")
    lines += [f"{indent8}}}", f"{indent8}return c.subspan(0, {len(pos)});", f"{indent}}}"]
    return "\n".join(lines)
```

For E2: `pos = [1 - y, y]` prints as `1 + -1 * y` and `y`; `neg = [-y, 1 + y]` prints as
`-1 * y` and `1 + y`. (Printer renders `1 - y` as `1 + -1 * y` — matches the committed file.)

### 3.5 `generate_interp_wall_method` — new helper (the big one)

Match `polyE2_1.cpp:58-139`. Structure is **inverted** vs `nbs`: `if (right) {...} else {...}`
outermost, `switch (i)` inside each branch, CSE applied **per branch over the whole branch's
coefficient list** (the `t5,t6,...` pool is shared across `case 0` and `case 1`):

```python
def generate_interp_wall_method(spec, printer) -> str:
    indent, indent8 = "    ", "        "
    out = [f"{indent}std::span<const real>",
           f"{indent}interp_wall(int i, real y, real psi, std::span<real> c, bool right) const",
           f"{indent}{{",
           f"{indent8}if (right) {{"]
    out += _emit_interp_switch(spec.interp_wall_right_rows, printer, depth=3)
    out.append(f"{indent8}}} else {{")
    out += _emit_interp_switch(spec.interp_wall_left_rows, printer, depth=3)
    out += [f"{indent8}}}", f"{indent8}return c.subspan(0, {spec.interp_T});", f"{indent}}}"]
    return "\n".join(out)


def _emit_interp_switch(rows, printer, depth) -> list:
    ind = "    " * depth
    flat = [e for row in rows for e in row]          # CSE over the whole branch
    repls, reduced = apply_cse(flat, start=5)        # start=5 matches the hand file
    lines = [f"{ind}const real {sym.name} = {printer.doprint(e)};" for sym, e in repls]
    lines.append(f"{ind}switch (i) {{")
    k = 0
    for ci, row in enumerate(rows):
        lines.append(f"{ind}case {ci}:")
        for j in range(len(row)):
            lines.append(f"{ind}    c[{j}] = {printer.doprint(reduced[k])};")
            k += 1
        lines.append(f"{ind}    break;")
    lines.append(f"{ind}}}")
    return lines
```

Behavioral asymmetry vs `nbs` (must NOT be copied from `generate_nbs_method`):
- **No `/h`** — interp is value-based.
- **No `*= -1` + `std::ranges::reverse`** for `right` — the right branch is a *separately
  derived mirror stencil* (right wall), already correct as emitted.
- **Column order moves the wall**: left → `c[0]` is the wall (carries `1/(1+psi)`),
  `c[1..T-1]` interior nodes increasing-x; right → `c[T-1]` is the wall, `c[0..T-2]` interior.
- **`1/(1+psi)`** is produced automatically by `StencilCodePrinter._print_Pow` (exp==-1,
  `printer.py:28-30`) acting on `(1+psi)**-1` and hoisted by CSE as `t16/t17` (right) or
  `t5/t6` (left) — no special handling.
- **No `default:`** in the switch (the hand file omits it; emit exactly `R-1` cases).

### 3.6 Assembly (`build_polyE2_1_spec`)

Add a thin orchestrator (in `interp.py` or a small `poly_assemble.py`) wired from
`__main__.py`'s `generate` stub (`__main__.py:25-30`):

1. Derive the derivative side (existing path) → flatten the floating matrix into
   `floating_coeffs` (R·T) and the dirichlet matrix into `dirichlet_coeffs` (R·T; codegen
   drops row 0 at `codegen.py:482`). Use the **shared** `fa` symbols.
2. `res = derive_poly_interp(p=1, q=2)` → populate `interp_interior_pos/neg`,
   `interp_wall_left_rows`, `interp_wall_right_rows`, `has_interp=True`,
   `interp_P=res.interp_P` (2), `interp_T=res.interp_T` (4).
3. `param_arrays = {"fa": 6, "da": 3, "ia": 4}`; `P=1, R=3, T=4, X=0`; `is_uniform=False`.
4. `generate_stencil_cpp(spec)` → `output/polyE2_1.cpp`.

`_emit_struct_preamble` (`codegen.py:315-325`, multi-array ctor) and `_emit_factory`
(`codegen.py:527-538`) already emit the 3-arg `fa/da/ia` constructor + factory; `LUA_KEY_MAP`
(`codegen.py:583-588`) already maps `fa/da/ia` → `floating_alpha/dirichlet_alpha/interpolant_alpha`.

---

## 4. Golden / test plan

Run sympy tests from `scripts/stencil_gen` with `SYMPY_CACHE_SIZE=50000 uv run pytest`.

### 4.1 Symbolic golden + derivation correctness — `tests/test_interp.py` (new)

| test name | assertion |
|---|---|
| `test_interior_interp_golden` | `derive_interior_interp(1, y).coeffs == [(y²-y)/2, 1-y², (y²+y)/2]` via `cancel(a-e)==0` (`PolyE2_1.nb.pdf` p.6) |
| `test_interior_runtime_2pt` | `runtime_pos == [1-y, y]`, `runtime_neg == [-y, 1+y]` |
| `test_derivative_from_interp_central` | `derivative_from_interp([(y²-y)/2,1-y²,(y²+y)/2], y) == [-1/2, 0, 1/2]` and equals `full_gamma_array(derive_interior(0,1,1))` |
| `test_left_row0_golden` | left wall row i=0 coeffs match `PolyE2_1.nb.pdf` p.2 `gamma[1,·]` (column-reordered to C++ `[wall,node,node,node]`); `cancel(diff)==0` per column |
| `test_left_row1_golden` | left wall row i=1 matches p.3 `gamma[2,·]` |
| `test_right_rows_golden` | right rows match p.4-5 `gamma[n-1,·]`, `gamma[n,·]` |
| `test_ia_vanishes_under_ddy` | for every wall coeff, `ia_k ∉ derivative_from_interp(row).free_symbols`; `fa_k ∈` |
| `test_value_deriv_decoupling` | `gamma|y=0` free syms ⊂ {ia,psi}; `d/dy|0` free syms ⊂ {fa,psi}; O(y²) remainder == 0 (`_proto_interp.py` CHECK 4) |
| `test_interp_exactness` | for each row deltas `δ_j`: `Σ_j c_j δ_j^k == y^k` for `k=0..q` (interp analogue of `conftest._check_taylor_accuracy`) |
| `test_duality_vs_nbs` | `derivative_from_interp(left_row_i)` (pre-`/h`) equals the corresponding `nbs_floating` row from the derivative path, for all i |

### 4.2 Codegen emission shape — `tests/test_codegen.py` (extend `poly_spec`)

Populate the new `interp_*` fields on the existing `poly_spec` (`test_codegen.py:506`), then
assert substrings in `generate_stencil_cpp(spec)`:

- `interp_info query_interp() const { return {2, 4}; }`
- `std::span<const real> interp_interior(real y, std::span<real> c) const`, `if (y > 0) {`,
  `return c.subspan(0, 2);`
- `interp_wall(int i, real y, real psi, std::span<real> c, bool right) const`, `if (right) {`,
  `switch (i) {`, `case 0:`, `case 1:`, `return c.subspan(0, 4);`
- `const real t5 =` (CSE present)
- **Negative**: `interp_wall` body contains no `/= h` and no `std::ranges::reverse`.

### 4.3 Numeric match vs the hard-coded C++ arrays

| oracle | source | check |
|---|---|---|
| dirichlet coeffs | `polyE2_1.t.cpp:63-70` (8 floats, `psi=0.001, h=1, da=[0.12,0.13,0.14]`) | regenerated `nbs_dirichlet` symbolic eval reproduces the array to `1e-12` (derivative path — should already pass) |
| interp_interior scan | `polyE2_1.t.cpp:249-265` (11 y-values) | `Σ v·bf(mesh) == bf(mesh[center]+y·h)`, `center=1-(y>0)` |
| interp_wall left/right ×4 | `polyE2_1.t.cpp:294-363` | each of left0/left1/right0/right1 reproduces the stated `bf(...)`; use fixed `fa=[0.1..0.6]`, `ia=[0.7,0.8,0.9,1.0]` for a deterministic diff vs the committed C++ formulas (`_proto_interp.py` CHECK 5 pattern; extend `tools/eval_e2_1.py` with `parse_interp_wall`) |

Important `y`-coordinate detail (from `stencil.hpp:250,264`): `interp_wall` receives the
**post-adjusted** `y` (the orchestrator adds `psi-1` on the left when `lc==ic`, `1-psi` on the
right when `rc==ic`). The symbolic rows must be written in that adjusted coordinate; the
deltas are `[-psi, 0, 1, 2]` (left) — see §6.2.

### 4.4 Capstone — regenerate and prove equivalence

`tests/test_codegen_poly.py` (new) or a marked test in `test_codegen.py`:

1. `spec = build_polyE2_1_spec(); code = generate_stencil_cpp(spec)`.
2. **Numeric equivalence** (not byte equivalence — CSE temp names/ordering may differ):
   parse both the regenerated `code` and the committed `src/stencils/polyE2_1.cpp` with the
   `tools/eval_e2_1.py` harness and assert agreement to `1e-12` across a sampled grid of
   `(y, psi, fa, ia, da)` for `interp_interior`, `interp_wall`, `nbs_floating`, `nbs_dirichlet`.
3. **Cross-method duality at the C++ level**: numerically differentiate the generated
   `interp_wall` w.r.t. `y` at `y=0` and assert it equals generated `nbs_floating`/`nbs_dirichlet`
   (after `/h`).
4. **The definition of done**: write `code` to `src/stencils/polyE2_1.cpp`, then
   `cmake --build build --target t-polyE2_1 && ctest --test-dir build -R t-polyE2_1` passes
   (all six TEST_CASEs unchanged). Optionally keep a `--update-golden` flag (sweeps' pattern)
   so the committed file is refreshed deliberately, not on every run.

---

## 5. Step-by-step implementation checklist (ordered, each independently verifiable)

1. **`_interp_rhs` + `build_interp_system`** in `taylor_system.py`.
   *Verify:* `build_interp_system(0, 3, 2, y)[1] == Matrix([1, y, y²/2])`.
2. **`derive_interior_interp`** in new `interp.py`.
   *Verify:* `.coeffs == [(y²-y)/2, 1-y², (y²+y)/2]`; `.runtime_pos == [1-y, y]`.
3. **`derivative_from_interp`**.
   *Verify:* central diff `[-1/2,0,1/2]`; equals `full_gamma_array(derive_interior(0,1,1))`.
4. **Cut-cell deltas + `solve_interp_row`** (single variant).
   *Verify:* with E2 left deltas `[-psi,0,1,2]` and one zeroed column, the resulting row is
   degree-1 exact: `Σ_j c_j δ_j^k == y^k` for `k=0,1`.
5. **`derive_cut_cell_interp`** (two variants + polyBlend + simpleAverage).
   *Verify:* left row i=0 matches `PolyE2_1.nb.pdf` p.2 `gamma[1,·]` symbolically (modulo
   column order); the `0.5 *` factors and `1/(1+psi)` denominators appear.
6. **`constrain_interp`** (left ascending / right descending ordering).
   *Verify:* the renamed rows reproduce the exact `fa[k]`/`ia[k]` indices in
   `polyE2_1.cpp` cases 0/1 for both branches (§6.3).
7. **`derive_poly_interp`** orchestrator + cross-check duality.
   *Verify:* `derivative_from_interp` of each left/right row equals the `nbs_floating` rows
   from the derivative path (`_proto_interp.py` CHECK 3 final cross-check).
8. **printer.py**: add `smap[Symbol("y")] = "y"`. *Verify:* `test_printer` still green;
   `printer.doprint(1 - y) == "1 + -1 * y"`.
9. **codegen.py**: add 4 `StencilGenSpec` fields; add `generate_interp_interior_method`,
   `generate_interp_wall_method`, `_emit_interp_switch`; thread `printer` into
   `_emit_query_methods` and update the call site.
   *Verify:* `tests/test_codegen.py` substring assertions (§4.2); existing codegen tests green.
10. **`build_polyE2_1_spec`** + wire `__main__.py generate polyE2_1`.
    *Verify:* `uv run python -m stencil_gen generate polyE2_1` writes `output/polyE2_1.cpp`.
11. **Numeric equivalence harness** (`tools/eval_e2_1.py` += `parse_interp_wall`/`parse_interp_interior`).
    *Verify:* regenerated vs committed agree to `1e-12` on all four methods.
12. **Capstone**: copy `output/polyE2_1.cpp` → `src/stencils/polyE2_1.cpp`;
    `cmake --build build --target t-polyE2_1 && ctest --test-dir build -R t-polyE2_1` passes.

---

## 6. Risks / open questions (and resolutions)

### 6.1 2-point linear vs 3-point quadratic interior interp (RESOLVE: match the C++)

Mathematica `interiorInterp` for E2 yields the 3-point quadratic `[(y²-y)/2, 1-y², (y²+y)/2]`.
But `polyE2_1.cpp:46-56` ships a **2-point linear** interp selected by `sign(y)`
(`query_interp().p == 2`), because `stencil::interp` (`stencil.hpp:225-227`) picks a 2-node
bracketing window via the `st1/st2` swap. **Resolution:** derive the full 3-point polynomial
(needed to derive the *interior derivative* `[-1/2,0,1/2]`), but emit the 2-point linear
runtime form (`runtime_pos/neg`) for `interp_interior`. `InterpInterior` carries both. This
is a deliberate divergence for runtime compatibility, not an error.

### 6.2 Cut-cell delta layout (RESOLVE: deltas are `[-psi, 0, 1, 2]`, y is post-adjusted)

The prior port-spec guidance used deltas `[0, psi, 1+psi, 2+psi]`; the prototype proved that
**wrong** (no exactness beyond k=1). The correct E2 left layout, reverse-engineered from the
test (`polyE2_1.t.cpp:294-309`) and confirmed in `_proto_interp.py`, is node x-positions
`[-psi, 0, 1, 2]` = `[wall, node0, node1, node2]`, with the interp target at `x = -psi + y`
where `y` is the **post-adjustment** value (`stencil::interp` already applied `±(psi-1)`).
Skipping that adjustment leaves a spurious `1-psi` residual. **Resolution:** build
`solve_interp_row` with these deltas; document the post-adjusted-`y` convention in the
docstring; assert §4.1 exactness against `Σ c_j δ_j^k = y^k`.

### 6.3 `constrain_interp` ordering is load-bearing (RESOLVE: cross-check against C++)

The left vs right rename order differs (`taylor.wl:987-990` ascending vs `991-994`
descending). It determines which `fa[k]`/`ia[k]` lands in which coefficient. Concretely
(`polyE2_1.cpp`): left case0 → `fa[0],fa[1],ia[0],ia[1]`; left case1 → `fa[2],fa[3],ia[2],ia[3]`;
the right branch mirrors these. Also note the shared `fa` set spans both: `fa[0..3]` appear in
both interp and floating-derivative rows, while `fa[4..5]` are floating-only (the third
derivative row has no interp wall row). **Resolution:** implement both orderings; the
capstone numeric-equivalence test (§4.4) is the gate. The prototype warns that a naive
"zero-and-average" without the correct variant column choice + sort does NOT reproduce the
golden basis combination.

### 6.4 Solver field for the `(psi, y)` 2-variable solve (RESOLVE: Matrix.solve + cancel)

E2 entries are rational in `psi` and polynomial in `y`; `y` never enters a denominator
(denominators are only `2(1+psi)`). `Matrix.solve` + `cancel` per coefficient is exact and
fast (with `SYMPY_CACHE_SIZE=50000`). **Resolution:** use `Matrix.solve` for E2; if a future
higher-order scheme puts `y` in a denominator, route through `temo.solve_in_field`
(`temo.py:828`) over `QQ.frac_field(psi)` treating `y` powers as alpha-like terms.

### 6.5 Right branch is NOT reverse+negate (RESOLVE: derive separately)

Unlike `nbs` (which does `*= -1` + `reverse` for `right`), `interp_wall`'s right branch is a
genuinely separate derivation (interpolation is value-based, no antisymmetry; the wall column
also moves from `c[0]` to `c[T-1]`). **Resolution:** `derive_cut_cell_interp(left=False)`
with right-anchored deltas + `constrain_interp(left=False)`; `generate_interp_wall_method`
must NOT reuse the `nbs` `/h`/reverse tail.

### 6.6 Open question — generalization beyond E2

This spec targets E2 (the north star). Higher-order poly stencils (E4-poly, etc.) need
`R-1 > 2` wall rows, larger `T`, the `boundaryInterpWithInterior` near-interior row
(`taylor.wl:845-859`), and the general `bIndex`/`rIndex`/`stencilColRange` indexing
(`taylor.wl:401-408, 515-517`) rather than the E2 hard-coded `[-psi,0,1,2]` layout. Keep
`derive_cut_cell_interp` parameterized by `(p, q)` and the deltas factored through a
`build_cut_cell_interp_deltas(i, T, psi, left)` helper so the E2 path is a special case.
Defer E4-poly until E2 round-trips through the capstone test.
