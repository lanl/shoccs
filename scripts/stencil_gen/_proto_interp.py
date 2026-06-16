"""PROTOTYPE / VALIDATION (throwaway, left for the lead).

Proves the "polynomial-interpolation -> derivative via d/dy|0" idea in sympy
for E2 (p=1, nu=1), the way temo/boundary code is structured. Run with:

    cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run python _proto_interp.py

Five checks:
 1. interior interp polynomial via Taylor matching (rhs[k] = y^k/k!)
 2. d/dy|0 of (1) == central difference [-1/2, 0, 1/2] == derive_interior(0,1,1)
 3. one left-boundary cut-cell interp row gamma[1,j](y,psi,fa,ia) matched to
    golden values (incl. the 1/(1+psi) denominator), and its d/dy|0 row matched
    to the floating-derivative golden row.
 4. value/derivative decoupling: ia only in y^0 term, fa only in y^1 term.
 5. numeric cross-check vs the actual C++ polyE2_1.cpp formulas.
"""

from sympy import (
    Matrix,
    Rational,
    Symbol,
    cancel,
    diff,
    factorial,
    symbols,
)

from stencil_gen.interior import derive_interior, full_gamma_array

y = Symbol("y")
psi = Symbol("psi")


def interp_rhs(n_eqs):
    """RHS functional for f[i+y]: Taylor row k is the coeff of f^(k)[i] in the
    expansion of f[i+y], i.e. y^k / k!. (This is the single structural change
    vs. the derivative path, whose RHS is a unit vector at row nu.)"""
    return Matrix([Rational(1) * y**k / factorial(k) for k in range(n_eqs)])


def passed(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


all_ok = []

# ----------------------------------------------------------------------------
print("=" * 72)
print("CHECK 1: interior interp polynomial on nodes {-1,0,1} via Taylor match")
print("=" * 72)

# Unknown coeffs gamma_{-1}, gamma_0, gamma_1 (functions of y).
g = symbols("g_m1 g_0 g_p1")
nodes = [-1, 0, 1]
n_eqs = 3  # 3 unknowns -> rows k = 0,1,2

# Vandermonde: row k, col j  ==  node_j^k / k!   (discreteTaylorSeries / toTable)
A = Matrix(
    [[Rational(nodes[j] ** k) / factorial(k) for j in range(3)] for k in range(n_eqs)]
)
b = interp_rhs(n_eqs)

# Entries are exact in y (no psi) -> Matrix.solve is fine; cancel to clean form.
sol_vec = A.solve(b)
interior_interp = [cancel(c) for c in sol_vec]

expected_interp = [(y**2 - y) / 2, 1 - y**2, (y**2 + y) / 2]
print("  derived gamma[-1,0,1](y):")
for s_, c_ in zip(["gamma[-1]", "gamma[0]", "gamma[1]"], interior_interp):
    print(f"    {s_} = {c_}")
ok1 = all(
    cancel(a - e) == 0 for a, e in zip(interior_interp, expected_interp)
)
all_ok.append(passed("interior interp == [(y^2-y)/2, 1-y^2, (y^2+y)/2]", ok1))

# ----------------------------------------------------------------------------
print("=" * 72)
print("CHECK 2: d/dy|0 of interior interp == central difference [-1/2,0,1/2]")
print("=" * 72)

deriv_from_interp = [cancel(diff(c, y).subs(y, 0)) for c in interior_interp]
print(f"  d/dy|0 -> {deriv_from_interp}")

# Compare to derive_interior(0, 1, 1) full gamma array (the production path).
ic = derive_interior(0, 1, 1)
prod_deriv = full_gamma_array(ic)
print(f"  derive_interior(0,1,1) full_gamma_array -> {prod_deriv}")
ok2 = (
    deriv_from_interp == [Rational(-1, 2), Rational(0), Rational(1, 2)]
    and [Rational(v) for v in deriv_from_interp] == [Rational(v) for v in prod_deriv]
)
all_ok.append(passed("d/dy|0 == [-1/2,0,1/2] == derive_interior(0,1,1)", ok2))

# ----------------------------------------------------------------------------
print("=" * 72)
print("CHECK 3: left-boundary cut-cell interp row i=0 (C++ interp_wall left case0)")
print("=" * 72)

# Param symbols use build_symbol_map's name_i convention: fa_0 -> "fa[0]".
fa0, fa1 = symbols("fa_0 fa_1")
ia0, ia1 = symbols("ia_0 ia_1")

# --- GOLDEN: the exact C++ interp_wall left case0 coefficients (polyE2_1.cpp:116-124),
# transcribed with the documented column relabel applied so columns are in the
# C++ order [wall, node0, node1, node2]:
#   c0 = wall@(ymin - psi*h),  c1 = node@ymin,  c2 = node@ymin+h,  c3 = node@ymin+2h
t6_ = 1 / (1 + psi)
golden = [
    (1 + psi + ia1 - y + fa1 * y) * Rational(1, 2) * t6_,                       # c0 wall
    (1 + psi + fa0 * y + ia0 - y) * Rational(1, 2),                             # c1 node0
    (-psi + (fa0 * y + ia0) * (-2) + y
     + (-2 * ia1 - psi * ia1 + y - 2 * fa1 * y - psi * fa1 * y) * t6_)
    * Rational(1, 2),                                                           # c2 node1
    (ia1 + fa0 * y + ia0 + fa1 * y) * Rational(1, 2),                           # c3 node2
]

# --- THE NODE LAYOUT (reverse-engineered from polyE2_1.t.cpp:294-309 "left 0").
# mesh = linear_distribute(ymin,ymax,t-1); set ymin=0, h=1 -> nodes at 0,1,2.
# Column vector m = [ymin-psi*h, ymin, ymin+h, ymin+2h] -> x-positions:
xs = [-psi, Rational(0), Rational(1), Rational(2)]
# stencil::interp adjusts y by (psi-1) when the closest node IS the wall node
# (left 0: lc==ic), so interp_wall receives the adjusted y. In wall-local coords
# the row reproduces a field evaluated at:
xe = -psi + y  # == (ymin - h + y_raw*h) with y the *adjusted* y; verified vs test
T = 4

# --- DERIVE from scratch (the TEMO-style recipe), proving the construction:
#  (1) particular solution: a deg-1-exact interp with NO free params (Lagrange on
#      a chosen 2-node subset, others zero) -> the param-free backbone.
#  (2) nullspace basis: vectors n with sum_j n_j x_j^k = 0 for k=0,1. Adding any
#      multiple preserves deg-1 exactness. polyBlend feeds each as (ia + y*fa).
# Exactness rows k=0,1 (constant + linear) -> 2 constraints on T=4 unknowns -> 2-dim
# free space (the 2 free DOFs that become ia0/fa0 and ia1/fa1).
def interp_rhs_at(n_eqs, x):
    return Matrix([x**k / factorial(k) for k in range(n_eqs)])

# Particular solution: solve using the LAST two columns held to 0 (cols 0,3 free=0),
# i.e. plain 2-node Lagrange on nodes {x1=0, x2=1}.
A_part = Matrix([[xs[j] ** k / factorial(k) for j in (1, 2)] for k in range(2)])
p_sub = A_part.solve(interp_rhs_at(2, xe))
particular = [Rational(0), cancel(p_sub[0]), cancel(p_sub[1]), Rational(0)]

# Nullspace basis: 2 vectors v with V @ v = 0 where V has rows k=0,1 over xs.
V = Matrix([[xs[j] ** k / factorial(k) for j in range(T)] for k in range(2)])
nullbasis = [list(v) for v in V.nullspace()]  # 2 vectors, length 4
print(f"  particular (param-free) = {[cancel(c) for c in particular]}")
print(f"  nullspace basis vectors = {[[cancel(e) for e in v] for v in nullbasis]}")

# The two free DOFs are blended (ia_k + y*fa_k). The golden values fix the exact
# linear combination of the nullspace basis; recover the blend coefficients by
# matching column 0 and column 3 (the cleanest columns) to golden.
# golden = particular + (ia0 + y*fa0)*b0 + (ia1 + y*fa1)*b1, where b0,b1 are a
# specific basis. We solve for that basis by least-structure: express the golden
# homogeneous part (golden - particular) in the nullspace and read the blend.
homog = [cancel(golden[j] - particular[j]) for j in range(T)]
# Build the matrix [nullbasis[0] nullbasis[1]] and solve column-wise for the
# coefficient (which is a function of y carrying ia + y*fa).
Nb = Matrix.hstack(Matrix(nullbasis[0]), Matrix(nullbasis[1]))
# Solve Nb @ [a; b] = homog (overdetermined 4x2 but consistent since homog is in span)
coeffs_ab = (Nb.T * Nb).inv() * Nb.T * Matrix(homog)
a_coef = cancel(coeffs_ab[0])  # multiplies nullbasis[0]; should be a (ia + y*fa) blend
b_coef = cancel(coeffs_ab[1])  # multiplies nullbasis[1]
print(f"  recovered blend on basis0 = {a_coef}")
print(f"  recovered blend on basis1 = {b_coef}")

# Reconstruct and verify EXACT match to golden.
derived = [
    cancel(particular[j] + a_coef * nullbasis[0][j] + b_coef * nullbasis[1][j])
    for j in range(T)
]
labels = ["c0(wall)", "c1(n0)", "c2(n1)", "c3(n2)"]
ok3a = True
for lbl, c_, g_ in zip(labels, derived, golden):
    match = cancel(c_ - g_) == 0
    ok3a = ok3a and match
    print(f"    {lbl}: {'MATCH' if match else 'MISMATCH'}")
    if not match:
        print(f"         derived = {c_}\n         golden  = {cancel(g_)}")
all_ok.append(passed("derived row (particular + nullspace blend) == golden", ok3a))

# Confirm the blends are exactly (ia + y*fa): y^0 part is ia-only, y^1 part fa-only.
ok3blend = True
for nm, blend in [("basis0", a_coef), ("basis1", b_coef)]:
    v0 = cancel(blend.subs(y, 0))
    v1 = cancel(diff(blend, y))
    rem = cancel(blend - v0 - y * v1)
    is_blend = (
        {fa0, fa1}.isdisjoint(v0.free_symbols)
        and {ia0, ia1}.isdisjoint(v1.free_symbols)
        and rem == 0
    )
    ok3blend = ok3blend and is_blend
    print(f"  {nm} blend = (ia-part: {v0}) + y*(fa-part: {v1})  -> {'ia+y*fa OK' if is_blend else 'NOT a blend'}")
all_ok.append(passed("free DOFs enter as (ia + y*fa) blends (polyBlend structure)", ok3blend))

# d/dy|0 of this interp row gives the floating-derivative row; verify it is exactly
# the y^1 part and depends on fa (not ia) -> this IS the nbs_floating-style row.
deriv_row = [cancel(diff(c_, y).subs(y, 0)) for c_ in derived]
golden_deriv = [cancel(diff(g_, y).subs(y, 0)) for g_ in golden]
print("  d/dy|0 (-> derivative row):")
ok3b = True
for lbl, c_, gd in zip(labels, deriv_row, golden_deriv):
    match = cancel(c_ - gd) == 0
    ok3b = ok3b and match
    has_no_ia = {ia0, ia1}.isdisjoint(c_.free_symbols)
    ok3b = ok3b and has_no_ia
    print(f"    d/dy {lbl} = {c_}  ({'no ia' if has_no_ia else 'HAS ia!'})")
all_ok.append(passed("d/dy|0 row == d/dy of golden, depends on fa only (derivative duality)", ok3b))

# ----------------------------------------------------------------------------
print("=" * 72)
print("CHECK 4: value/derivative decoupling (ia in y^0 only, fa in y^1 only)")
print("=" * 72)

# Use c2 (node1, the most coupled coefficient) as the witness.
g13 = golden[2]
value_part = cancel(g13.subs(y, 0))  # y=0 value
deriv_part = cancel(diff(g13, y).subs(y, 0))  # d/dy|0
# higher-order remainder
remainder = cancel(g13 - value_part - y * deriv_part)
print(f"  c2(y)          = {cancel(g13)}")
print(f"  value (y=0)    = {value_part}")
print(f"  deriv (d/dy|0) = {deriv_part}")
print(f"  O(y^2) remaind = {remainder}")
value_syms = value_part.free_symbols
deriv_syms = deriv_part.free_symbols
ok4 = (
    {fa0, fa1}.isdisjoint(value_syms)  # no fa in the value part
    and {ia0, ia1}.isdisjoint(deriv_syms)  # no ia in the deriv part
    and remainder == 0  # exactly linear in y for this coeff
)
print(f"  value free syms = {sorted(str(s) for s in value_syms)}")
print(f"  deriv free syms = {sorted(str(s) for s in deriv_syms)}")
all_ok.append(passed("ia only in value(y=0), fa only in d/dy|0, exact-linear", ok4))

# ----------------------------------------------------------------------------
print("=" * 72)
print("CHECK 5: numeric cross-check vs C++ polyE2_1.cpp formulas")
print("=" * 72)

# 5a) interior interp: C++ ships the 2-point linear reduction (query_interp {2,4}).
#     For y>0:  c0 = 1 - y, c1 = y.  Our 3-pt poly restricted to nodes {0,1}
#     is exactly that 2-pt linear interp; verify numerically at y=0.3.
yv = 0.3
# 2-pt linear (forward) interp the C++ uses:
cpp_int = [1 + -1 * yv, yv]
# Build the 2-pt poly ourselves (nodes {0,1}, rhs y^k/k!, rows k=0,1):
A2 = Matrix([[Rational(0) ** k / factorial(k), Rational(1) ** k / factorial(k)] for k in range(2)])
sol2 = A2.solve(interp_rhs(2))
our_int = [float(cancel(c).subs(y, yv)) for c in sol2]
print(f"  interior y=0.3: ours={our_int}  cpp={cpp_int}")
ok5a = all(abs(a - b_) < 1e-12 for a, b_ in zip(our_int, cpp_int))

# 5b) boundary interp_wall left case0: our DERIVED row (proven == golden) must
#     match the literal C++ formulas element-wise at sample (psi,y,fa,ia).
#     C++ left case0 (polyE2_1.cpp:116-124), with t6=1/(1+psi), t7=-y,
#     t14=fa0*y,t15=ia0,t8=fa1,t10=ia1,t9=fa1*y:
psiv, yvv = 0.8, 0.3
fa0v, fa1v, ia0v, ia1v = 0.1, 0.2, 0.7, 0.8
t6n = 1.0 / (1 + psiv)
t7n, t14n, t15n, t8n, t10n, t9n = -yvv, fa0v * yvv, ia0v, fa1v, ia1v, fa1v * yvv
cpp_row = [
    (1 + psiv + t10n + t7n + t9n) * 0.5 * t6n,                                  # c0
    (1 + psiv + t14n + t15n + t7n) * 0.5,                                       # c1
    (-1 * psiv + (t14n + t15n) * -2 + yvv
     + (-2 * t10n - 1 * psiv * t10n + yvv - 2 * t8n * yvv - 1 * psiv * t8n * yvv) * t6n)
    * 0.5,                                                                      # c2
    (t10n + t14n + t15n + t9n) * 0.5,                                           # c3
]
subs_map = {psi: Rational(8, 10), y: Rational(3, 10),
            fa0: Rational(1, 10), fa1: Rational(2, 10),
            ia0: Rational(7, 10), ia1: Rational(8, 10)}
our_row = [float(cancel(c).subs(subs_map)) for c in derived]
print(f"  interp_wall left case0 full row:")
print(f"    ours = {our_row}")
print(f"    cpp  = {cpp_row}")
ok5b = all(abs(a - b_) < 1e-12 for a, b_ in zip(our_row, cpp_row))

# 5c) End-to-end test oracle: the row must reproduce bf(x)=2x+1 at the eval point
#     the C++ test checks. C++ "left 0": y_raw=0.3, psi=0.8, ymin=0,h=1; the
#     orchestrator adjusts y->y+psi-1=0.1, and expects bf(ymin - h + y_raw*h)=bf(-0.7).
y_adj = 0.1  # = y_raw + psi - 1
xnodes = [-psiv, 0.0, 1.0, 2.0]  # wall, n0, n1, n2 (ymin=0,h=1)
row_at_adj = [float(cancel(c).subs({psi: Rational(8, 10), y: Rational(1, 10),
                                    fa0: Rational(1, 10), fa1: Rational(2, 10),
                                    ia0: Rational(7, 10), ia1: Rational(8, 10)}))
              for c in derived]
bf = lambda x: 2 * x + 1
interp_val = sum(row_at_adj[j] * bf(xnodes[j]) for j in range(4))
expected = bf(-1 + 0.3)  # bf(ymin - h + y_raw*h)
print(f"  bf reproduction: interp={interp_val}  expected bf(-0.7)={expected}")
ok5c = abs(interp_val - expected) < 1e-12

all_ok.append(
    passed("numeric match vs C++ (interior 2pt + interp_wall full row + bf oracle)",
           ok5a and ok5b and ok5c)
)

# ----------------------------------------------------------------------------
print("=" * 72)
print(f"SUMMARY: {sum(all_ok)}/{len(all_ok)} checks passed")
print("=" * 72)
if not all(all_ok):
    raise SystemExit("ONE OR MORE CHECKS FAILED")
print("ALL CHECKS PASSED -- the poly-interp -> derivative idea works in sympy.")
