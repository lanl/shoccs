# Operators (`src/operators/`)

> **Maturity:** mature · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose

This subsystem turns finite-difference *stencils* into applied *discrete differential operators* on the cut-cell grid. A `derivative` assembles, for one axis, all the per-line matrices needed to compute `du = d/dx_dir(u)` on a field that has both a regular Cartesian (`D`) component and embedded-boundary ray (`Rx`/`Ry`/`Rz`) components. `gradient` and `laplacian` are thin compositions of three `derivative`s (one per axis). These three operators are the spatial-discretization core of every PDE system in SHOCCS. The directory also hosts two satellites: the `operator_visitor`/`eigenvalue_visitor` analysis hook (materialize the 1D operator as a dense matrix and take its spectrum for CFL/stability work) and the small standalone `shoccs-bcs` library that parses boundary-condition specs from Lua.

## Where it lives

| File | Role |
| --- | --- |
| `src/operators/derivative.hpp` | `derivative` class declaration: the O/B/N/Bf*/Br* matrix members, eager `operator()`, `visit()` (1D-only), and the templated `add_graph_nodes` Kokkos-Graph builders. |
| `src/operators/derivative.cpp` | The heavy lifting (~614 lines): `domain_discretization` (builds O/B/N per grid line) and `cut_discretization` (builds Bf*/Br* per ray direction, incl. the `interp_deriv_coefficients` interpolation path), the eager apply kernels, `build_graph`/`submit_graph`, and explicit template instantiations for `eq_t`/`plus_eq_t`. |
| `src/operators/gradient.{hpp,cpp}` | Owns three `derivative`s; `operator()` returns a closure writing three independent outputs `(du_x, du_y, du_z)`; `add_graph_nodes` zeros then fans out; `visit` forwards `dx` only. |
| `src/operators/laplacian.{hpp,cpp}` | Owns three `derivative`s that *accumulate* into one output with `plus_eq`; Neumann overload; `build_graph`/`submit_graph`/`add_graph_nodes`. |
| `src/operators/operator_visitor.hpp` | Tiny abstract base: one pure virtual `visit(const derivative&)` for double-dispatch analysis passes. |
| `src/operators/eigenvalue_visitor.{hpp,cpp}` | The only concrete `operator_visitor`. Materializes the 1D operator as a dense matrix and computes its eigenvalues with LAPACK `geev`. Consumed by `hyperbolic_eigenvalues` for spectral CFL stats. |
| `src/operators/boundaries.{hpp,cpp}` | `shoccs-bcs` library (separate target from `shoccs-operators`): the `bcs::type`/`Line`/`Grid`/`Object` BC vocabulary and the `from_lua` parser. |
| `src/operators/identity_stencil.hpp` | Test-only identity stencil (`ccs::stencils::identity`) used by the operator tests to isolate assembly logic from real coefficients. **Not a production scheme** (the Lua scheme factory in `stencils/stencil.cpp` cannot select it). |
| `src/operators/CMakeLists.txt` | Defines `shoccs-bcs` and `shoccs-operators` and the four operator tests. Line 20 is a commented-out `divergence` test — dead (see Maturity). |

## Public API / entry points

All symbols are in namespace `ccs` (BCs in `ccs::bcs`). `scalar_view` is a read-only 4-component handle `{D, Rx, Ry, Rz}` of `std::span<const real>`; `scalar_span` is the writable mutable counterpart (`src/fields/scalar.hpp`). `Op` is an accumulation policy: `eq_t` (overwrite, `x = y`) or `plus_eq_t` (accumulate, `x += y`), with values `ccs::eq` / `ccs::plus_eq` (`src/types.hpp`).

### `derivative`

```cpp
derivative();  // default — empty, for the degenerate / not-yet-built case

derivative(int dir,
           const mesh& m,
           const stencil& st,
           const bcs::Grid& grid_bcs,
           const bcs::Object& object_bcs,
           const logs& = {});            // ctor assembles O / B / N / Bf* / Br*

// Eager apply (internally fences). Non-Neumann:
template <typename Op = eq_t>            // Op constrained: invocable<Op, real&, real>
void operator()(scalar_view u, scalar_span du, Op op = {}) const;

// Eager apply with Neumann extra-data field nu:
template <typename Op = eq_t>
void operator()(scalar_view u, scalar_view nu, scalar_span du, Op op = {}) const;

// Pre-instantiated Kokkos graph (bakes in buffer pointers):
template <typename Op = eq_t> void build_graph(scalar_view u, scalar_span du, Op = {});
template <typename Op = eq_t> void build_graph(scalar_view u, scalar_view nu, scalar_span du, Op = {});
void submit_graph();

// Append nodes to an existing Kokkos::Experimental graph, chaining from `parent`.
// Returns a when_all() of the leaf nodes so the caller keeps chaining.
template <typename Op = eq_t, typename NodeT>
auto add_graph_nodes(NodeT parent, scalar_view u, scalar_span du, Op = {}) const;
template <typename Op = eq_t, typename NodeT>
auto add_graph_nodes(NodeT parent, scalar_view u, scalar_view nu, scalar_span du, Op = {}) const;

void visit(matrix::visitor& v) const;    // 1D-only: visits O, B, Bfx, Brx
```

Only `eq_t` and `plus_eq_t` are explicitly instantiated for `operator()`/`build_graph` (end of `derivative.cpp`); other `Op` types will not link.

### `gradient`

```cpp
gradient();
gradient(const mesh&, const stencil&, const bcs::Grid&, const bcs::Object&, const logs& = {});

// Returns a closure: call it with three output spans.
std::function<void(scalar_span, scalar_span, scalar_span)> operator()(scalar_view u) const;
// usage: grad(u)(du_x, du_y, du_z);

void visit(operator_visitor& v) const;   // forwards ONLY dx (gradient.hpp:30)

template <typename NodeT>
auto add_graph_nodes(NodeT parent, scalar_view u,
                     scalar_span du_x, scalar_span du_y, scalar_span du_z) const;
```

### `laplacian`

```cpp
laplacian();
laplacian(const mesh&, const stencil&, const bcs::Grid&, const bcs::Object&, const logs& = {});

std::function<void(scalar_span)> operator()(scalar_view u) const;             // no Neumann
std::function<void(scalar_span)> operator()(scalar_view u, scalar_view nu) const;  // Neumann
// usage: du = lap(u);   or   du = lap(u, nu);   (scalar_span::operator=(Fn) invokes it)

void build_graph(scalar_view u, scalar_span du);
void build_graph(scalar_view u, scalar_view nu, scalar_span du);
void submit_graph();

template <typename NodeT> auto add_graph_nodes(NodeT parent, scalar_view u, scalar_span du) const;
template <typename NodeT> auto add_graph_nodes(NodeT parent, scalar_view u, scalar_view nu, scalar_span du) const;
```

### Analysis: `operator_visitor` / `eigenvalue_visitor`

```cpp
struct operator_visitor { virtual void visit(const derivative&) = 0; };

class eigenvalue_visitor : public operator_visitor {
public:
    eigenvalue_visitor();
    template <Range Rx, Range Ry, Range Rz>
    eigenvalue_visitor(int3 nxyz, Rx&& rx, Ry&& ry, Rz&& rz);   // asserts nxyz[1]==nxyz[2]==1
    void visit(const derivative&) override;
    std::span<const real> eigenvalues_real() const;
    std::span<const real> eigenvalues_imag() const;
};
```

### `shoccs-bcs` (boundary-condition vocabulary)

```cpp
namespace ccs::bcs {
enum class type { Dirichlet, D = Dirichlet, Floating, F = Floating, Neumann, N = Neumann };
struct Line { type left, right; };          // <=> default-comparable
using Grid   = std::array<Line, 3>;          // [X, Y, Z] domain boundaries
using Object = std::vector<type>;            // one entry per immersed shape

// convenience constants: dd ff nn dn nd df fd fn nf
constexpr auto dd = Line{type::D, type::D};  // (and so on)

std::optional<std::pair<Grid, Object>>
from_lua(const sol::table&, index_extents, const logs& = {});
}
```

`from_lua` reads `domain_boundaries.{xmin,xmax,ymin,...}` (strings `"dirichlet"`/`"floating"`/`"neumann"`; default is Floating) and per-shape `shapes[i].boundary_condition`. Dirichlet/Neumann on an axis with extent 1 is downgraded to Floating with a warning.

## How it works

### The matrix decomposition produced by a `derivative`

For axis `dir`, the constructor assembles up to nine small per-line matrices (`derivative.hpp:24-30`):

- **O** — `matrix::block`. The fluid interior + boundary closures over the regular `D` field. Internally a stack of `inner_block = [dense_left | circulant_interior | dense_right]`, one per grid line in `dir`.
- **B** — `matrix::csr`. Couples grid-boundary closures to the ray data. Its *source* component is `R{dir}` (Rx for dir 0, Ry for 1, Rz for 2 — see the `b_src` selection and the `apply_kernels` switch).
- **N** — `matrix::csr`. Adds Neumann extra-data contributions (only populated where a grid BC is Neumann).
- **Bfx/Brx, Bfy/Bry, Bfz/Brz** — six `matrix::csr` for the cut-cell ray points. `Bf*` are fluid→ray (`D → R`) maps; `Br*` are ray→ray (`R → R`) maps. The pair for ray direction `r` produces the derivative *value at the immersed-boundary intersection points* `m.R(r)`.

### Two-phase assembly (in the ctor)

1. Early-return if `m.extents()[dir] < 2` (degenerate axis ⇒ all matrices empty).
2. Query the stencil (`query_max`), build the interior circulant coefficients via `st.interior(h, ...)`.
3. **`domain_discretization`** — for each grid line in `dir` (skipping pure-Dirichlet lines): query the stencil for the left/right BC (`st.query` + `st.nbs`), build a `dense` left and right closure plus a `circulant` interior, and emit them as an `inner_block` into O. Grid-boundary terms go into B; Neumann extra data goes into N. Dirichlet rows are dropped (`remove_left_row`/`remove_right_row`); object closures additionally drop the first column (handled by the R operators) via `remove_left_row_col`.
4. **`cut_discretization`** is called three times, once per ray direction `r` (0,1,2), to build the `Bf{r}`/`Br{r}` pair. When `dir == r` no interpolation is needed (the ray is aligned with the derivative). When `dir != r`, `interp_deriv_coefficients` + `st.interp` build an interpolation stencil onto the closest mesh line. **Fast exit:** `cut_discretization` returns immediately if the ray set is empty *or* every object BC is Dirichlet — so "no cut-cell operator built" is normal for pure-Dirichlet immersed bodies.

### Applying it (eager path)

`derivative::operator()` → `apply_kernels` runs, in order:

- ray updates: `Bfx(u.D, du.Rx)`, `Bfy(u.D, du.Ry)`, `Bfz(u.D, du.Rz)`, then `Brx(u.Rx, du.Rx)` etc.;
- fluid update: `O(u.D, du.D, op)`, then `B(u.R{dir}, du.D)` (and `N(nu.D, du.D)` on the Neumann overload);
- then a `Kokkos::fence()` per call.

`gradient::operator()(u)` returns a closure that zeros `du_x/du_y/du_z` and calls `dx/dy/dz` with `eq` (independent outputs). `laplacian::operator()(u)` returns a closure that zeros `du` then calls `dx/dy/dz` with **`plus_eq`** into the *same* output (the three second-derivatives sum to the Laplacian).

### Eager vs Kokkos-Graph

- **Eager** (`operator()`) fences every call — simple, used in the analysis path and as the correctness oracle.
- **Self-contained graph** (`build_graph` + `submit_graph`) bakes raw buffer pointers in at build time and fences only at submit.
- **Fused graph** (`add_graph_nodes`) lets a *system* splice the whole RHS into one graph: `gradient`/`laplacian` insert explicit zero-fill nodes, then chain `dx/dy/dz` (independent for gradient, sequential `plus_eq` for laplacian), returning a `when_all` of leaf nodes. The canonical wiring lives in `heat.cpp` (`lap.add_graph_nodes(root, u, nu, du)` then the source-term nodes) and `scalar_wave.cpp` (`grad.add_graph_nodes(...)` then the dot-product nodes).

### Analysis path

`hyperbolic_eigenvalues::stats` builds `eigenvalue_visitor{m.extents(), ...}` and calls `grad.visit(v)`. `gradient::visit` forwards `dx` only; `eigenvalue_visitor::visit` first runs a `matrix::unit_stride_visitor` over `derivative::visit` (O/B/Bfx/Brx) to learn the dense layout, then *moves* that into a `matrix::coefficient_visitor`, re-walks the operator to fill a dense matrix, and calls LAPACK `geev` to get the (complex) spectrum. The system reports `-h * min(Re eigenvalue)` as the stability number.

## How to extend

**Add a new differential operator** (e.g. revive `divergence`): copy the `gradient`/`laplacian` pattern.

1. New class owns three `derivative dx/dy/dz` members + an `index_extents ex`.
2. Construct them in the ctor from `(mesh, stencil, Grid, Object)`. Build a `logs` sublog like `gradient.cpp:19` if you want per-row interpolation logging.
3. Compose results in `operator()` (gradient writes three independent outputs with `eq`; laplacian/divergence accumulate into one output with `plus_eq` — remember to zero the output first) and mirror it in `add_graph_nodes` (zero-fill nodes, then chain — sequential when accumulating).
4. Guard degenerate axes: `if (ex[0] > 1) dx(...)`.
5. Add the `.cpp` to the `add_library(shoccs-operators ...)` list in `src/operators/CMakeLists.txt`.
6. Add a `t-<name>` test block by **copying an existing one** (e.g. the `t-derivative` block). Do **not** use the `add_unit_test` helper here — operator tests need a custom `Kokkos::ScopeGuard` `main()` and link `Catch2::Catch2` (not `Catch2WithMain`).

**Add a new boundary-condition kind:** extend `bcs::type` (`boundaries.hpp`), add a branch to `get_bc` in `boundaries.cpp`, and handle the new type in the `st.query`/`st.nbs` branches of `domain_discretization` and `cut_discretization`. The stencil must also support it (`stencils/` `query`/`nbs`).

**Add a new stencil scheme:** nothing in operators changes — schemes plug in transparently through the type-erased `stencil` interface. Add the scheme in `stencils/` and register it in the Lua factory (`stencils/stencil.cpp`).

**Add a new analysis pass:** subclass `operator_visitor`, implement `visit(const derivative&)`, and use `derivative::visit(matrix::visitor&)` to walk O/B/Bfx/Brx into a matrix-level visitor (the `eigenvalue_visitor` is the worked example). Note the strict 1D restriction below.

## Gotchas & invariants

- **`derivative::visit()` and `gradient::visit()` are 1D-only.** `visit` only walks O, B, Bfx, Brx and is commented `// Assumes 1d`. `gradient::visit` forwards **only `dx`** (drops dy/dz). `eigenvalue_visitor` hard-asserts `nxyz[1] == nxyz[2] == 1`. This is a purpose-built 1D stability hook, not general multi-D introspection.
- **Graph paths bake raw pointers.** `build_graph`/`add_graph_nodes` capture `u.D.data()` etc. by value into `KOKKOS_LAMBDA`s at build time. The graph is invalid if the backing buffers are reallocated/resized — rebuild the graph after any field-registry resize.
- **`plus_eq` accumulation requires a zeroed output.** `laplacian` and the heat RHS accumulate; the eager `operator()` does `du = 0` first, and `add_graph_nodes` inserts explicit zero-fill nodes. If you wire up an externally-managed `du` yourself and forget to zero it, you accumulate garbage.
- **Pure-Dirichlet objects build no cut-cell operator.** `cut_discretization` fast-exits when all object BCs are Dirichlet, and Dirichlet rows are skipped throughout — an empty Bf*/Br* is expected, not a bug.
- **Degenerate axes are silently zero.** `gradient`/`laplacian` skip an axis when `ex[dir] <= 1`; `derivative`'s ctor early-returns for `extents[dir] < 2`. A zero derivative in a flat dimension is intentional.
- **B's source component is `R{dir}`.** `b_src = Rx` for dir 0, `Ry` for 1, `Rz` for 2 (the `apply_kernels` switch and the `add_graph_nodes`/`build_graph` ternary). Wiring B to the wrong R-component is a subtle correctness trap.
- **`eigenvalue_visitor` cannibalizes its `unit_stride_visitor`.** `v = matrix::coefficient_visitor{MOVE(u)}` (`eigenvalue_visitor.cpp:22`) leaves `u` moved-from — do not reuse it.
- **`eigenvalue_visitor` must pass `ldvl = ldvr = n`** even though `jobl = jobr = 'N'`; leaving them at 1 errors inside `dgeev_work` (documented in-code).
- **LAPACK lives only here.** `shoccs-operators` links `lapackpp` PRIVATE solely for `eigenvalue_visitor`. It is the only LAPACK dependency in the operator stack.
- **Eager vs graph timing differs.** Eager `operator()` fences per call; the graph path fences only at submit. Results are tested equivalent, but do not assume identical synchronization semantics.
- **Build status (project-wide, fixed 2026-06-04).** The C++ build is green: the Kokkos 5.0→5.1.1 `create_graph` API break (which had left the prior `build/` tree linking a stale `libkokkoscore.so.5.0`) was resolved by migrating all 17 call sites to the templated 1-arg `create_graph<execution_space>(closure)` form. `cmake --build build` builds the whole tree; `ctest --test-dir build` = 47/48. Three operator tests (`t-derivative`, `t-gradient`, `t-eigenvalue_visitor`) pass; `t-laplacian` is now the *only* remaining failure project-wide — one cut-cell numerics failure (see the Tests section and [Cleanup Plan §0a](../CLEANUP_PLAN.md)), unrelated to the build. (The two previously-failing non-operator tests, `t-csr` and `t-E2_1`, were fixed in the same 2026-06-04 Kokkos 5.1 pass — `t-csr` got a custom `Kokkos::ScopeGuard` `main()` linking `Catch2::Catch2` + `Kokkos::kokkos`, and `t-E2_1` got a `.margin(1e-12)` on its `Approx` comparisons.)

## Maturity & known gaps

**Verdict: mature.** `derivative`/`gradient`/`laplacian` are the spatial-discretization core with real production callers: `heat` holds a `laplacian lap` (calls `lap(u, nu)` and `lap.add_graph_nodes`), `scalar_wave` holds a `gradient grad` (`grad(u)(dux,duy,duz)` and `grad.add_graph_nodes`), and `hyperbolic_eigenvalues` holds a `gradient grad` and calls `grad.visit(eigenvalue_visitor)`. All are wired into the `system` variant and exercised end-to-end (heat in `simulation_cycle.t.cpp`). Both eager and graph execution paths are fully implemented with explicit `eq_t`/`plus_eq_t` instantiations; last touched in the Phase-19 Kokkos-Graph migration (2026-03-27). (The build is green as of 2026-06-04; the one remaining `t-laplacian` failure is a cut-cell numerics question, not an assembly or execution-path gap.)

Item-by-item (verified flags):

- **`divergence` operator — DEAD (zero callers, safe to delete the comment).** No `divergence.{hpp,cpp,t.cpp}` exists anywhere; the source stub was deleted in commit `a50788a` (Phase 18, "Delete dead files"). Only `src/operators/CMakeLists.txt:20` survives as a commented-out test line `#add_unit_test(divergence "operators" operators random)`. It cannot be uncommented without re-implementing the operator from scratch. No Lua config, test, or executable path reaches it. (The many `divergence` hits under `src/mms/` are unrelated — manufactured-solution source-term *values*, not this operator.) Recommend deleting the dangling comment. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`operator_visitor` base + `gradient::visit` — PARTIAL (works, deliberately narrow).** Real and used: exactly one subclass (`eigenvalue_visitor`), real test coverage (`t-eigenvalue_visitor`), and a complete live path from `eigenvalues.lua` through `hyperbolic_eigenvalues::stats`. But `gradient::visit` forwards only `dx`, and the whole path asserts a strictly 1D mesh. It is a 1D eigenvalue-analysis hook, **not** a general multi-D operator-introspection framework. Keep it; do not assume multi-D visitor support exists.
- **`identity_stencil.hpp` — MATURE fixture, TEST-ONLY.** A complete, stable (~4.5 years, no churn) identity stencil used by `derivative.t.cpp`, `laplacian.t.cpp`, `eigenvalue_visitor.t.cpp` to isolate assembly logic. It is **not** dead and **not** experimental, but it is **not a production scheme**: the Lua scheme factory (`stencils/stencil.cpp`) has no `"identity"` branch, so it can never be selected via config. Caveat: it lives in the production include path rather than a test dir, which can mislead. Do not use it in real configs; do not delete (breaks three tests).
- **`hyperbolic_eigenvalues` (sole production consumer of `eigenvalue_visitor`, in `src/systems/`)** — not part of this subsystem, but worth knowing it is the only non-test caller. The audit confirmed it is a *complete, tested diagnostics tool*, not unfinished: its empty `rhs`/`initialize`/`update_boundary` and `timestep_size()==1.0` are by design (it reports a spectral stability statistic, it does not advance a PDE). So `eigenvalue_visitor` is production-supporting, not dead.

## Tests

Registered with label `operators` (BCs with label `bcs`):

| Test | TEST_CASEs | Covers |
| --- | --- | --- |
| `t-derivative` | 9 | 1D derivative with Dirichlet/Floating/Neumann grid BCs, mixed combos (DDFNFD, NNDDDF, FNDDDF, …), embedded objects (Dirichlet + Floating), 2D, identity-stencil sanity, E2/E2-poly, graph-vs-eager equivalence (incl. resubmit determinism + Neumann overload). |
| `t-laplacian` | 5 | Domain, Dirichlet/Floating objects, 2D, graph-vs-eager, Neumann overload. |
| `t-gradient` | 4 | Domain, Dirichlet/Floating objects, 2D, graph-vs-eager. |
| `t-eigenvalue_visitor` | 2 | Identity stencil (eigs == 1) and a calibrated E2-poly max-eigenvalue regression value (1D). |
| `t-boundaries` | 1 | `bcs::from_lua` parsing (label `bcs`). |

**Not covered / gaps:** (1) `divergence` — no test (dead). (2) No standalone `operator_visitor` test — exercised only via `eigenvalue_visitor`. (3) `eigenvalue_visitor`/`visit` is asserted and tested 1D-only. (4) `gradient::add_graph_nodes` is unit-tested less directly than `derivative`/`laplacian` (its main exercise is `scalar_wave`). (5) **Current status (build green 2026-06-04, ctest 47/48):** `t-derivative`, `t-gradient`, and `t-eigenvalue_visitor` pass. `t-laplacian` is the **only remaining failure** project-wide and **FAILS** for a real numerical reason — the cut-cell R-point ("E2 with Floating Objects") `rx_vec` values differ ~2-3% from expected; the interior `d_vec` assertion passes. This is a genuine cut-cell numerics question, not a build/link problem (was the Kokkos 5.1 `create_graph` break, fixed 2026-06-04). The two other previously-documented failures are now fixed: `t-csr` (custom `Kokkos::ScopeGuard` `main()` + `Catch2::Catch2`/`Kokkos::kokkos` link) and `t-E2_1` (`.margin(1e-12)` on its `Approx` comparisons). Tracked in [Cleanup Plan §0a](../CLEANUP_PLAN.md).

## Related docs

- [Stencils](stencils.md) — the coefficient source that operators consume (the `stencil` type-erased `query`/`nbs`/`interior`/`interp` interface).
- [Matrices](matrices.md) — `block`/`inner_block`/`dense`/`circulant`/`csr` and the `matrix::visitor` / `unit_stride_visitor` / `coefficient_visitor` used by `eigenvalue_visitor`.
- [Fields](fields.md) — `scalar_view`/`scalar_span` and the `D`/`Rx`/`Ry`/`Rz` 4-component data model.
- [Core types](core-types.md) — `eq_t`/`plus_eq_t`, `int3`, `real`, etc.
- **Legacy (pre-Kokkos, treat as archive):** `docs/discrete_operators.md` and the operator sections of `SHOCCS_ARCHITECTURE_AND_KOKKOS_MIGRATION_SPEC.md` describe an "Operator Hierarchy" rooted at a `discrete_operator` base with `divergence`/`directional` branches and SBP/SAT operators. **None of this matches the current code:** `discrete_operator`, `divergence`, and `directional` were deleted (Phases 12/18); multi-D composition is just `derivative → {gradient, laplacian}` with the real base abstraction being `operator_visitor`; boundary closures are dense stencil-driven closures with strong/direct BC enforcement, **not** SBP-SAT; and the migration off range-v3 is complete. Read those docs only for historical rationale.
