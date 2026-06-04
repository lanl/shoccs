# Matrices (`src/matrices/`)

> **Maturity:** mature · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose

`matrices/` is SHOCCS's per-line "small matrix" operator library. It applies high-order finite-difference stencils along **one mesh axis at a time** and deliberately does **not** assemble a global sparse system. Each axis discretization is a composite `block` made of many `inner_block`s, where each `inner_block = [dense left boundary | circulant interior | dense right boundary]` describes one line through the field. Cut-cell embedded-boundary coupling (fluid domain ↔ object surface points) is held separately in `csr` matrices. Every mat-vec is a hand-written Kokkos kernel — there is intentionally **no KokkosKernels / KokkosSparse dependency** anywhere in the codebase.

## Where it lives

| File | Role |
| --- | --- |
| `src/matrices/common.hpp` | `matrix::matrix_base` geometry (rows/cols/offsets/stride) + the BC `flag` bit system (`ldd`/`rdd`, `rowspace_*`/`colspace_*` r-tags) consumed by the analysis visitors. |
| `src/matrices/dense.hpp` / `dense.cpp` | Dense boundary-closure block. Stores coeffs in a `device_view<real*>`; serial `operator()` matvec (test-only at apply time — see gaps). |
| `src/matrices/circulant.hpp` / `circulant.cpp` | Banded interior-stencil matrix. Half-bandwidth = `coeffs.size()/2`. `RangePolicy` matvec. |
| `src/matrices/inner_block.hpp` / `inner_block.cpp` | `[dense_left \| circulant \| dense_right]` wrapper for one line. Sets component offsets/stride at construction and **deletes** the offset/stride setters to lock geometry. Eager `operator()` is test-only post-Phase 17. |
| `src/matrices/inner_block_meta.hpp` | POD `inner_block_meta` struct (per-line metadata) copied to device for the `block` TeamPolicy kernel. |
| `src/matrices/block.hpp` | Multi-line composite. `build_device_arrays()` flattens its `inner_block`s into device `meta_d`/`coeffs_d`; `matvec_functor` TeamPolicy kernel; `operator()` + `graph_node()` (**production hot path**); nested `builder` with disjoint-row debug assert. |
| `src/matrices/csr.hpp` / `csr.cpp` | CSR sparse boundary-coupling matrix (`w`/`v`/`u` arrays). `operator()` is RangePolicy **`+=`**; `graph_node()` is **always `+=`**; nested `builder` (`add_point`/`to_csr`). |
| `src/matrices/matrix_visitor.hpp` | Abstract `visitor` base — double-dispatch over `dense`/`circulant`/`csr`. |
| `src/matrices/unit_stride_visitor.hpp` / `.cpp` | First analysis pass: assigns a dense global row/col numbering across a derivative's matrices, skipping Dirichlet rows/holes; `mapped()` lookups. |
| `src/matrices/coefficient_visitor.hpp` / `.cpp` | Second analysis pass: scatters each matrix's coefficients into a flat dense global matrix `m` for eigenvalue/stability analysis. |
| `src/operators/derivative.cpp` | **Primary consumer** (not in this subsystem): assembles all matrices and drives both matvec paths. |

## Public API / entry points

### Geometry & flags (`common.hpp`)

```cpp
class matrix::matrix_base {
    matrix_base(integer rows, integer columns,
                integer row_offset = 0, integer col_offset = 0, integer stride = 1);
    auto rows() const; auto columns() const;
    auto row_offset() const; auto col_offset() const; auto stride() const;
    matrix_base& row_offset(integer); // setters (overridden = delete in inner_block)
    matrix_base& col_offset(integer);
    matrix_base& stride(integer);
};

using flag = uint8_t;                       // BC tag attached to a matrix
constexpr flag ldd = 1, rdd = 2;            // dirichlet-dropped left / right column
constexpr flag rowspace_rx/ry/rz, colspace_rx/ry/rz; // r-space tagging for csr
constexpr bool is_ldd(flag), is_rdd(flag), is_rx(flag), is_ry(flag);
```

### Storage matrices

```cpp
// dense (dense.hpp) — contiguous row-major boundary closure
template <std::ranges::input_range R>
dense(integer rows, integer columns, R&& rng, flag boundary = 0);
template <std::ranges::input_range R>
dense(integer rows, integer columns,
      integer row_offset, integer col_offset, integer stride, R&& rng, flag boundary = 0);
integer size() const;                       // number of stored coeffs (rows*cols)
std::span<const real> data() const;         // host-readable coeff span (USE THIS)
const device_view<real*>& coeffs_view() const;   // DEAD accessor (see gaps)
flag flags() const; void flags(flag);
template <typename Op = eq_t> void operator()(span<const real> x, span<real> b, Op = {}) const;
void visit(visitor&) const;

// circulant (circulant.hpp) — banded interior stencil
circulant(integer rows, std::span<const real> coeffs);                 // col_offset = size/2
circulant(integer rows, integer row_offset, integer stride, std::span<const real> coeffs);
integer size() const;                        // = coeffs.size() (band width)
std::span<const real> data() const;
const device_view<real*>& coeffs_view() const;   // DEAD accessor (see gaps)
template <typename Op = eq_t> void operator()(span<const real> x, span<real> b, Op = {}) const;
void visit(visitor&) const;
```

### Composite matrices

```cpp
// inner_block (inner_block.hpp) — one line [dense | circulant | dense]
inner_block(dense&& left, circulant&& i, dense&& right);
inner_block(integer columns, integer row_offset, integer col_offset, integer stride,
            dense&& left, circulant&& i, dense&& right);
const dense&     left() const;
const circulant& interior_circ() const;
const dense&     right() const;
template <typename Op = eq_t> void operator()(span<const real> x, span<real> b, Op = {}) const; // test-only
void visit(visitor&) const;

// block (block.hpp) — many lines for one axis of a 3D field
block(std::vector<inner_block>&& blocks);    // flattens to device arrays at construction
integer rows() const;                        // size query only (see gotcha)
int num_lines() const;
const device_view<inner_block_meta*>& metadata_view() const;
const device_view<real*>&             coefficients_view() const;   // LIVE (≠ coeffs_view)
template <typename Op = eq_t> void operator()(span<const real> x, span<real> b, Op = {}) const;
template <typename NodeType, typename Op = eq_t>
auto graph_node(NodeType parent, const real* x_ptr, real* b_ptr, Op = {}) const;
void visit(visitor&) const;

struct block::builder {
    builder(); builder(integer reserve_n);
    template <typename... Args> void add_inner_block(Args&&... args); // forwards to inner_block ctor
    block to_block() &&;                       // debug-asserts disjoint output row ranges
};
```

### Sparse boundary coupling

```cpp
// csr (csr.hpp)
template <Range W, Range V, Range U>
csr(W&& w, V&& v, U&& u, flag row_col_space = 0);   // w=values, v=col idx, u=row offsets
integer rows() const;                          // u.size() - 1
integer size() const;                          // nnz
std::span<const integer> column_indices(integer row) const;
std::span<const real>    column_coefficients(integer row) const;
void operator()(span<const real> x, span<real> b) const;            // ALWAYS += (no Op)
template <typename NodeType>
auto graph_node(NodeType parent, const real* x_ptr, real* b_ptr) const;  // ALWAYS +=
flag flags() const; void flags(flag);
void visit(visitor&) const;

struct csr::builder {
    builder(); builder(integer reserve_n);
    void add_point(integer row, integer col, real v);
    csr to_csr(integer nrows);                 // sorts points, builds u offsets
};
```

### Analysis visitors

```cpp
struct matrix::visitor {                       // matrix_visitor.hpp
    virtual void visit(const dense&) = 0;
    virtual void visit(const circulant&) = 0;
    virtual void visit(const csr&) = 0;
};

class unit_stride_visitor : public visitor {   // pass 1 — global numbering
    unit_stride_visitor(integer rows, integer columns);
    template <Range Rx, Range Ry, Range Rz>
    unit_stride_visitor(int3 nxyz, Rx&& rx, Ry&& ry, Rz&& rz); // r* = dirichlet skip masks
    std::array<integer,2> mapped_dims() const; integer mapped_size() const;
    std::span<const integer> mapped(integer first_row, integer rows,
                                    integer first_col, integer cols) const; // dense/circulant
    std::span<const integer> mapped(integer row, const csr&) const;          // csr
};

class coefficient_visitor : public visitor {   // pass 2 — scatter into dense matrix
    coefficient_visitor(unit_stride_visitor&& v_);
    std::span<const real> matrix() const;       // the assembled dense global matrix
};
```

`Op` is `eq_t` (assign) or `plus_eq_t` (accumulate) from `src/types.hpp`.

## How it works

### The composite hierarchy

```
stencil coeffs
   │
   ├── dense  (left/right boundary closure rows)
   └── circulant  (interior band, half-bandwidth = size/2)
        │
        └── inner_block = [dense_left | circulant | dense_right]   ← one line
                 │   (× many, via block::builder::add_inner_block)
                 └── block   ← all lines of ONE axis of a 3D field

csr  ← separate: fluid-domain ↔ object-surface boundary coupling (B / N matrices)
```

A `block` is the discretization of a single direction (X, Y, or Z) over the whole domain. The `stride` of each `inner_block` encodes the axis layout: a unit-stride X line uses `stride = 1`, a Y line `stride = Nx`, a Z line `stride = Nx*Ny` — so one `block` lays down every line of one axis by giving each `inner_block` the right `row_offset` and `stride`.

### Two live mat-vec execution paths

Both paths are production code; both run the **same** TeamPolicy kernel logic:

1. **Eager** — `block::operator()(x, b, op)` launches a `Kokkos::TeamPolicy` with `matvec_functor` over the flattened device arrays. `circulant::operator()` and `csr::operator()` use `Kokkos::RangePolicy`; `dense::operator()` is a literal serial `std::inner_product` loop. Used by `derivative::operator()` (`derivative.cpp:489`, `O(u.D, du.D, op)`).
2. **Kokkos::Graph** — `block::graph_node(parent, x_ptr, b_ptr, op)` and `csr::graph_node(parent, x_ptr, b_ptr)` chain `parent.then_parallel_for(...)` nodes. Used by `derivative::build_graph` (`derivative.cpp:535–585`) for `heat` and `scalar_wave`.

### CRITICAL data-flow fact: `block` does NOT call `inner_block`

`block(std::vector<inner_block>&&)` calls `build_device_arrays()` **at construction**, flattening each line's `left().data()` / `interior_circ().data()` / `right().data()` into a single contiguous `coeffs_d` plus one `inner_block_meta` per line in `meta_d`. The hot-path `matvec_functor` (block.hpp:118–178) reads **only** `meta_d`/`coeffs_d` — it never iterates the `std::vector<inner_block>` and never calls `inner_block::operator()` or `dense::operator()`. Consequence: **`inner_block` is a builder-time value object, not a runtime applier**, and the eager `inner_block`/`dense` matvecs are now test-only reference implementations (see Maturity & known gaps).

The kernel walks `total_rows = left_rows + interior_rows + right_rows` per team (one team per line) with a `TeamThreadRange` over output rows and a `ThreadVectorRange` reduction over each row's stencil, writing `op(b_ptr[out_idx], dot)` once per row via `Kokkos::single`.

### The analysis (visitor) pipeline — separate from application

This is **not** how the operator is applied; it builds a dense global matrix for eigenvalue/stability spectra:

- `unit_stride_visitor` (pass 1): visit each matrix to assign global row/col indices, skipping Dirichlet rows and r-space holes; `-1` marks an unmapped (skipped) index.
- `coefficient_visitor` (pass 2, constructed by **moving** the populated `unit_stride_visitor` into it): scatter each matrix's coefficients into the flat `m` using the pass-1 numbering.
- Consumed by `src/operators/eigenvalue_visitor.{hpp,cpp}` (`d.visit(u); v = coefficient_visitor{MOVE(u)}; d.visit(v);`), reached from the `eigenvalues` Lua system (`hyperbolic_eigenvalues`).

## How to extend

### Most common case — a new stencil scheme: do NOT touch `matrices/`

To add a new per-line FD scheme you add a **stencil** (`src/stencils/`) that yields interior circulant coeffs + boundary dense rows + optional cut-cell CSR points. `src/operators/derivative.cpp` then assembles them; copy that pattern:

```cpp
// per-line interior + boundary closures → one block
matrix::block::builder O_builder;
O_builder.add_inner_block(sub.columns, row_offset, col_offset, stride,
                          matrix::dense{rLeft, tLeft, left},          // left closure
                          matrix::circulant{n_interior, interior},    // interior band
                          matrix::dense{rRight, tRight, right});      // right closure
matrix::block O = MOVE(O_builder).to_block();

// cut-cell coupling → CSR
matrix::csr::builder B_builder;
B_builder.add_point(row, obj->object_coordinate, coeff);
matrix::csr B = MOVE(B_builder.to_csr(num_rows));
```

(See `derivative.cpp:324–442` for the full assembly, including the `N` Neumann-closure CSR.)

### Add a new matrix STORAGE type (e.g. a different banded layout)

1. Write a class with `matrix_base`-compatible geometry, an `operator()(x, b, op)` Kokkos matvec, a `graph_node(parent, ...)` overload returning `parent.then_parallel_for(...)`, `data()`/`size()` accessors, and a `visit(visitor&)` hook.
2. Add a pure-virtual `visit(const NewType&)` to `matrix::visitor` (matrix_visitor.hpp) and implement it in **both** `unit_stride_visitor` and `coefficient_visitor` (and `operators/eigenvalue_visitor` if it should appear in spectra).
3. Register its `.cpp` in `src/matrices/CMakeLists.txt`; add a `*.t.cpp` test target following the explicit `add_executable` + `add_test` + `set_tests_properties(... LABELS "matrices")` pattern used for `t-dense`/`t-block` (`csr` also follows this pattern, with a custom `main()` + `Kokkos::ScopeGuard` linking `Catch2::Catch2` + `Kokkos::kokkos`).

### Add a new accumulation policy

Define an `Op` functor like `eq_t`/`plus_eq_t` (`src/types.hpp`) and **add explicit template instantiations** of `operator()` in `dense.cpp`/`circulant.cpp`/`inner_block.cpp` — each matvec is explicitly instantiated for `eq_t` and `plus_eq_t` only, so a new `Op` link-fails without them.

## Gotchas & invariants

- **`block` flattens at construction.** Mutating `inner_block`s after `to_block()` is not reflected; `block::operator()` never calls `inner_block::operator()`. Treat `inner_block` as builder-time data, not a runtime applier.
- **`csr` is always `+=`.** Both `csr::operator()` and `csr::graph_node` accumulate; there is no `eq` variant. The `Bf*→Br*` and `O→B→N` graph node chains rely on this accumulate-into-the-same-output ordering. `block` has both `eq_t`/`plus_eq_t`.
- **Bounds/correctness checks are `assert()`** (dense/circulant matvec bounds; `block::builder` disjoint-row check inside `#ifndef NDEBUG`). In `RelWithDebInfo`/`Release` these are compiled out — overlapping `inner_block` row ranges or oversized spans silently corrupt instead of failing.
- **`circulant` half-bandwidth convention.** `columns = rows + size - 1`; the simple ctor sets `col_offset = size/2`; the strided ctor sets `col_offset = -1` as a **sentinel** that `inner_block` later overwrites with the real offset. Misreading `size()/2` vs `stride` interplay breaks the index math.
- **Explicit `Op` instantiation only.** A new accumulation functor link-fails unless you add its explicit instantiation in the `.cpp` files.
- **`block::rows()` is a size query, never on the compute path.** It returns `b.row_offset() + b.rows()*b.stride()` for the last line and carries a 2021 comment about an unhandled "last point inside an object" case. The matvec uses `num_lines()` + per-line metadata, not `rows()`; cut-cell geometry is handled in `domain_discretization` (which does not generate that edge case). See gaps.
- **Visitors are 1D / unit-stride only.** `unit_stride_visitor` and `coefficient_visitor` `assert(stride == 1)` and early-return otherwise; they serve the 1D eigenvalue pipeline (`eigenvalue_visitor` asserts `nxyz[1] == nxyz[2] == 1`), not general 3D analysis.
- **Hand-encoded flag bit layout.** `rx=1, ry=2, rz=4` shifted by `row_shift=3` for rowspace; `unit_stride_visitor::visit(csr)` indexes a `std::array<…,5>` by flag value, relying on those numeric values. Changing the bit values silently breaks CSR remapping.
- **Device-correctness caveat.** `dense`/`circulant` store coeffs in a `device_view` but the serial `dense::operator()` reads via `data()` on host. This is correct only because `execution_space` is currently host-only (`DefaultHostExecutionSpace`); a real device backend would break the serial dense matvec.
- **`t-csr` was fixed under Kokkos 5.1 — harness issue, not a code regression.** The build is green (fixed 2026-06-04; the Kokkos 5.0→5.1.1 `create_graph` break was resolved by migrating all call sites to the templated 1-arg form `create_graph<execution_space>(closure)`). All 7 `matrices` test binaries now pass. `t-csr` previously SIGABRTed because Kokkos 5.1 rejects constructing an OpenMP execution space before `Kokkos::initialize()`, and `csr.t.cpp` linked `Catch2WithMain` with no `Kokkos::ScopeGuard`. This was fixed by giving `csr.t.cpp` a custom `main()` with `Kokkos::ScopeGuard` and linking `Catch2::Catch2` + `Kokkos::kokkos` (instead of `Catch2WithMain`); the fix is tracked in [Cleanup Plan](../CLEANUP_PLAN.md) §0a. The `graph_node` API surface is correct under 5.1.1.

## Maturity & known gaps

**Verdict: mature.** The core path is production-load-bearing and recently maintained. `dense`/`circulant`/`inner_block`/`block`/`csr` + visitors are all constructed in `derivative.cpp`; the resulting `block`/`csr` matvecs and graph nodes drive every PDE (`gradient`/`laplacian`/`heat`/`scalar_wave`). Files were rewritten in Phases 13/15/16 (Kokkos parallelization), Phase 17 (Kokkos 5.0 TeamPolicy + Graph), and `block.hpp` touched as recently as Phase 19 (2026-03-27). No KokkosKernels/KokkosSparse anywhere — confirms the explicit-loop design. (The build is green as of 2026-06-04 and the test suite runs; all 7 `matrices` targets pass — `t-csr` was fixed by giving it a custom `main()` + `Kokkos::ScopeGuard`, see Gotchas.)

Verified items in this subsystem (from audited flags):

- **`dense::operator()` serial matvec — partial (vestigial/test-only).** The `dense` **class** is mature production STORAGE (built in `derivative.cpp`, read by `block::build_device_arrays` and the visitors via `data()`/`size()`); but its `operator()` matvec has **zero production callers** (only `t-dense` and, transitively, `t-inner_block`). It survives as the reference oracle `block.t.cpp` compares its TeamPolicy kernel against. Document-as-legacy, do not delete the class. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`inner_block::operator()` eager matvec — partial (test-only).** Same status: the class is load-bearing as a builder value object (`block` stores `std::vector<inner_block>` and reads `left()/interior_circ()/right()`), but the apply method has zero production callers after the Phase-17 TeamPolicy migration moved all real matvecs into `block::matvec_functor`. Only `t-inner_block` exercises it. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`coeffs_view()` on `dense` and `circulant` — dead (zero callers, safe to delete).** Speculative API added in Phase 17 "for the future TeamPolicy kernel"; that kernel was implemented via the block-level flattened `coeffs_d` (built through `data()`, not `coeffs_view()`), so these two inline accessors were never called anywhere in the repo. **Do not confuse with the live `block::coefficients_view()`** (different name, exercised by `block.t.cpp`). See [Cleanup Plan](../CLEANUP_PLAN.md).
- **Unused flag constants in `common.hpp` — dead (safe to delete the 5 symbols).** `colspace_r`, `detail::domain`, `detail::dirichlet`, `detail::right`, and `is_rz()` have zero callers (their siblings `colspace_rx/ry/rz`, `rowspace_*`, `ldd`, `rdd`, `is_ldd/is_rdd/is_rx/is_ry`, `matrix_base` are all actively used). Vestigial BC-tagging scaffolding; delete only these 5, not the file. See [Cleanup Plan](../CLEANUP_PLAN.md).

Items flagged but **refuted** by verification (i.e. actually mature — listed so a new dev doesn't re-flag them):

- **`block::rows()` "domain-edge inside object" comment** — refuted as partial; the case is unreachable (`domain_discretization` does not generate it), `rows()` is never on a compute path, and the disjointness invariant it depends on is now enforced by the construction-time debug assert in `to_block()`. The 2021 comment is a stale clarifying note.
- **Commented-out `fmt::print` / `assert` lines in `unit_stride_visitor.cpp`** — refuted as experimental; these are inert breadcrumbs (the file does not even include `<fmt/...>`), and the removed `assert`s in `mapped()` were superseded by the intended `-1` (unmapped) return contract. The file is fully tested and on the live eigenvalues path. Optional hygiene only.
- **`block::graph_node` / `csr::graph_node` "no direct tests"** — refuted as unknown; they have dedicated tests in `src/fields/graph_poc.t.cpp` (`"CSR graph_node"`, `"Block graph_node"`, `"Block + CSR graph chain"`) asserting byte-exact equivalence with the eager matvec, and the methods verified correct under Kokkos 5.1.1. (The `create_graph` driver call sites were the Kokkos 5.1 break, fixed 2026-06-04.)

## Tests

All 7 dedicated test files carry the `matrices` ctest label (run `ctest --test-dir build -L matrices`):

| Target | Covers |
| --- | --- |
| `t-dense` | square/non-square/strided eager matvec, identity, `plus_eq`. |
| `t-circulant` | identity/random/strided, both `eq` and `plus_eq`. |
| `t-inner_block` | identity/random-boundary/strided eager matvec incl. `ldd`/`rdd` column dropping (tests the now test-only apply path). |
| `t-block` | identity/random/strided eager matvec + "device metadata arrays" / "device metadata with stride" inspecting `metadata_view()`/`coefficients_view()`. |
| `t-csr` | identity/random direct + builder roundtrip (uses a custom `main()` with `Kokkos::ScopeGuard`, linking `Catch2::Catch2` + `Kokkos::kokkos`). |
| `t-unit_stride_visitor` | no-boundary/dirichlet/inner_block/csr index mapping. |
| `t-coefficient_visitor` | dense/inner-block/csr scatter into the dense global matrix. |

**Graph paths** are tested in `src/fields/graph_poc.t.cpp` (label `fields`, target `t-graph_poc`), not in `block.t.cpp`/`csr.t.cpp`.

**Not directly covered:** the `block::rows()` "last point inside object" edge case (flagged unreachable); the dead `coeffs_view()` accessors. The eager `inner_block`/`dense` matvecs are tested but the path they cover is production-dead (coverage protects a test-only oracle). No disabled/commented-out tests within the matrices test files. **All 7 matrices targets pass** (build fixed 2026-06-04); `t-csr` was fixed by giving `csr.t.cpp` a custom `main()` with `Kokkos::ScopeGuard` and linking `Catch2::Catch2` + `Kokkos::kokkos` (instead of `Catch2WithMain`), resolving the Kokkos 5.1 pre-`initialize()` OpenMP-exec-space abort — a harness fix (see Gotchas).

## Related docs

- [Core types](core-types.md) — `real`/`integer`/`int3`, `eq_t`/`plus_eq_t`, `MOVE`/`FWD`, the `Range` concept used by these ctors.
- [Fields](fields.md) — `Kokkos::View` buffers / `device_view` that back the matrix coeff storage and the `x`/`b` spans.
- Stencils reference — the upstream source of the circulant/dense/csr coefficients these matrices apply (see `docs/stencils.md`).
- Operators / derivative reference — the primary consumer (`derivative.cpp`) that assembles `block` + `csr` and drives both matvec paths.
- **Legacy (pre-Kokkos rationale archive):** `docs/matrices.md`, `docs/discrete_operators.md`, `docs/lazyness.md` — describe the original range-v3 lazy `operator*` design; useful for *why* the composite split exists, but the apply mechanism is now the Kokkos kernels documented here, not lazy ranges.
- Migration history: `plans/kokkos-view-migration-impact.md` (old `block→inner_block→dense` chain), `plans/17-kokkos5-upgrade-and-optimization.md` (TeamPolicy + Graph + the `coeffs_view()` additions), `plans/18-cleanup-and-dedup.md`, `plans/meta.md`.
