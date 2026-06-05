# Fields (`src/fields/`)

> **Maturity:** mature · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose
The `fields` subsystem owns every numerical buffer in the solver and provides the element-wise algebra used to fill them. `field_registry` holds all field data as a flat array of `Kokkos::View<real*>`, organized by *slot* (time-level) and by a fixed per-field buffer layout. Lightweight tokens (`field_ref`, `scalar_handle`, `vector_handle`) index into that storage without owning anything. On top of storage it supplies a 4-component access view (`scalar_span`/`scalar_view`), expression-template leaf nodes that drive `Kokkos::parallel_for` assignment kernels, and selection descriptors that apply boundary conditions over field subsets. This is the central data-flow hub: systems, operators, integrators, and I/O all read and write through it.

## Where it lives
| File | Role |
|------|------|
| `src/fields/field_registry.hpp` | Owns all buffers as `std::array<Kokkos::View<real*>, MaxSlots*buffers_per_slot>`. Defines `field_ref`, `system_size`, the `extract_scalar_span`/`extract_scalar_view` bridge, and the sole concrete type `sim_registry = field_registry<8,8,4>`. |
| `src/fields/handle.hpp` | Compile-time index arithmetic: `field_layout<MaxS,MaxV>`, `buf_handle`/`scalar_handle`/`vector_handle`, and the `consteval` `make_*_handle` factories. Defines the D/Rx/Ry/Rz and x/y/z buffer layout. |
| `src/fields/scalar.hpp` | `scalar_span`/`scalar_view` — the 4-component `{D, Rx, Ry, Rz}` `std::span` wrappers that operators and systems actually compute on. |
| `src/fields/expr.hpp` | Expression-template leaves (`handle_expr`, `scalar_literal_expr`), composite nodes (`binary_expr`, `unary_expr`), `parallel_for` `assign`/compound-assign kernels, and `contains_ptr` aliasing detection. |
| `src/fields/selection_desc.hpp` | BC selection descriptors (`contiguous`/`strided`/`gather`), plane/gather factories, `assign_selected`/`fill_selected`/`plus_assign_selected`, and `for_each_grid_bc_desc`. |
| `src/fields/lazy_views.hpp` | Project-local C++ range polyfills (`repeat_n`, `stride`, `cartesian_product`, `linear_distribute`) + a `std::basic_common_reference<tuple,...>` backport. Physically in `fields/` but is a cross-cutting utility used by `mesh`/`matrices`/`stencils`/`io`, **not** by other fields files. |
| `src/fields/graph_poc.t.cpp` | Kokkos Graph API regression test. Labeled `fields` and located here but exercises only `matrices::csr`/`block` `graph_node` plus raw `Kokkos::Graph` — misfiled, belongs under `matrices`. |

## Public API / entry points

### Storage: `field_registry<MaxSlots, MaxS, MaxV>` (`field_registry.hpp`)
The one concrete instantiation is `using sim_registry = field_registry<8, 8, 4>` (8 slots, up to 8 scalars and 4 vectors per slot). Always use `sim_registry` in solver code.

```cpp
// Allocation (strictly sequential per slot; see Gotchas)
field_ref allocate_scalar(int slot, int scalar_index,
                          int d_sz, int rx_sz, int ry_sz, int rz_sz);
field_ref allocate_vector(int slot, int vector_index,
                          int d_sz, int rx_sz, int ry_sz, int rz_sz);

// Access (h is any buf_handle, e.g. scalar_handle{0}.D())
      Kokkos::View<real*>& view(field_ref ref, buf_handle h);
const Kokkos::View<real*>& view(field_ref ref, buf_handle h) const;
      real* data(field_ref ref, buf_handle h);          // .data() of the View
const real* data(field_ref ref, buf_handle h) const;
int          size(field_ref ref, buf_handle h) const;   // extent(0) as int

// Bulk slot operations (time-stepping)
void deep_copy_slot(int dst, int src);  // copies all buffers dst <- src
void swap_slots(int a, int b);          // swaps buffers + metadata
```

Sizing/identity tokens:
- `struct field_ref { int slot = -1; int n_scalars = 0; int n_vectors = 0; }` — trivially copyable, `sizeof == 12`, fits in SBO. Returned by `allocate_*`; carries a slot's *allocation state*, not a per-buffer index.
- `struct system_size { integer nscalars, nvectors, d_size, rx_size, ry_size, rz_size; }` — plain sizing token with defaulted `operator<=>`.

Span bridge (free functions):
```cpp
scalar_span extract_scalar_span(field_registry& reg, field_ref ref, scalar_handle h);
scalar_view extract_scalar_view(const field_registry& reg, field_ref ref, scalar_handle h);
```

### Handles: `handle.hpp`
```cpp
template <int MaxS, int MaxV> struct field_layout {
    static constexpr int scalar_stride = 4;   // D, Rx, Ry, Rz
    static constexpr int vector_stride = 12;  // 3 components x 4 buffers
    static constexpr int vector_base   = MaxS * scalar_stride;
    static constexpr int total_buffers = vector_base + MaxV * vector_stride;
    int n_scalars = 0, n_vectors = 0;         // active counts
};

struct buf_handle    { int id   = -1; };                  // a single buffer index
struct scalar_handle { int base = -1;                      // = scalar_index * 4
    buf_handle D(), Rx(), Ry(), Rz();
    std::array<buf_handle,4> all();   // {D,Rx,Ry,Rz}
    std::array<buf_handle,3> R();     // {Rx,Ry,Rz}
};
struct vector_handle { int base = -1;                      // = vector_base + vector_index*12
    scalar_handle x(), y(), z();      // components as scalar_handles
    buf_handle Dx(),Dy(),Dz(), xRx(),xRy(),xRz(), yRx(),yRy(),yRz(), zRx(),zRy(),zRz();
    std::array<buf_handle,12> all();
    std::array<scalar_handle,3> components();
};
```
All handle types are trivially-copyable aggregates with defaulted `operator==`, so they are valid C++20 NTTPs. The `consteval` factories `make_scalar_handle(layout, i)` / `make_vector_handle(layout, i)` give compile-time bounds checking but are **test-only** (production builds handles with direct arithmetic, e.g. `scalar_handle{0}` — see Maturity).

### 4-component access: `scalar.hpp`
```cpp
struct scalar_span {                       // mutable
    std::span<real> D, Rx, Ry, Rz;
    scalar_span& operator=(T val);         // broadcast fill, T arithmetic  (e.g. du = 0)
    scalar_span& operator=(Fn&& fn);       // functional, Fn invocable on scalar_span&
};                                         //   (e.g. u_rhs = lap(u, nu))
struct scalar_view {                       // read-only; implicitly converts from scalar_span
    std::span<const real> D, Rx, Ry, Rz;
};
```
This is the type operators (`derivative`/`gradient`/`laplacian`) and systems (`heat`/`scalar_wave`) compute on. The functional `operator=` is how `du_sp = lap(u)` works: `laplacian::operator()` returns an invocable that mutates the destination span.

### Expression nodes & kernels: `expr.hpp`
```cpp
struct handle_expr        { real* ptr;   real operator()(int i) const { return ptr[i]; } };
struct scalar_literal_expr{ real value;  real operator()(int) const { return value; } };
template <class Op,class Lhs,class Rhs> struct binary_expr { Op op; Lhs lhs; Rhs rhs; ... };
template <class Op,class Arg>           struct unary_expr  { Op op; Arg arg; ... };

bool contains_ptr(const Expr&, const real* target);     // aliasing check

template <class Expr> void assign       (real* dst, int n, Expr e);  // alias-safe (stages temp)
template <class Expr> void plus_assign  (real* dst, int n, Expr e);  // dst[i] += e(i)
template <class Expr> void minus_assign (real* dst, int n, Expr e);
template <class Expr> void times_assign (real* dst, int n, Expr e);
template <class Expr> void divide_assign(real* dst, int n, Expr e);
void times_assign_scalar(field_registry&, field_ref, scalar_handle, real value); // all 4 bufs
```
**There is no `operator+`/`operator*` DSL** — expression trees are built by hand, e.g. `binary_expr{std::plus<>{}, handle_expr{a}, binary_expr{std::multiplies<>{}, ...}}`. Production uses only the leaf nodes (`handle_expr`, `scalar_literal_expr`) with the `_selected` helpers below; the composite nodes and bare `assign()` are tested/benchmarked infrastructure (see Maturity).

### Selection descriptors: `selection_desc.hpp`
A descriptor is any trivially-copyable struct exposing `KOKKOS_INLINE_FUNCTION int element(int) const` and `int count() const`.
```cpp
struct contiguous_selection { int offset_, count_; };                     // x-plane
struct strided_selection    { int offset_, inner_count_, outer_count_, outer_stride_; }; // y/z-plane
struct gather_selection     { Kokkos::View<const int*, memory_space> indices_; };         // fluid/object

// Plane factories (extents {nx,ny,nz})
contiguous_selection make_x_plane_desc(index_extents, int i);
strided_selection    make_y_plane_desc(index_extents, int j);
strided_selection    make_z_plane_desc(index_extents, int k);
// Gather factories
gather_selection make_gather_from_slices(std::span<const index_slice>);
template <class Pred> gather_selection make_gather_from_predicate(std::span<const mesh_object_info>, Pred);

// Kernels over selected elements
template <class Desc,class Expr> void assign_selected     (real* dst, Desc, Expr); // dst[idx]  = e(idx)
template <class Desc>            void fill_selected       (real* dst, Desc, real value);
template <class Desc,class Expr> void plus_assign_selected(real* dst, Desc, Expr); // dst[idx] += e(idx)

// Iterate the 6 grid faces whose BC == B (compile-time), calling fn(desc) per match
template <bcs::type B, class GridT, class Fn>
void for_each_grid_bc_desc(const GridT& g, index_extents ext, Fn fn);
```

### Range polyfills: `lazy_views.hpp`
```cpp
inline constexpr repeat_n_fn         repeat_n{};          // n copies of a value (lazy view)
inline constexpr stride_fn           stride{};            // every k-th element (lazy view)
inline constexpr cartesian_product_fn cartesian_product{};// product of ranges (lazy view)
template <class T> std::vector<T> linear_distribute(T mn, T mx, int n); // eager, n evenly-spaced points
```
(Note: this header is unrelated to field storage — see Maturity/Gotchas.)

## How it works

**Storage model.** A `field_registry<MaxSlots,MaxS,MaxV>` is a flat `std::array<Kokkos::View<real*>, MaxSlots * buffers_per_slot>`, where `buffers_per_slot = field_layout<MaxS,MaxV>::total_buffers = 4*MaxS + 12*MaxV`. A *slot* is a complete set of fields at one time-level (RK/Euler stages use multiple slots). Within a slot:
- scalar field `i` occupies indices `[i*4 .. i*4+3]` as `{D, Rx, Ry, Rz}`;
- vector field `i` occupies `[vector_base + i*12 .. +11]` as `x{D,Rx,Ry,Rz}, y{...}, z{...}`.

Each field is split into a **domain** buffer `D` (interior cell values) plus three **cut-cell boundary** buffers `Rx`, `Ry`, `Rz` (object/boundary intersection values per direction). A buffer is addressed as `buffers_[ref.slot * buffers_per_slot + handle.id]`.

**Handles carry only indices.** `field_ref` names a slot and its allocation counts. `scalar_handle{base}`/`vector_handle{base}` name a field within the slot; their `D()`/`Rx()`/... accessors return `buf_handle`s. No handle stores a length — sizes are queried from the registry View at launch time (`reg.size(ref, h)`).

**Access flow.** `allocate_scalar(...)` returns a `field_ref`. To compute, bridge to spans: `extract_scalar_view(reg, ref, scalar_handle{0})` yields a read-only `scalar_view` and `extract_scalar_span(...)` a mutable `scalar_span` (each holding four `std::span`s). Operators consume these directly; e.g. `heat::rhs`:

```cpp
auto u     = extract_scalar_view(reg, input, scalar_handle{0});
auto u_rhs = extract_scalar_span(out_reg, output, scalar_handle{0});
u_rhs = lap(u, scalar_view{...});                    // functional operator=
times_assign_scalar(out_reg, output, sh, diffusivity);
```

**Assignment kernels.** `assign`/`plus_assign`/... wrap a `Kokkos::parallel_for` over `RangePolicy<execution_space>`. The `Expr` (a `handle_expr`/literal/composite tree) is captured by value into a `KOKKOS_LAMBDA` and evaluated per index. `assign()` first calls `contains_ptr(expr, dst)`; if the destination aliases an input it stages through a temporary `Kokkos::View` then `deep_copy`s back. Compound-assigns skip that check (element-local, always safe).

**BC application.** Selection descriptors replace old iterator-based selectors on the hot path. A plane factory builds a `contiguous`/`strided` descriptor from mesh extents; gather factories build a `gather_selection` (a `Kokkos::View<int*>` of indices) from fluid slices or an object predicate. `assign_selected`/`fill_selected`/`plus_assign_selected` then run `parallel_for(0, desc.count())`, mapping thread `i` to `desc.element(i)`. `for_each_grid_bc_desc<bcs::Dirichlet>(grid, ext, fn)` visits the 6 faces, calling `fn(desc)` for each face whose BC matches `B`.

## How to extend

**Add a field to a system** (most common). In the system's `initialize`/builder:
1. If you need more capacity than `field_registry<8,8,4>`, bump `MaxS`/`MaxV` in the `sim_registry` alias at the bottom of `field_registry.hpp`.
2. Allocate **sequentially** per slot: `reg.allocate_scalar(slot, idx, d_sz, rx_sz, ry_sz, rz_sz)` where `idx` must equal the slot's current scalar count (you cannot skip indices). Same for `allocate_vector`.
3. Access via `scalar_handle{idx * 4}` (or `vector_handle{vector_base + idx*12}`) and `reg.data(ref, sh.D())`, or grab a 4-span view with `extract_scalar_span`/`extract_scalar_view`. Copy the pattern in `src/systems/heat.cpp` (`rhs`, lines ~119-162).

**Add a new BC selection pattern.** Define a trivially-copyable struct with `KOKKOS_INLINE_FUNCTION int element(int) const` and `int count() const` (the `gather_selection` `Kokkos::View` exception is documented). It works with `assign_selected`/`fill_selected`/`plus_assign_selected` unchanged. Pattern: `selection_desc.hpp` `contiguous_selection`.

**Add an expression operation.** Define a trivially-copyable `Op` functor (e.g. a `std::function`-free lambda or `struct`) and wrap leaves in `binary_expr{op, lhs, rhs}` / `unary_expr{op, arg}`, then pass to `assign_selected`/`plus_assign_selected`. Remember: no operator sugar — build nodes by hand. See `benchmarks/bench_expr.cpp` for full trees.

**Add a range polyfill.** Add a `*_view` class plus a `*_fn` CPO struct and `inline constexpr *_fn name{};` in `lazy_views.hpp`, following `repeat_n`/`stride`/`cartesian_product`.

## Gotchas & invariants
- **Host-only invariant.** `assign`/`plus_assign`/`assign_selected`/`fill_selected`/`scalar_span::operator=` capture **raw `real*` pointers** in their `KOKKOS_LAMBDA` and rely on `execution_space` being the synchronous `DefaultHostExecutionSpace`. The headers warn that device/async execution would need `Kokkos::View` capture instead — porting this layer to GPU will break it.
- **Sequential allocation.** `allocate_scalar`/`allocate_vector` assert `scalar_index == metadata_[slot].n_scalars` (resp. vectors). You cannot allocate index 2 before index 1, nor into arbitrary slots out of order.
- **Slot shape must match.** `swap_slots` asserts both slots have identical `n_scalars`/`n_vectors`; `deep_copy_slot` asserts matching extents per buffer. Mismatch aborts under assertions / corrupts under `NDEBUG`.
- **Unallocated buffers are zero-extent Views:** `data()` returns `nullptr`, `size()` returns 0. `deep_copy_slot` silently skips zero-extent **source** buffers, so copying from a partially-allocated slot is a no-op for the empty buffers (not an error).
- **`bcs::type` forward-declaration trick.** `selection_desc.hpp` declares `namespace bcs { enum class type; }` to avoid pulling in the heavy `operators/boundaries.hpp` (spdlog/fmt). Callers of `for_each_grid_bc_desc` **must include `operators/boundaries.hpp` themselves** or get incomplete-type errors.
- **`strided_selection::element` divides by `inner_count_`,** so `inner_count_ > 0` is required (documented invariant); the z-plane uses `inner_count_ = 1`.
- **Aliasing only in `assign()`.** Only `assign()` runs the `contains_ptr` check and stages through a temp. The compound-assigns (`plus`/`minus`/`times`/`divide`) skip it by design — do not assume they are alias-safe for non-element-local expressions.
- **`lazy_views.hpp` is cross-cutting.** It lives in `fields/` but is included by `mesh`/`matrices`/`stencils`/`io`, not by any fields header. Editing it affects those subsystems. The `ccs::stride` here is unrelated to `matrix_base::stride()`.

## Maturity & known gaps
**Verdict: mature.** Storage and access are load-bearing and used across ~25 non-fields files: `field_registry`/`field_ref`/`sim_registry` in `systems`/`temporal`/`simulation`; `scalar_span`/`scalar_view` in `operators`/`io`/`systems`; selection descriptors in `mesh`/`operators`/`systems`. Four dedicated test executables (`t-field_registry`, `t-selection_desc`, `t-expr`, `t-handle`) are registered with the `fields` label, and git history shows deliberate multi-phase construction (Phases 8-10 infrastructure through Phase 18 cleanup).

**Build-reality note.** The build is **fixed and green** (2026-06-04). The Kokkos 5.0 → 5.1.1 upgrade had changed the `Kokkos::Experimental::create_graph` overload set (now 1-arg `create_graph(Closure&&)`); all 17 call sites were migrated to the templated `create_graph<execution_space>(closure)` form. The graph call sites live in `operators`/`systems` plus this subsystem's `graph_poc.t.cpp`. `cmake --build build` builds the whole tree and `ctest --test-dir build` is 47/48; all four Kokkos-runtime fields tests (`t-field_registry`/`t-expr`/`t-selection_desc`, plus `t-handle`) pass. See `docs/CLEANUP_PLAN.md` §0a for the one remaining failure (`t-laplacian`); `t-csr` and `t-E2_1` have since been fixed (none of these are in `fields`).

Verified flags within this subsystem (all confirmed as **keep**, not dead — none are zero-caller deletions):
- **Composite `binary_expr`/`unary_expr` + bare `assign()`/`minus_assign`/`divide_assign`** (`expr.hpp`) — *infrastructure, not on the production hot path.* No production `.cpp` constructs composite nodes or calls bare `assign`/`minus_assign`/`divide_assign`; production uses leaf nodes with the `_selected` helpers, and `times_assign` is reached only via `times_assign_scalar`. These are complete, statically-asserted, unit-tested (`t-expr`), benchmarked (`bench_expr`), and were explicitly **kept** in the Phase 18 dead-code pass as forward-looking (GPU/device) design. Do not delete. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **No `dst = a + alpha*b` operator DSL** (`expr.hpp`) — *documentation drift, not a missing feature.* CLAUDE.md still advertises algebraic syntax; the `scalar_expr` operator-overload layer was **deliberately deleted** in Plan 18.3b as unused. The lean primitive API is the intended surface. Fix is doc-side. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`consteval make_scalar_handle`/`make_vector_handle`** (`handle.hpp`) — *partial: test-only prototype.* Zero callers outside `handle.t.cpp`; their compile-time-bounds-check benefit is architecturally inapplicable to the real allocation path, which uses **runtime** indices (see the comment at `field_registry.hpp:79-81`). The invalid-index rejection is not tested. Kept as cheap scaffolding through Phase 18. Document as experimental; deletion would be defensible but is not required. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`graph_poc.t.cpp`** (`fields/`) — *mature but misfiled.* Provides real regression coverage for `matrix::csr::graph_node`/`matrix::block::graph_node` (production RHS-graph path), but includes no fields header, exercises only `matrices`, and is labeled `fields`. Recommended cleanup: rename to `matrices/graph_node.t.cpp`, move under `src/matrices/`, relabel ctest to `matrices`, drop the `fields` link. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`lazy_views.hpp`** (`fields/`) — *mature but mislocated, stale docs.* Fully implemented and live (`mesh/cartesian`, `io/xdmf`), but unrelated to field storage and used only by other subsystems. CLAUDE.md's Code Conventions wrongly lists `zip_transform` (removed in task 12.3b3) and `bind_back` (never existed) among the `lazy_views.hpp` polyfills. Optional relocation to `src/utility/`; fix the doc drift. See [Cleanup Plan](../CLEANUP_PLAN.md).

CLAUDE.md doc-staleness to be aware of: the `dst = a + alpha * b` claim and "lightweight handle (slot index)" gloss on `field_ref` are imprecise; the `zip_transform`/`bind_back` polyfill claim is false.

## Tests
All registered under the **`fields`** label (`src/fields/CMakeLists.txt`):
- **`t-field_registry`** (`field_registry.t.cpp`): scalar/vector/mixed/sequential allocation, `view`/`data`/`size` access, unallocated-slot sentinels, `deep_copy_slot`, `swap_slots`, `extract_scalar_span`/`view`, span-bridge integration, `field_ref` SBO size. *Custom `main()` with Kokkos `ScopeGuard`.*
- **`t-selection_desc`** (`selection_desc.t.cpp`): element/count and trivial-copyability of all three descriptors, plane flat-index cross-checks, `assign`/`fill`/`plus_assign_selected` over each descriptor kind, `make_gather_from_slices`/`predicate` edge cases (empty/single/disjoint/all-match), `for_each_grid_bc_desc` face selection, one `assign_selected` + `scalar_literal_expr` case.
- **`t-expr`** (`expr.t.cpp`): `handle_expr`/`scalar_literal_expr`/`binary_expr`/`unary_expr` evaluation, nested `(a+b)*c`, `contains_ptr` aliasing, all of `assign`/`plus`/`minus`/`times`/`divide_assign`/`times_assign_scalar`.
- **`t-handle`** (`handle.t.cpp`): `field_layout` arithmetic, handle accessors, `consteval` factory happy-path. Uses `add_unit_test()` (links `Catch2WithMain`, no Kokkos runtime) — *the one fields test with no Kokkos runtime dependency.*

Coverage gaps: the bare (non-`_selected`) `assign`/`minus_assign`/`divide_assign` and the `assign()` alias-temp branch are not directly exercised in production; `scalar_span`'s broadcast/functional `operator=` has no dedicated fields test (covered indirectly by `t-laplacian`/`t-gradient` in `operators/`, which assert numerical correctness); the `make_*_handle` invalid-index compile error is untested; `lazy_views.hpp` has no fields tests (tested incidentally via `mesh`/`matrices`/`stencils`). No disabled or commented-out tests. `graph_poc.t.cpp` is labeled `fields` but is really a `matrices` test.

## Related docs
- [Operators reference](operators.md) — consumes `scalar_span`/`scalar_view` and selection descriptors; owns the `create_graph` call sites (migrated to the 1-arg form, fixed 2026-06-04).
- [Systems reference](systems.md) — `heat`/`scalar_wave` are the canonical callers of `extract_scalar_*`, `handle_expr`, and the `_selected` helpers.
- [Temporal reference](temporal.md) — uses `deep_copy_slot`/`swap_slots` and `field_ref` slots across time steps.
- [Matrices reference](matrices.md) — true home of the `graph_node` methods exercised by `graph_poc.t.cpp`.
- [Capability Audit](../CAPABILITY_AUDIT.md) · [Cleanup Plan](../CLEANUP_PLAN.md) · [Onboarding](../ONBOARDING.md)
- Legacy design (pre-Kokkos rationale archive): `plans/08-registry-and-handles.md`, `plans/10-expression-templates.md` (D-ET1/ET2/ET3 node + aliasing design), `plans/18-cleanup-and-dedup.md` (Phase 18 keep/delete decisions). These document *why* the current API looks the way it does, including the deliberately-removed operator DSL.
