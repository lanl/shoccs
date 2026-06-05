# Core Types & Indexing (`src/shoccs_config.hpp`, `src/kokkos_types.hpp`, `src/indexing.hpp`, `src/index_extents.hpp`, `src/index_view.hpp`)

> **Maturity:** partial · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose
This is the bottom of the dependency stack. It defines the project's foundational scalar/vector type aliases (`real`, `integer`, `int3`, `real3`), the Kokkos execution/memory-space and view aliases (`execution_space`, `memory_space`, `device_view`), and the multi-dimensional `(i,j,k) -> flat` index mapping for structured Cartesian grids. The central type is `index_extents` — the grid-shape type carried by mesh, operators, io, fields and systems, and the value bound to the `index_extents = {nx, ny, nz}` Lua config key. Nearly everything in SHOCCS transitively pulls in these aliases.

Two helper headers (`src/types.hpp`, `src/real3_operators.hpp`) are functionally part of this subsystem even though `CLAUDE.md` only names the five index files: `indexing.hpp` and `index_extents.hpp` `#include "types.hpp"` (not `shoccs_config.hpp` directly), and `real3_operators.hpp` provides the vector math (`dot`/`length`/`clamp_lo`/elementwise ops) that operate on `real3`.

## Where it lives
| File | Role |
| --- | --- |
| `src/shoccs_config.hpp` | The lowest-level header. Scalar/vector aliases: `real`, `integer`, `int3`, `real3`, plus the unused `int2`/the lightly-used `real2`. |
| `src/kokkos_types.hpp` | Maps Kokkos spaces to project aliases. `execution_space = Kokkos::DefaultHostExecutionSpace` (host-only today), `memory_space`, `device_view<T>`. The single switch point for a future GPU port (plan 14). |
| `src/index_extents.hpp` | The central grid-shape type `index_extents{int3 extents}`: canonical row-major flat-index `operator()(int3)`, `size()`, `int3` conversions, `operator[]`, and full `std::tuple` protocol. Binds to the Lua `index_extents` key. |
| `src/indexing.hpp` | `namespace ccs::index`. Compile-time slow/fast axis ordering (`dir<I>`), runtime `dirs(int)` / `stride<I>(int3)` helpers used by mesh & operators. Also houses dead `transpose<>()` and `bounds<I>` code. |
| `src/index_view.hpp` | `index_view<I>(...)` — builds `std::vector<int3>` coordinate lists for a direction or a boundary plane. Orphaned (no production callers) but tested. |
| `src/types.hpp` | Functionally part of core-types. Re-includes `shoccs_config.hpp`; adds the `Numeric`/`NumericTuple`/`Range` concepts, `dim` enum, `index_slice`, `null_v`, the `FWD`/`MOVE` macros, `eq`/`plus_eq` accumulation policies, and a debug demangler under `!NDEBUG`. |
| `src/real3_operators.hpp` | Functionally part of core-types. `NumericTuple`-constrained elementwise `+ - * /` plus `dot`/`length`/`clamp_lo` over `real3` and tuple-likes. Used by mesh/sphere, io/xdmf, systems/scalar_wave, systems/inviscid_vortex. |
| `src/CMakeLists.txt` | Declares the (largely vestigial) `indexing` INTERFACE target and registers the three core-types unit tests (`t-indexing`, `t-index_view`, `t-real3_operators`). |

## Public API / entry points

### Scalar & vector aliases — `src/shoccs_config.hpp`
```cpp
namespace ccs {
using real    = double;
using real3   = std::array<real, 3>;
using real2   = std::array<real, 2>;   // used only in mesh/rect
using integer = long;                  // "prefer higher precision than regular int"
using int3    = std::array<int, 3>;
using int2    = std::array<int, 2>;     // DEAD: zero callers
}
```
Keeping vectors as `std::array` is load-bearing: it makes them satisfy the `NumericTuple` concept, so `real3_operators.hpp` arithmetic and the `index_extents` tuple protocol apply for free.

### Kokkos aliases — `src/kokkos_types.hpp`
```cpp
namespace ccs {
using execution_space = Kokkos::DefaultHostExecutionSpace;
using memory_space    = typename execution_space::memory_space;

template <typename T>
using device_view = Kokkos::View<T, memory_space>;
}
```
Host-only today. `device_view<T>` is currently just a host `Kokkos::View`. Used directly by the matrix headers (block/dense/circulant); `execution_space` is referenced in ~25 files.

### Grid-shape type — `src/index_extents.hpp`
```cpp
namespace ccs {
struct index_extents {
    int3 extents;

    constexpr operator const int3&() const;       // implicit conversion to int3
    constexpr operator int3&();

    // canonical row-major flat index: ijk[0]*ny*nz + ijk[1]*nz + ijk[2]
    constexpr integer operator()(int3 ijk) const;

    constexpr auto  operator[](int i) const;       // axis access
    constexpr auto& operator[](int i);

    constexpr integer size() const;                // extents[0]*extents[1]*extents[2]
};

// tuple protocol: get<I>, std::tuple_size, std::tuple_element are specialized,
// so structured bindings `auto&& [nx, ny, nz] = e;` work.
}
```
`operator()(int3)` returns `integer` (64-bit) computed from 32-bit `int3` axis values; the `nx/ny/nz` are cast to `integer` first to avoid 32-bit overflow on the product. It is exercised in production at `src/mesh/mesh.cpp:94-97` (`extents(start.mesh_coordinate)`). `class cartesian : public index_extents` (`src/mesh/cartesian.hpp:28`), so the whole mesh layer IS-A `index_extents`.

### Direction / index-ordering helpers — `src/indexing.hpp` (`namespace ccs::index`)
```cpp
struct index_pairs { int fast; int slow; };

template <int I> struct dir;            // compile-time slow/fast axes for direction I
//   dir<0>: slow=1, fast=2
//   dir<1>: slow=0, fast=2
//   dir<2>: slow=0, fast=1

constexpr index_pairs dirs(int i);      // runtime {fast, slow} for direction i
template <auto I> constexpr auto stride(int3 n);   // product of extents past axis I

// --- the rest of this header is DEAD (see Maturity & known gaps) ---
template <int AD, int BD> constexpr auto transpose(int3 a);   // test-only
template <int I> class bounds { ... int index(const int3&) ...; };  // zero callers
```
The `dir`/`dirs`/`stride` symbols are the live, load-bearing part — used by mesh.cpp, cartesian, rect and object_geometry.

### Coordinate-list generator — `src/index_view.hpp` (orphaned)
```cpp
template <int I = 2> std::vector<int3> index_view(int3 extents);          // volume
template <int I>     std::vector<int3> index_view(int3 extents, int i);   // plane at i
```
The plane overload wraps a negative `i` from the end (`i + extents[I]`), so `-1` means the last plane. No production consumers — treat as legacy (see gaps).

### Concepts, macros & helpers — `src/types.hpp`
```cpp
template <typename T> using span = std::span<T>;
template <typename T> concept Numeric      = std::integral<T> || std::floating_point<T>;
template <typename T> concept NumericTuple = /* tuple-like, all elements arithmetic */;
template <typename T> concept Range        = std::ranges::input_range<T> && !same_as<int3,...>;

enum class dim { X, Y, Z };
struct index_slice { integer first; integer last; };
template <typename T = real> constexpr auto null_v = std::numeric_limits<T>::max();

struct eq_t      { /* x  = y */ };   constexpr auto eq      = eq_t{};
struct plus_eq_t { /* x += y */ };   constexpr auto plus_eq = plus_eq_t{};

#define FWD(x)  static_cast<decltype(x)&&>(x)
#define MOVE(x) static_cast<std::remove_reference_t<decltype(x)>&&>(x)
```
`eq` / `plus_eq` are the matrix/operator accumulation policy objects. `debug::type<T>()` (a libstdc++ demangler) is only compiled when `!NDEBUG`.

### Vector math — `src/real3_operators.hpp`
```cpp
// elementwise binary ops over NumericTuple x NumericTuple, NumericTuple x scalar,
// scalar x NumericTuple — all return std::array<real, N>:
operator+  operator-  operator*  operator/

template <NumericTuple U, NumericTuple V> constexpr real dot(U&&, V&&);
template <NumericTuple U, Numeric V>      constexpr auto clamp_lo(U&&, V);  // max(elem, v)
template <NumericTuple U>                 real           length(U&&);       // sqrt(dot(u,u))
```
The binary operators are generated by the `SHOCCS_GEN_OPERATORS(op, acc)` macro. Because the operands need only satisfy `NumericTuple`, these work on `real3`, `std::tuple<real,real,real>`, and `cartesian_product_view`-style tuples alike (see the `real3/tuple` test case).

## How it works

**Slow/fast index ordering.** SHOCCS lays out 3D fields so that a direction-wise operator sees *its* direction contiguous. For a derivative applied in direction `I`, axis `I` is the contiguous (unit-stride) axis; of the remaining two, the one with the larger stride is `slow` and the smaller is `fast`. `dir<I>` encodes this table at compile time, and `dirs(i)` returns it at runtime as `{fast, slow}`. `stride<I>(n)` gives the flat-array stride for stepping along axis `I` (the product of all extents past `I`).

**The canonical flat index.** `index_extents::operator()(int3 ijk)` is plain row-major: `i*ny*nz + j*nz + k`. It has no slow/fast awareness — it is the single canonical address map for the registry's flat buffers, and it is what mesh.cpp uses to turn a mesh coordinate into a buffer offset. The grid shape flows in from Lua:

- `cartesian::from_lua` reads `simulation.mesh.index_extents` (`src/mesh/cartesian.cpp:55-57`) — it is **required**; missing it is a hard error.
- The parsed `int3` becomes `index_extents{n}` (`cartesian.cpp:89`), and `cartesian` inherits from `index_extents`, so the grid shape is carried by the mesh object itself.
- mesh / operators / io / fields / systems all consume that `index_extents` (by value, by conversion to `int3`, or by `operator[]`).

**Dependency arrival.** `indexing.hpp` and `index_extents.hpp` include `types.hpp`, which re-includes `shoccs_config.hpp`. So pulling in `index_extents.hpp` drags in the full `types.hpp` surface (concepts, `<ranges>`, `<span>`, the `FWD`/`MOVE` macros, and the debug demangler under `!NDEBUG`) — not just the bare aliases.

## How to extend

- **Add a scalar/vector alias:** edit `src/shoccs_config.hpp`. Keep it `std::array`-based so the `NumericTuple` concept and `real3_operators` pick it up automatically.
- **Add `real3`/vector math:** add a `NumericTuple`-constrained free function in `src/real3_operators.hpp`. For a new elementwise binary op, use the `SHOCCS_GEN_OPERATORS(op, acc)` macro (it emits the tuple/tuple, tuple/scalar, scalar/tuple overloads at once). For reductions (like `dot`), follow the index-sequence-fold pattern already there.
- **Add an index-ordering / direction helper:** add it to `namespace ccs::index` in `src/indexing.hpp`, following the `dir<I>::slow/fast` convention. Prefer the `index_extents::operator()` row-major flat-index convention for any new addressing — do **not** copy `bounds<I>::index()` (it is dead and uses a different, direction-aware convention).
- **Enable GPU (large, unstarted):** per `plans/14-kokkos-gpu.md`, change `execution_space` in `src/kokkos_types.hpp` to `Kokkos::DefaultExecutionSpace` (`memory_space` and `device_view` derive automatically). This is a cross-cutting effort — the `std::span` bridge in `fields` becomes UB on device — and has not been done.

## Gotchas & invariants
- **Two incompatible flat-index conventions coexist.** `index_extents::operator()(int3)` is plain row-major (`i*ny*nz + j*nz + k`, no slow/fast awareness). `index::bounds<I>::index()` reorders by `dir<I>::slow/fast` and gives *different* results. New code must use `index_extents`; `bounds<>` is dead.
- **`integer` is `long` (64-bit) but `int3` axes are 32-bit `int`.** `index_extents` stores extents as `int3` and casts each to `integer` *before* multiplying in `operator()`, so the resulting flat index is 64-bit even though axis values are 32-bit. Preserve that cast when touching the address map.
- **Including `index_extents.hpp`/`indexing.hpp` pulls in all of `types.hpp`.** You inherit concepts, ranges, span, the `FWD`/`MOVE` macros, and (in debug builds) `<cxxabi.h>`. They include `types.hpp`, not `shoccs_config.hpp` directly.
- **`device_view<T>` is a host view today.** `view.data()` wrapped in `std::span` works now but is documented (plan 14) to become UB once `execution_space` switches to GPU. Do not rely on host-pointer access surviving a GPU switch.
- **The `indexing` CMake INTERFACE target is not linked by any subsystem.** `src/CMakeLists.txt:3-4` declares it and links Kokkos, but subsystems get the headers via their own `target_include_directories` pointing at `src/` and link `Kokkos::kokkos` directly. The target is only the link arg for `t-indexing` and `t-index_view`. Adding code to `indexing.hpp` that needs a real link dependency will silently not reach most consumers.
- **`int2` is defined but used nowhere; `real2` is used only in `src/mesh/rect.{hpp,cpp}`.**
- **`Range` concept deliberately excludes `int3`** to avoid ambiguous overloads when an `int3` is passed where either a range or an `int3` would match.

## Maturity & known gaps
**Verdict: partial.** The *core* is mature — `index_extents` is used at 40+ call sites (mesh, all operators, io/xdmf, io/field_data, fields/selection_desc, every system test) and is a mandatory live Lua key; the scalar aliases are universal; `execution_space`/`memory_space` appear in ~25 files; `device_view` in three matrix headers; `dir`/`dirs`/`stride` have real callers in mesh/cartesian/rect/object_geometry; `real3_operators` (`dot`/`length`/`clamp_lo`) are used across mesh and systems. The `t-indexing`, `t-index_view`, and `t-real3_operators` tests PASS (they link no Kokkos runtime, so they are unaffected by the stale-build link failures hitting other tests). The "partial" label is driven entirely by several dead/orphaned pieces that a newcomer should not build on:

- **`index_view.hpp` (both overloads) — DEAD.** Zero production includes; the only reference is its own test `src/index_view.t.cpp`. It replaced cppcoro generators in the Phase 0 migration (commit b3fc5a2); nothing in the live data flow consumes the returned `std::vector<int3>`, and its intended replacement is `Kokkos::MDRangePolicy`. The volume overload `index_view<I>(extents)` has *no* caller at all (not even the test, which only exercises the plane overload). Verdict: deprecate. See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`index::transpose<AD,BD>(int3)` — DEAD.** Only caller is `src/indexing.t.cpp` (the `transpose` test case); a self-justifying test artifact, body untouched (except whitespace) since 2020. Its helper `detail::pos` is used only by `transpose`. Verdict: delete (function + test case). See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`index::bounds<I>` (class + `index()`) — DEAD.** Zero callers anywhere, including its own test. A pre-Kokkos, direction-aware flat-index that is superseded by `index_extents::operator()`. Last touched 2021. Verdict: delete lines `86-105` only (keep the sibling `dir`/`dirs`/`stride`/`index_pairs`). See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`int2` alias — DEAD.** Zero callers since its 2021 introduction. Verdict: delete `src/shoccs_config.hpp:14` (zero risk). See [Cleanup Plan](../CLEANUP_PLAN.md).
- **`indexing` INTERFACE CMake target — vestigial but not "partial".** Re-audit verdict: **mature** test-support shim, not partial. It is never linked by a subsystem (headers travel via include paths), but it does exactly its job: hand Kokkos + the `src/` include path to `t-indexing` and `t-index_view`. Optional cleanup is to inline `Kokkos::kokkos` into those two tests and drop the target; do **not** remove it without that refactor or both tests lose their Kokkos/include path.
- **`index_extents` has no dedicated test — coverage note, not a maturity gap.** It is unambiguously mature (production-critical, mandatory Lua key, base class of `cartesian`), but its `operator()(int3)` flat-index map, `size()`, and tuple-protocol specializations have no *direct* assertions — coverage is incidental via mesh/selection tests. Adding `src/index_extents.t.cpp` would lock in this behavior; worth doing during onboarding.

## Tests
- `src/indexing.t.cpp` — label **indexing** (`t-indexing`). Covers **only** `index::transpose` round-trips (which exercises `dir<I>` indirectly). Does **not** test `dirs`, `stride`, or `bounds`.
- `src/index_view.t.cpp` — label **indexing** (`t-index_view`). Covers the plane overload `index_view<I>(extents, i)` thoroughly (positive and negative plane offsets). The volume overload `index_view<I>(extents)` is **not** called by any test.
- `src/real3_operators.t.cpp` — label **real3** (`t-real3_operators`). Covers `+ - * /`, `dot`, `length` on real3/real3, real3/Numeric, and real3/`std::tuple`.
- **Not covered:** `index_extents` has no dedicated test file (no direct assertion of `operator()(int3)`, `size()`, or the tuple-protocol specializations). `index::dirs`/`stride` are untested. `index::bounds<I>::index()` is uncalled and untested. `kokkos_types.hpp` aliases are validated only by downstream compilation. No disabled tests in this subsystem.

## Related docs
- [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md) · [Cleanup Plan](../CLEANUP_PLAN.md)
- Reference docs for the immediate consumers: **fields**, **mesh**, **matrices**, **operators** (all depend on these aliases / `index_extents` / `device_view`).
- `plans/14-kokkos-gpu.md` — the planned (unstarted) GPU port; the single reason `kokkos_types.hpp` is structured as one switch point.
- `plans/00-foundation.md` and `SHOCCS_ARCHITECTURE_AND_KOKKOS_MIGRATION_SPEC.md` — record the `index_view(int3) -> Kokkos::MDRangePolicy` migration intent (pre-Kokkos rationale archives; do not read as current architecture).
- **Stale, do not trust for onboarding:** `design.md` predates the Kokkos migration and frames the field API around range-v3 selections. `CLAUDE.md` Architecture item 1 overstates "Indexing ... all subsystems depend on this": the *scalar aliases* and `index_extents` are near-universal, but `index_view.hpp` has zero production callers and `indexing.hpp` is included only by mesh + index_view.
