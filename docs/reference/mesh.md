# Mesh (`src/mesh/`)

> **Maturity:** mature · **Audited:** 2026-05-29 · See [Capability Audit](../CAPABILITY_AUDIT.md) · [Onboarding](../ONBOARDING.md)

## Purpose
The mesh subsystem defines the discretization domain. It pairs a uniform Cartesian grid (`cartesian`) with embedded cut-cell geometry (`object_geometry`) that ray-casts grid lines against embedded `shape`s (spheres, axis-aligned rects) to find fluid/solid intersection points and the 1D cut-cell distance `psi` at each one. The `mesh` class composes these two, builds per-direction `line` lists (pairs of domain-wall / object boundaries) and fluid-point selection descriptors. Every spatial operator and PDE system depends on it: it is the geometric backbone of the solver.

## Where it lives

| File | Role |
| --- | --- |
| `src/mesh/mesh.hpp` | Public face: the `mesh` class aggregating `cartesian` + `object_geometry`, exposing `lines`/`R`/`fluid_desc`/`*_object_desc`/`interp_line`/`dirichlet_line`. |
| `src/mesh/mesh.cpp` | Builds per-direction `line` lists (`init_line`), fluid slices/selection (`init_slices`); implements `interp_line`, `dirichlet_line`, and `mesh::from_lua` (the real config entry point). |
| `src/mesh/cartesian.hpp` / `cartesian.cpp` | Uniform Cartesian grid: 1D coordinate arrays, spacings, dims; `cartesian::from_lua` parses `index_extents` + `domain_bounds`. Derives from `index_extents`. |
| `src/mesh/object_geometry.hpp` / `object_geometry.cpp` | Heart of cut-cell geometry: ray-casts each grid line against all shapes (`init_line<I>` + `closest_hit`), computes `psi` (snap+clamp logic), marks solid points (`init_solid`), and parses shapes from Lua (`from_lua` — defines which shape *types* are supported). Most recently modified mesh file (Phase 26.6 psi fix). |
| `src/mesh/shapes.hpp` | The `Shape` concept, the type-erased `shape` value class, `hit_info`, and the `make_*` factory declarations. The extension point for new geometry. |
| `src/mesh/sphere.cpp` | Sphere shape (quadratic ray–sphere intersection + radial normal). One of two Lua-reachable shapes. |
| `src/mesh/rect.hpp` / `rect.cpp` | Axis-aligned planar `rect<I>` template + `make_{xy,xz,yz}_rect` factories. Only `yz_rect` is wired into Lua config. |
| `src/mesh/mesh_types.hpp` | Shared POD structs: `mesh_object_info`, `boundary`, `object_boundary`, `line`, `domain_extents`. |
| `src/ray.hpp` | `ray{origin, direction}` with `position(t)`. (Lives at `src/ray.hpp`, not in `src/mesh/`.) |
| `src/mesh/CMakeLists.txt` | Defines `shoccs-mesh` and the four tests (`t-cartesian`, `t-object_geometry`, `t-shapes` via `add_unit_test`; `t-mesh` wired manually because it needs Kokkos + `shoccs-random`). |

## Public API / entry points

### `mesh` (`mesh.hpp`) — the class everyone uses
Construction:
```cpp
mesh(const index_extents& extents, const domain_extents& bounds, const logs& = {});
mesh(const index_extents& extents, const domain_extents& bounds,
     const std::vector<shape>& shapes, const logs& = {});
static std::optional<mesh> from_lua(const sol::table&, const logs& = {});
```
Grid queries (forwarded to `cartesian`):
```cpp
constexpr auto    size()  const;           // total cell count (integer)
constexpr int     dims()  const;           // number of active dimensions
constexpr std::span<const real> x()/y()/z() const;
constexpr real    h(int i) const;          // spacing in direction i
constexpr real3   h()      const;
constexpr decltype(auto) extents() const;  // const index_extents&
constexpr auto    stride(int dir) const;   // index::stride<dir>(extents())
constexpr integer ic(int3 ijk) const;      // int3 -> flattened linear index
```
Cut-cell intersection access (spans into the merged, per-direction `mesh_object_info` buffers):
```cpp
std::span<const mesh_object_info> Rx()/Ry()/Rz() const;
std::span<const mesh_object_info> R(int dir) const;   // dir 0/1/2
std::array<std::span<const mesh_object_info>,3> R() const;
```
Lines + boundary classification:
```cpp
const std::vector<line>& lines(int i) const;          // per-direction line list
line interp_line(int dir, int3 pt) const;             // bounding boundaries for interpolation at pt
bool dirichlet_line(const int3& start, int dir, const bcs::Grid&) const;
```
Selection descriptors (consumed by operators/systems to scatter/gather BCs):
```cpp
const gather_selection& fluid_desc() const;
gather_selection dirichlet_object_desc(int dir, const bcs::Object&) const;
gather_selection non_dirichlet_object_desc(int dir, const bcs::Object&) const;
```

### `cartesian` (`cartesian.hpp`) — uniform grid (derives from `index_extents`)
```cpp
cartesian(span<const int> n, span<const real> min, span<const real> max);
static std::optional<std::pair<index_extents, domain_extents>>
       from_lua(const sol::table&, const logs& = {});
umesh_line line(int i) const;              // {min, max, h, n} for direction i
int dims() const; integer size() const; integer plane_size(int i) const;
std::span<const real> x()/y()/z() const;
real3 h() const; real h(int i) const;
const index_extents& extents() const; int3 n_ijk() const; int n(int i) const;
bool on_boundary(int dim, bool right_wall, const int3& coord) const;
```
`umesh_line` is `{real min; real max; real h; int n;}`.
> Several directional-coordinate helpers (`n_dir`, `plane_size`, `ucf_dir`, `uc_dir`, `ucf_ijk2dir`, `uc_ijk2dir`) and `domain()` are present on `cartesian` but are test-only / dead — see [Maturity & known gaps](#maturity--known-gaps).

### `object_geometry` (`object_geometry.hpp`)
```cpp
object_geometry(std::span<const shape>, const cartesian& m);
static std::optional<std::vector<shape>>
       from_lua(const sol::table&, index_extents, const domain_extents&, const logs& = {});
std::span<const mesh_object_info> Rx()/Ry()/Rz() const;     // merged, all shapes
std::span<const mesh_object_info> R(int dir) const;
std::span<const mesh_object_info> Rx(int id)/Ry(int id)/Rz(int id) const; // per-shape (test-only)
std::span<const int3> Sx()/Sy()/Sz() const; std::span<const int3> S(int dir) const; // solid points (test-only)
```

### Shapes (`shapes.hpp`)
- `Shape` concept: any type providing
  `std::optional<hit_info> hit(const ray&, real t_min, real t_max) const` and
  `real3 normal(const real3&) const`.
- `shape` — a type-erased value wrapper (hand-written copy/move/clone) holding any `Shape`.
- `hit_info { real t; real3 position; bool ray_outside; int shape_id; }`.
- Factories: `make_sphere(int id, const real3& origin, real radius)`,
  `make_yz_rect / make_xz_rect / make_xy_rect(int id, const real3& corner0, const real3& corner1, real fluid_normal)`.
  (Only `make_sphere` and `make_yz_rect` are reachable from Lua config.)

### Data structs (`mesh_types.hpp`)
```cpp
struct mesh_object_info { real psi; real3 position; real3 normal;  // outward shape normal
                          bool ray_outside; int3 solid_coord; int shape_id; };
struct object_boundary  { integer object_coordinate; integer objectID; real psi; };
struct boundary         { int3 mesh_coordinate; std::optional<object_boundary> object; };
struct line             { integer stride; boundary start; boundary end; };
struct domain_extents   { real3 min; real3 max; };
```

## How it works

**Config → mesh.** `mesh::from_lua(tbl)` runs `cartesian::from_lua` (reads `simulation.mesh.index_extents` and `simulation.mesh.domain_bounds` → `{index_extents, domain_extents}`), then `object_geometry::from_lua` (reads `simulation.shapes[]` → `vector<shape>`), then constructs `mesh{n, domain, shapes, logger}`. Note: this is called from each *system's* `from_lua` (`heat.cpp:84`, `scalar_wave.cpp:174`, `hyperbolic_eigenvalues.cpp:50`), **not** from `simulation_builder` (which is a stub).

**Grid.** `cartesian`'s constructor pads `n`/`min`/`max` to 3 components, builds `x_`/`y_`/`z_` via `linear_distribute`, sets `h_[i] = (max-min)/(n-1)`, and counts active dims. A dimension with `n==1` is **inactive**: its `h` is `null_v` and operators skip it. This is how 1D/2D problems are expressed — there is no separate 2D vs 3D path.

**Ray-casting (cut cells).** For each direction `I`, `object_geometry::init_line<I>` walks every (slow, fast) grid line, shoots a ray along `I` from the domain min, and repeatedly calls `closest_hit` (which returns the nearest `hit_info` across all shapes, advancing `t_min` via `std::nextafter` to find successive intersections on the same ray). For each hit it:
- snaps `t/h` to the nearest integer cell if within `snap_tol` (`1e-12`), else truncates (`static_cast<int>`);
- sets `solid_coord[I] = i_cell + ray_outside`;
- computes `psi` as the fractional distance from the adjacent **fluid** grid point to the intersection (`off = 1 - 2*ray_outside` selects which neighbor is fluid), then **clamps** to `[snap_tol, 1-snap_tol]`;
- pushes a `mesh_object_info {psi, position, normal, ray_outside, solid_coord, shape_id}` into both the merged `r{x,y,z}_` buffer and the per-shape `r{x,y,z}_m_[id]` buffer.

`init_solid<I>` then walks the merged buffer to enumerate purely-solid grid points (`Sx/Sy/Sz`) using the `ray_outside`/plane-transition logic.

**Lines.** `mesh::init_line<I>` (in `mesh.cpp`, distinct from the one above) converts each grid line into one or more `line`s. A `line` is `[start boundary, end boundary]`, where each `boundary` is either a domain wall (`object == nullopt`) or an `object_boundary` carrying the index into `R(dir)` plus `objectID`/`psi`. Line types: `[domain,domain]`, `[domain,object]`, `[object,domain]`, `[object,object]`. The early-exit `if (extents[I]==1) return;` keeps inactive directions empty.

**Fluid selection.** `init_slices` turns the line list of the **highest active direction** (`i = extents[2]>1 ? 2 : extents[1]>1 ? 1 : 0`, `mesh.cpp:135`) into contiguous `index_slice`s of fluid linear indices, merged where adjacent, then `make_gather_from_slices` builds `fluid_desc_`.

**BC descriptors.** `dirichlet_object_desc` / `non_dirichlet_object_desc` build `gather_selection`s by predicate over `R(dir)`, filtering on `info.shape_id` against the per-object `bcs::Object`. The returned indices are **positions within `R(dir)`** and assume the `R(dir)` buffer order matches the field data buffer order by construction.

## How to extend

**Add a new cut-cell shape** (the most common extension):
1. Define a struct satisfying the `Shape` concept (model on `sphere.cpp` or `rect.hpp`): provide
   `std::optional<hit_info> hit(const ray&, real t_min, real t_max) const` (return `t`/`position`/`ray_outside`/`shape_id`) and `real3 normal(const real3&) const` (outward normal).
2. Declare a `make_<shape>(int id, ...)` factory in `shapes.hpp` and define it in a new `.cpp`; add that `.cpp` to the `add_library(shoccs-mesh ...)` line in `src/mesh/CMakeLists.txt`.
3. **Critically**, wire it into `object_geometry::from_lua` (`object_geometry.cpp` ~line 251): add an `else if (type == "<name>")` branch that parses the Lua params and `push_back`s the shape. Without this the shape is unreachable from config — exactly the current state of `xy_rect`/`xz_rect`.
4. Update the error-message string at `object_geometry.cpp:300` listing valid types.
5. The ray-casting in `init_line<I>` and the line-building in `mesh.cpp` are shape-agnostic and need no changes.

**Add a new grid/geometry query:** add a method to `cartesian` or `mesh`, forward it through `mesh` if consumers need it (consumers see only the `mesh` surface; `cartesian` is a private member), and add a case in `cartesian.t.cpp` / `mesh.t.cpp`.

## Gotchas & invariants
- **Fully-solid lines are not handled.** Both `mesh.cpp` (`init_line` comment) and `object_geometry.cpp` (`init_solid`) only *assert* against a grid line entirely inside a solid; in release this is UB. Do not configure geometry that fully blocks a line.
- **`psi` snap+clamp is intentional (Phase 26.6).** `psi` is snapped to a grid cell, then clamped to `[1e-12, 1-1e-12]`. Because `psi` and `position` come from different arithmetic paths, the stored `psi` can disagree slightly with `position` — by design, to keep `psi` consistent with the snapped cell and avoid degenerate near-zero `psi` that breaks the cut-cell stencil. Do not "simplify" this away.
- **A dimension is inactive when `extents[I]==1`** (`n=1`, `h=null_v`, `init_line<I>` early-exits, fluid selector uses the highest active direction). This is the only 1D/2D mechanism.
- **`R(dir)` buffer order is load-bearing.** `dirichlet_object_desc`/`non_dirichlet_object_desc` produce gather indices that assume `R(dir)` layout matches the data buffer "by construction" (`mesh.hpp:96`). Reordering intersection points silently corrupts BC application.
- **`shape` uses raw-pointer type erasure** (`new`/`delete` + `clone`), with hand-written copy/move/assign. Correct, but pre-modern — do not assume `unique_ptr`/RAII smart-pointer semantics when editing.
- **Two extent types.** `domain_extents` lives in `mesh_types.hpp`; `index_extents` lives in the indexing subsystem. `cartesian::from_lua` returns a pair of both and `mesh::from_lua` threads them — easy to confuse.
- **`mesh::from_lua` is the real entry point**, called inside each system's own `from_lua`, not via a central `simulation_builder` (which is a dead stub). The mesh is built transitively per system, not by the simulation layer.

## Maturity & known gaps
**Verdict: mature.** The core path is heavily exercised in production: `mesh::from_lua` is called by every PDE system, and `lines(dir)`, `interp_line`, `dirichlet_line`, `R(dir)`, `fluid_desc`, `dirichlet_object_desc`/`non_dirichlet_object_desc` are consumed by `operators/derivative.cpp` (lines 242, 328, 329) and `systems/{heat,scalar_wave,hyperbolic_eigenvalues}.cpp`. Four test files cover 1D/2D/3D grids and sphere/yz_rect intersections. `object_geometry.cpp` was actively maintained as recently as 2026-03-31; `shapes.hpp`/`sphere.cpp`/`rect.cpp` are old but the live implementation.
> Build caveat: `t-cartesian`, `t-object_geometry`, `t-shapes` are reported PASS; `t-mesh` shows a link FAIL that is the environment-wide Kokkos 5.0→5.1 runtime-linker mismatch (it links `Kokkos::kokkos`), **not** a mesh-logic failure. These verdicts were from a stale build tree and not re-verified against Kokkos 5.1.1.

Partial / dead items in this subsystem (verified flags):
- **`make_xy_rect` / `make_xz_rect` (and `rect<2>`/`rect<1>`) — partial.** `shapes.hpp:124-125`, `rect.cpp`. The `rect<I>` mechanism is mature (the `I=0` / `yz_rect` instantiation is production-critical), but `xy_rect` is test-only and `xz_rect` has *zero* callers; neither is wired into `object_geometry::from_lua`, so neither is reachable from any Lua config. Recommendation: **finish** (add `xy_rect`/`xz_rect` dispatch branches) — see [Cleanup Plan](../CLEANUP_PLAN.md).
- **Solid-point API `Sx()/Sy()/Sz()/S(dir)` + `sx_/sy_/sz_` + `init_solid` — partial/experimental.** `object_geometry.hpp:70-84`, `object_geometry.cpp:140-200,209-211`. Fully implemented and unit-tested, but consumed by *zero* production paths and not even forwarded through `mesh.hpp`. It is the documented (unimplemented) foundation for the CSR solid-point / `R^x→S^x` boundary-coupling design in `docs/discrete_operators.md`. Recommendation: **document-as-experimental** — see [Cleanup Plan](../CLEANUP_PLAN.md).
- **Per-shape `Rx(int)/Ry(int)/Rz(int)` + `rx_m_/ry_m_/rz_m_` — partial.** `object_geometry.hpp:39-49`, `object_geometry.cpp:216-233`. Backing vectors are populated every construction (real cost), but the read API has a single test-only caller (`g.Rz(0)`); production gets per-shape info for free from `mesh_object_info::shape_id` on the merged arrays. Recommendation: **deprecate** — see [Cleanup Plan](../CLEANUP_PLAN.md).
- **`object_geometry::domain()` and `cartesian::domain()` — dead.** `object_geometry.hpp:63-67`, `cartesian.hpp:72`. Zero callers anywhere; their only consumers (`mesh::xyz`/`vxyz`) were removed in Phase-12 cleanup (commit `86cdb0a`) but these orphans were left behind. Safe to delete — see [Cleanup Plan](../CLEANUP_PLAN.md).
- **`cartesian` directional-coordinate helpers (`n_dir`, `plane_size`, `ucf_dir`, `uc_dir`, `ucf_ijk2dir`, `uc_ijk2dir`) — dead.** `cartesian.hpp:54-122`. Pre-Kokkos slow/fast indexing utilities; only `ucf_ijk2dir`/`ucf_dir` are referenced, and only by `cartesian.t.cpp`. Private to `mesh`, not forwarded, no production path. Safe to delete (with the test references) — see [Cleanup Plan](../CLEANUP_PLAN.md).
- **Commented-out `Sxyz()` accessor — dead.** `object_geometry.hpp:86`. References `vector_range`, a range-v3-era type that no longer exists. Safe to delete — see [Cleanup Plan](../CLEANUP_PLAN.md).

## Tests
All carry the **`mesh`** CTest label.
- `t-cartesian` (`cartesian.t.cpp`) — `TEST_CASE("mesh api")` with `3d`/`2d`/`1d` sections: `line()`, `x/y/z`, `ucf_ijk2dir`, `ucf_dir`. (`add_unit_test`, no Kokkos.)
- `t-shapes` (`shapes.t.cpp`) — `sphere`, `xy_rect` (IN/OUT), `yz_rect` (IN/OUT). This is the **only** place `make_xy_rect` is exercised.
- `t-object_geometry` (`object_geometry.t.cpp`) — `sphere intersections` (X/y/z), `rect_intersections`, `1D rect_intersections`, `grid-aligned sphere - cross-direction consistency`; also checks `Sx/Sy/Sz` solid points and the one `g.Rz(0)` per-shape call.
- `t-mesh` (`mesh.t.cpp`) — `lines with no cut-cells`, `lines` (X/Y/Z), `selections`, `selections with object`, `fluid_desc`, `dirichlet_object_desc and non_dirichlet_object_desc`. Linked manually (needs Kokkos + `shoccs-random`).

**Not covered:** `make_xz_rect` (no test, no Lua); `make_xy_rect` (test-only, not Lua-reachable). The per-shape accessors and the solid-point API are touched only by `object_geometry.t.cpp`. No disabled or commented-out tests within the mesh test files.

## Related docs
- [operators](operators.md) — primary consumer (`derivative` uses `lines`, `interp_line`, `dirichlet_line`, `R(dir)`).
- [systems](systems.md) — `heat`/`scalar_wave`/`hyperbolic_eigenvalues` own a `mesh` and use `fluid_desc`/`*_object_desc`/`R(dir)`.
- [core-types](core-types.md) — `index_extents`, indexing, `real`/`real3`/`int3`, selection descriptors.
- [stencils](stencils.md) — cut-cell stencils consume `psi`.
- Legacy design docs (still useful for *rationale*, but pre-Kokkos): `docs/geometry.md` (ray-tracing intersection design) and `docs/discrete_operators.md` (the `R^x → S^x` / `T`-mapping solid-point design that the currently-unused `Sx/Sy/Sz` API was meant to back).
