#pragma once

#include "index_extents.hpp"
#include "kokkos_types.hpp"
#include "mesh/mesh_types.hpp"

#include <cassert>
#include <limits>
#include <span>

namespace ccs
{

// ---------------------------------------------------------------------------
// Selection descriptors: lightweight structs describing which elements to
// access in a flat buffer. Used by assign_selected / fill_selected /
// plus_assign_selected to replace iterator-based selector views.
// ---------------------------------------------------------------------------

// Contiguous range: elements [offset, offset + count).
// Used for x-plane selections.
struct contiguous_selection {
    int offset_;
    int count_;

    KOKKOS_INLINE_FUNCTION int element(int i) const { return offset_ + i; }
    KOKKOS_INLINE_FUNCTION int count() const { return count_; }
};

// Strided pattern: outer_count blocks of inner_count contiguous elements,
// separated by outer_stride.
// Used for y-plane (inner_count = nz) and z-plane (inner_count = 1).
// Invariant: inner_count_ must be > 0 (used as divisor in element()).
struct strided_selection {
    int offset_;
    int inner_count_;
    int outer_count_;
    int outer_stride_;

    KOKKOS_INLINE_FUNCTION int element(int i) const
    {
        return offset_ + (i / inner_count_) * outer_stride_ + (i % inner_count_);
    }
    KOKKOS_INLINE_FUNCTION int count() const { return inner_count_ * outer_count_; }
};

// Gather pattern: arbitrary index list stored in a Kokkos::View.
// Used for fluid (multi_slice) and predicate (object BC) selections.
struct gather_selection {
    Kokkos::View<const int*, memory_space> indices_;

    KOKKOS_INLINE_FUNCTION int element(int i) const { return indices_(i); }
    KOKKOS_INLINE_FUNCTION int count() const
    {
        return static_cast<int>(indices_.extent(0));
    }
};

// ---------------------------------------------------------------------------
// Trivially-copyable assertions for contiguous and strided descriptors.
// gather_selection holds a Kokkos::View which may not be trivially copyable,
// but is capturable in KOKKOS_LAMBDA.
// ---------------------------------------------------------------------------

static_assert(std::is_trivially_copyable_v<contiguous_selection>);
static_assert(std::is_trivially_copyable_v<strided_selection>);

// ---------------------------------------------------------------------------
// Plane descriptor factory functions.
//
// For a mesh with extents {nx, ny, nz}:
//   x-plane at i: contiguous_selection{i * ny * nz, ny * nz}
//   y-plane at j: strided_selection{j * nz, nz, nx, ny * nz}
//   z-plane at k: strided_selection{k, 1, nx * ny, nz}
// ---------------------------------------------------------------------------

inline contiguous_selection make_x_plane_desc(index_extents ext, int i)
{
    int ny = ext[1];
    int nz = ext[2];
    assert(static_cast<long>(ny) * nz <= std::numeric_limits<int>::max());
    return {i * ny * nz, ny * nz};
}

inline strided_selection make_y_plane_desc(index_extents ext, int j)
{
    int nx = ext[0];
    int ny = ext[1];
    int nz = ext[2];
    assert(nz > 0);
    assert(static_cast<long>(ny) * nz <= std::numeric_limits<int>::max());
    return {j * nz, nz, nx, ny * nz};
}

inline strided_selection make_z_plane_desc(index_extents ext, int k)
{
    int nx = ext[0];
    int ny = ext[1];
    int nz = ext[2];
    assert(static_cast<long>(nx) * ny > 0);
    assert(static_cast<long>(nx) * ny <= std::numeric_limits<int>::max());
    return {k, 1, nx * ny, nz};
}

// ---------------------------------------------------------------------------
// Factory: build gather_selection from index_slice arrays.
// Flattens all [first, last) ranges into a single index array.
// ---------------------------------------------------------------------------

inline gather_selection make_gather_from_slices(std::span<const index_slice> slices)
{
    // Count total elements across all slices.
    int total = 0;
    for (auto& s : slices)
        total += static_cast<int>(s.last - s.first);

    Kokkos::View<int*, memory_space> indices("gather_indices", total);
    auto h = Kokkos::create_mirror_view(indices);

    int pos = 0;
    for (auto& s : slices)
        for (integer idx = s.first; idx < s.last; ++idx) {
            assert(idx <= std::numeric_limits<int>::max());
            h(pos++) = static_cast<int>(idx);
        }

    Kokkos::deep_copy(indices, h);
    return gather_selection{indices};
}

// ---------------------------------------------------------------------------
// Factory: build gather_selection from a predicate over mesh_object_info array.
// Collects indices i where pred(infos[i]) is true.
// ---------------------------------------------------------------------------

// The returned indices are positions within the `infos` span. Callers must
// ensure the data buffer at the same position holds the value associated
// with infos[i].
template <typename Pred>
gather_selection make_gather_from_predicate(std::span<const mesh_object_info> infos, Pred pred)
{
    // First pass: count matching elements.
    int total = 0;
    for (int i = 0; i < static_cast<int>(infos.size()); ++i)
        if (pred(infos[i]))
            ++total;

    Kokkos::View<int*, memory_space> indices("gather_pred_indices", total);
    auto h = Kokkos::create_mirror_view(indices);

    int pos = 0;
    for (int i = 0; i < static_cast<int>(infos.size()); ++i)
        if (pred(infos[i]))
            h(pos++) = i;

    Kokkos::deep_copy(indices, h);
    return gather_selection{indices};
}

// ---------------------------------------------------------------------------
// Selected operations: assign, fill, and plus-assign over descriptor-selected
// elements. These dispatch via Kokkos::parallel_for and replace iterator-based
// selector view operations on hot paths.
//
// Same synchronous execution_space requirement as assign() in expr.hpp —
// dst and Expr capture raw real* pointers valid only for the call duration.
// ---------------------------------------------------------------------------

template <typename Desc, typename Expr>
void assign_selected(real* dst, Desc desc, Expr expr)
{
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, desc.count()),
        KOKKOS_LAMBDA(int i) {
            int idx = desc.element(i);
            dst[idx] = expr(idx);
        });
}

template <typename Desc>
void fill_selected(real* dst, Desc desc, real value)
{
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, desc.count()),
        KOKKOS_LAMBDA(int i) { dst[desc.element(i)] = value; });
}

template <typename Desc, typename Expr>
void plus_assign_selected(real* dst, Desc desc, Expr expr)
{
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, desc.count()),
        KOKKOS_LAMBDA(int i) {
            int idx = desc.element(i);
            dst[idx] += expr(idx);
        });
}

// ---------------------------------------------------------------------------
// Grid BC descriptor helper: iterates over 6 mesh faces, calling fn(desc)
// for each face whose BC type matches B.
// fn must be a generic lambda since descriptors may be contiguous or strided.
// Caller must include "operators/boundaries.hpp" for bcs::Grid and bcs::type.
// ---------------------------------------------------------------------------

// Forward declare bcs::type so it can be used as a non-type template parameter
// without pulling in the heavy boundaries.hpp header (which includes spdlog/fmt).
namespace bcs { enum class type; }

template <bcs::type B, typename GridT, typename Fn>
void for_each_grid_bc_desc(const GridT& g, index_extents ext, Fn fn)
{
    // x faces
    if (g[0].left == B) fn(make_x_plane_desc(ext, 0));
    if (g[0].right == B) fn(make_x_plane_desc(ext, ext[0] - 1));
    // y faces
    if (g[1].left == B) fn(make_y_plane_desc(ext, 0));
    if (g[1].right == B) fn(make_y_plane_desc(ext, ext[1] - 1));
    // z faces
    if (g[2].left == B) fn(make_z_plane_desc(ext, 0));
    if (g[2].right == B) fn(make_z_plane_desc(ext, ext[2] - 1));
}

} // namespace ccs
