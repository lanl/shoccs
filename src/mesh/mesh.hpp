#pragma once

#include "cartesian.hpp"
#include "fields/selection_desc.hpp"
#include "io/logging.hpp"
#include "mesh_types.hpp"
#include "object_geometry.hpp"
#include "operators/boundaries.hpp"

#include <sol/forward.hpp>

namespace ccs
{

class mesh
{
    cartesian cart;
    object_geometry geometry;
    std::array<std::vector<line>, 3> lines_;
    std::vector<index_slice> fluid_slices;
    gather_selection fluid_desc_;
    logs logger;

public:
    mesh() = default;
    mesh(const index_extents& extents, const domain_extents& bounds, const logs& = {});

    mesh(const index_extents& extents,
         const domain_extents& bounds,
         const std::vector<shape>& shapes,
         const logs& = {});

    bool dirichlet_line(const int3& start, int dir, const bcs::Grid& cartesian_bcs) const;

    constexpr auto size() const { return cart.size(); }

    constexpr int dims() const { return cart.dims(); }

    constexpr std::span<const real> x() const { return cart.x(); }
    constexpr std::span<const real> y() const { return cart.y(); }
    constexpr std::span<const real> z() const { return cart.z(); }

    const auto& lines(int i) const { return lines_[i]; }

    constexpr real h(int i) const { return cart.h(i); }

    constexpr real3 h() const { return cart.h(); }

    constexpr decltype(auto) extents() const { return cart.extents(); }

    constexpr auto stride(int dir) const
    {
        switch (dir) {
        case 0:
            return index::stride<0>(extents());
        case 1:
            return index::stride<1>(extents());
        default:
            return index::stride<2>(extents());
        }
    }

    // convert an int3 coordinate to a flattened integer coordinate
    constexpr integer ic(int3 ijk) const
    {
        const auto& n = extents();
        return ijk[0] * n[1] * n[2] + ijk[1] * n[2] + ijk[2];
    }

    // Intersection of rays in x and all objects
    std::span<const mesh_object_info> Rx() const { return geometry.Rx(); }
    // Intersection of rays in y and all objects
    std::span<const mesh_object_info> Ry() const { return geometry.Ry(); }
    // Intersection of rays in z and all objects
    std::span<const mesh_object_info> Rz() const { return geometry.Rz(); }

    auto R() const
    {
        return std::array<std::span<const mesh_object_info>, 3>{Rx(), Ry(), Rz()};
    }

    std::span<const mesh_object_info> R(int dir) const
    {
        switch (dir) {
        case 0:
            return Rx();
        case 1:
            return Ry();
        default:
            return Rz();
        }
    }

    const gather_selection& fluid_desc() const { return fluid_desc_; }

    // Indices into R(dir); R(dir) buffer layout matches the data buffer by construction.
    gather_selection dirichlet_object_desc(int dir, const bcs::Object& o) const
    {
        return make_gather_from_predicate(
            R(dir),
            [&o](const mesh_object_info& info) {
                return o[info.shape_id] == bcs::Dirichlet;
            });
    }

    // Indices into R(dir); R(dir) buffer layout matches the data buffer by construction.
    gather_selection non_dirichlet_object_desc(int dir, const bcs::Object& o) const
    {
        return make_gather_from_predicate(
            R(dir),
            [&o](const mesh_object_info& info) {
                return o[info.shape_id] != bcs::Dirichlet;
            });
    }

    line interp_line(int dir, int3 pt) const;

    static std::optional<mesh> from_lua(const sol::table&, const logs& = {});
};
} // namespace ccs
