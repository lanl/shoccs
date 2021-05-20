#pragma once

#include "operators/boundaries.hpp"
#include "cartesian.hpp"
#include "fields/selector.hpp"
#include "mesh_types.hpp"
#include "object_geometry.hpp"

#include <sol/forward.hpp>

namespace ccs
{

class mesh
{
    cartesian cart;
    object_geometry geometry;
    std::array<std::vector<line>, 3> lines_;
    std::vector<index_slice> fluid_slices;

public:
    mesh() = default;
    mesh(const index_extents& extents, const domain_extents& bounds);

    mesh(const index_extents& extents,
         const domain_extents& bounds,
         const std::vector<shape>& shapes);

    bool dirichlet_line(const int3& start, int dir, const bcs::Grid& cartesian_bcs) const;

    constexpr auto size() const { return cart.size(); }

    constexpr int dims() const { return cart.dims(); }

    const auto& lines(int i) const { return lines_[i]; }

    constexpr real h(int i) const { return cart.h(i); }

    constexpr real3 h() const { return cart.h(); }

    constexpr decltype(auto) extents() const { return cart.extents(); }

    // convert an int3 coordinate to a flattened integer coordinate
    constexpr integer ic(int3 ijk) const
    {
        const auto& n = extents();
        return ijk[0] * n[1] * n[2] + ijk[1] * n[2] + ijk[2];
    }

    // constexpr auto location() const { return views::location(cart, geometry); }

    // Intersection of rays in x and all objects
    std::span<const mesh_object_info> Rx() const { return geometry.Rx(); }
    // Intersection of rays in y and all objects
    std::span<const mesh_object_info> Ry() const { return geometry.Ry(); }
    // Intersection of rays in z and all objects
    std::span<const mesh_object_info> Rz() const { return geometry.Rz(); }

    auto R() const { return tuple{geometry.Rx(), geometry.Ry(), geometry.Rz()}; }

    auto ss() const // scalar size
    {
        return tuple{tuple{size()}, tuple{Rx().size(), Ry().size(), Rz().size()}};
    }

    static std::optional<mesh> from_lua(const sol::table&);

    sel::xmin_t xmin;
    sel::xmax_t xmax;
    sel::ymin_t ymin;
    sel::ymax_t ymax;
    sel::zmin_t zmin;
    sel::zmax_t zmax;
    sel::multi_slice_t fluid;
    tuple<decltype(std::declval<cartesian>().domain()),
          decltype(std::declval<object_geometry>().domain())>
        location;
};
} // namespace ccs
