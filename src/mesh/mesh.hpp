#pragma once

#include "cartesian.hpp"
#include "fields/selector.hpp"
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

    template <bcs::type B, int I>
    auto grid_boundaries(const bcs::Grid& g) const
    {
        if constexpr (I == -1) {
            return tuple{xmin, xmax, ymin, ymax, zmin, zmax} |
                   tuple{sel::optional_view(g[0].left == B),
                         sel::optional_view(g[0].right == B),
                         sel::optional_view(g[1].left == B),
                         sel::optional_view(g[1].right == B),
                         sel::optional_view(g[2].left == B),
                         sel::optional_view(g[2].right == B)};
        } else {
            return tuple{xmin, xmax, ymin, ymax, zmin, zmax} |
                   tuple{sel::optional_view(I == 0 && g[0].left == B),
                         sel::optional_view(I == 0 && g[0].right == B),
                         sel::optional_view(I == 1 && g[1].left == B),
                         sel::optional_view(I == 1 && g[1].right == B),
                         sel::optional_view(I == 2 && g[2].left == B),
                         sel::optional_view(I == 2 && g[2].right == B)};
        }
    }

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

    auto ss() const // scalar size
    {
        return tuple{tuple{size()}, tuple{Rx().size(), Ry().size(), Rz().size()}};
    }

    auto vs() const // vector size
    {
        return tuple{ss(), ss(), ss()};
    }

    template <int I = -1>
    auto dirichlet(const bcs::Grid& g) const
    {
        return grid_boundaries<bcs::Dirichlet, I>(g);
    }

    template <int I = -1>
    auto neumann(const bcs::Grid& g) const
    {
        return grid_boundaries<bcs::Neumann, I>(g);
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
        xyz;
    tuple<tuple<decltype(std::declval<cartesian>().domain()),
                decltype(std::declval<object_geometry>().domain())>,
          tuple<decltype(std::declval<cartesian>().domain()),
                decltype(std::declval<object_geometry>().domain())>,
          tuple<decltype(std::declval<cartesian>().domain()),
                decltype(std::declval<object_geometry>().domain())>>
        vxyz;
};
} // namespace ccs
