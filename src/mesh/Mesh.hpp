#pragma once

#include "Cartesian.hpp"
#include "CutGeometry.hpp"
#include "boundaries.hpp"
#include "fields/Location.hpp"
#include "fields/Tuple.hpp"
#include "mesh_types.hpp"

#include "Selections.hpp"

namespace ccs::mesh
{

class Mesh
{
    Cartesian cartesian;
    CutGeometry geometry;
    std::array<std::vector<Line>, 3> lines_;

public:
    Mesh() = default;
    Mesh(const IndexExtents& extents, const DomainBounds& bounds);

    Mesh(const IndexExtents& extents,
         const DomainBounds& bounds,
         const std::vector<shape>& shapes);

    bool dirichlet_line(const int3& start, int dir, const bcs::Grid& cartesian_bcs) const;

    constexpr auto size() const { return cartesian.size(); }

    constexpr int dims() const { return cartesian.dims(); }

    const auto& lines(int i) const { return lines_[i]; }

    constexpr real h(int i) const { return cartesian.h(i); }

    constexpr real3 h() const { return cartesian.h(); }

    constexpr decltype(auto) extents() const { return cartesian.extents(); }

    // convert an int3 coordinate to a flattened integer coordinate
    constexpr integer ic(int3 ijk) const
    {
        const auto& n = extents();
        return ijk[0] * n[1] * n[2] + ijk[1] * n[2] + ijk[2];
    }

    constexpr auto location() const { return views::location(cartesian, geometry); }

    constexpr auto xmin() const { return views::xmin(extents()); }

    constexpr auto xmax() const { return views::xmax(extents()); }

    constexpr auto ymin() const { return views::ymin(extents()); }

    constexpr auto ymax() const { return views::ymax(extents()); }

    constexpr auto zmin() const { return views::zmin(extents()); }

    constexpr auto zmax() const { return views::zmax(extents()); }

    auto F() const
    {
        auto ex = extents();
        auto [nx, ny, nz] = ex;
        if (nz > 1)
            return views::F(ex, lines(2));
        else if (ny > 1)
            return views::F(ex, lines(1));
        else
            return views::F(ex, lines(0));
    }

    // Intersection of rays in x and all objects
    std::span<const mesh_object_info> Rx() const { return geometry.Rx(); }
    // Intersection of rays in y and all objects
    std::span<const mesh_object_info> Ry() const { return geometry.Ry(); }
    // Intersection of rays in z and all objects
    std::span<const mesh_object_info> Rz() const { return geometry.Rz(); }

    auto R() const { return field::Tuple{geometry.Rx(), geometry.Ry(), geometry.Rz()}; }
};
} // namespace ccs::mesh