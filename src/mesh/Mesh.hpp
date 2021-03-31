#pragma once

#include "Cartesian.hpp"
#include "CutGeometry.hpp"
#include "boundaries.hpp"
#include "fields/Location.hpp"
#include "fields/Tuple.hpp"

#include "Selections.hpp"

namespace ccs::mesh
{

struct ObjectBoundary {
    integer object_coordinate; // the coordinate in R associated with this point
    integer objectID;
    real psi;
};

struct Boundary {
    int3 mesh_coordinate; // mesh coordinate associated with this point
    std::optional<ObjectBoundary> object_boundary;
};

struct Line {
    integer offset; // offset of mesh_coordinate in F
    integer stride; // stride associated with moving in this diriection
    Boundary start;
    Boundary end;
};

struct DomainBounds {
    real3 min;
    real3 max;
};

struct IndexExtents {
    int3 extents;
};

class Mesh
{
    Cartesian cartesian;
    CutGeometry geometry;
    field::tuple::Location location;
    std::array<std::vector<Line>, 3> lines_;

public:
    Mesh() = default;
    Mesh(const DomainBounds& bounds, const IndexExtents& extents);

    Mesh(const DomainBounds& bounds,
         const IndexExtents& extents,
         const std::vector<shape>& shapes);

    bool dirichlet_line(const int3& start, int dir, const bcs::Grid& cartesian_bcs) const;

    constexpr auto size() const { return cartesian.size(); }

    constexpr int dims() const { return cartesian.dims(); }

    operator field::tuple::Location() const { return location; }

    const auto& lines(int i) const { return lines_[i]; }

    constexpr real h(int i) const { return cartesian.h(i); }

    constexpr real3 h() const { return cartesian.h(); }

    constexpr int3 extents() const { return cartesian.extents(); }

    constexpr auto xmin() const { return detail::xmin(extents()); }

    constexpr auto xmax() const { return detail::xmax(extents()); }

    constexpr auto ymin() const { return detail::ymin(extents()); }

    constexpr auto ymax() const { return detail::ymax(extents()); }

    constexpr auto zmin() const { return detail::zmin(extents()); }

    constexpr auto zmax() const { return detail::zmax(extents()); }
};
} // namespace ccs::mesh