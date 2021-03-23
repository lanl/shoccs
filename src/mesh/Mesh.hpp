#pragma once

#include "Cartesian.hpp"
#include "CutGeometry.hpp"
#include "fields/Location.hpp"

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

    constexpr long size() const { return cartesian.size(); }

    operator field::tuple::Location() const { return location; }

    const auto& lines(int i) const { return lines_[i]; }
};
} // namespace ccs::mesh