#pragma once

#include "Cartesian.hpp"
#include "CutGeometry.hpp"
#include "fields/Location.hpp"

namespace ccs::mesh
{

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

public:
    Mesh() = default;
    Mesh(const DomainBounds& bounds, const IndexExtents& extents)
        : cartesian{bounds.min, bounds.max, extents.extents}, geometry{}
    {
    }

    constexpr long size() const { return cartesian.size(); }

    operator field::tuple::Location() const { return location; }
};
} // namespace ccs::mesh