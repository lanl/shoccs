#pragma once
#include "types.hpp"

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
    integer stride; // stride associated with moving in this diriection
    Boundary start;
    Boundary end;
};

struct DomainBounds {
    real3 min;
    real3 max;
};

} // namespace ccs::mesh