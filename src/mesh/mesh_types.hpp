#pragma once
#include "types.hpp"

namespace ccs
{

struct object_boundary {
    integer object_coordinate; // the coordinate in R associated with this point
    integer objectID;
    real psi;
};

struct boundary {
    int3 mesh_coordinate; // mesh coordinate associated with this point
    std::optional<object_boundary> object;
};

struct line {
    integer stride; // stride associated with moving in this diriection
    boundary start;
    boundary end;
};

struct domain_extents {
    real3 min;
    real3 max;
};

} // namespace ccs