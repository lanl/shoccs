#pragma once
#include "types.hpp"
#include <optional>

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

// only difference between this and hit_info is the solid_coord.
struct mesh_object_info {
    real psi; // 1D cutcell distance
    real3 position;
    real3 normal; // outward shape normal
    bool ray_outside;
    int3 solid_coord;
    int shape_id;
};

} // namespace ccs
