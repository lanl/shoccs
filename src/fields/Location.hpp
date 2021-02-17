#pragma once

#include "types.hpp"

namespace ccs::mesh
{

struct Location {
    std::span<const real> x;
    std::span<const real> y;
    std::span<const real> z;
    std::span<const real3> rx;
    std::span<const real3> ry;
    std::span<const real3> rz;
};
} // namespace ccs::mesh