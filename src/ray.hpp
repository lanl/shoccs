#pragma once
#include "types.hpp"

namespace ccs
{

struct ray {
    real3 origin;
    real3 direction;

    constexpr real3 position(real t) const
    {
        return {origin[0] + t * direction[0],
                origin[1] + t * direction[1],
                origin[2] + t * direction[2]};
    }
};

} // namespace shoccs
