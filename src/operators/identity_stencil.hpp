#pragma once

#include "stencils/stencils.hpp"
#include <cassert>
// a simple identity stencil for testing operators

namespace ccs
{
struct identity_stencil {
    stencil_info query(boundary b) const
    {
        if (b == boundary::neumann)
            return {0, 2, 3, 2};
        else
            return {0, 2, 3, 0};
    }
    stencil_info query_max() const { return {0, 2, 3, 2}; }

    void interior(real, std::span<real> c) const { c[0] = 1; }

    void nbs(real,
             boundary b,
             real psi,
             bool right_wall,
             std::span<real> c,
             std::span<real> x) const
    {
        assert(0 <= psi && psi <= 1);
        if (b == boundary::neumann) {
            if (right_wall) {
                x[0] = 1;
                x[1] = 2;

                c[0] = 0;
                c[1] = 1;
                c[2] = -1;
                c[3] = 0;
                c[4] = 0;
                c[5] = -1;
            } else {
                x[0] = 2;
                x[1] = 1;

                c[0] = -1;
                c[1] = 0;
                c[2] = 0;
                c[3] = -1;
                c[4] = 1;
                c[5] = 0;
            }
        } else if (right_wall) {
            c[0] = 0;
            c[1] = 1;
            c[2] = 0;
            c[3] = 0;
            c[4] = 0;
            c[5] = 1;
        } else {
            c[0] = 1;
            c[1] = 0;
            c[2] = 0;
            c[3] = 0;
            c[4] = 1;
            c[5] = 0;
        }
    }
};
} // namespace ccs