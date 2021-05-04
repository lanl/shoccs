#pragma once

#include "stencils/stencil.hpp"
#include <cassert>
// a simple identity stencil for testing operators

namespace ccs::stencils
{
namespace detail
{
struct identity {
    info query(bcs::type b) const
    {
        switch (b) {
        case (bcs::Neumann):
            return {0, 2, 3, 2};
        case (bcs::Floating):
            return {0, 2, 3, 0};
        default:
            return {0, 1, 4, 0};
        }
    }
    info query_max() const { return {0, 2, 4, 2}; }

    void interior(real, std::span<real> c) const { c[0] = 1; }

    void nbs(real,
             bcs::type b,
             real psi,
             bool right_wall,
             std::span<real> c,
             std::span<real> x) const
    {
        assert(0 <= psi && psi <= 1);
        switch (b) {
        case (bcs::Neumann):
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
            break;
        case (bcs::Floating):
            if (right_wall) {
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
            break;
        default:
            if (right_wall) {
                c[0] = 0;
                c[1] = 0;
                c[2] = 1;
                c[3] = 0;
            } else {
                c[0] = 0;
                c[1] = 1;
                c[2] = 0;
                c[3] = 0;
            }
        }
    }
};
} // namespace detail
constexpr auto identity = detail::identity{};
} // namespace ccs::stencils