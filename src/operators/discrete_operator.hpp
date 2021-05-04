#pragma once

#include "boundaries.hpp"
#include "mesh/mesh.hpp"
#include "stencils/stencil.hpp"
#include "types.hpp"

namespace ccs
{

class discrete_operator
{
    stencil st;

public:
    discrete_operator() = default;

    discrete_operator(stencil st, const mesh&) : st{MOVE(st)} {}

    template <typename T>
    T to(const bcs::Grid&)
    {
        // constexpr auto order = T::order;

        return {};
    }
};
} // namespace ccs