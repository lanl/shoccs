#pragma once

#include "boundaries.hpp"
#include "mesh/Mesh.hpp"
#include "stencils/stencils.hpp"
#include "types.hpp"

namespace ccs::operators
{

class DiscreteOperator
{
    stencils::Stencil stencil;

public:
    DiscreteOperator() = default;

    DiscreteOperator(stencils::Stencil stencil, const mesh::Mesh&)
        : stencil{MOVE(stencil)}
    {
    }

    template <typename T>
    T to(const bcs::Grid&)
    {
        return {};
    }
};
} // namespace ccs::operators