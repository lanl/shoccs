#pragma once

#include "mesh/Cartesian.hpp"
#include "mesh/CutGeometry.hpp"
#include "stencils/stencils.hpp"
#include "types.hpp"

namespace ccs::operators
{

class DiscreteOperator
{
    stencils::Stencil stencil;

public:
    DiscreteOperator() = default;

    DiscreteOperator(const mesh::Cartesian&, const mesh::CutGeometry&) {}

    template <typename T>
    T to()
    {
        return {};
    }
};
} // namespace ccs::operators