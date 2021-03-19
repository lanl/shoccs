#pragma once

#include "types.hpp"

#include "boundaries.hpp"
#include "fields/Scalar.hpp"
#include "mesh/Mesh.hpp"
#include "stencils/Stencils.hpp"
#include "matrices/Block.hpp"
#include "matrices/CSR.hpp"

namespace ccs::operators
{
class Derivative
{
    matrix::Block O;
    matrix::CSR B;
    std::vector<real> interior_c;

public:
    Derivative() = default;

    Derivative(int dir,
               int order,
               const stencils::Stencil& stencil,
               const bcs::Grid& grid_bcs,
               const bcs::Object& object_bcs);

    void operator()(field::ScalarView_Const, field::ScalarView_Mutable) {}
};
} // namespace ccs::operators