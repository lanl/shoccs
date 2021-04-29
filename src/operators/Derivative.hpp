#pragma once

#include "types.hpp"

#include "boundaries.hpp"
#include "fields/Scalar.hpp"
#include "matrices/Block.hpp"
#include "matrices/CSR.hpp"
#include "mesh/Mesh.hpp"
#include "stencils/Stencils.hpp"

namespace ccs::operators
{
class Derivative
{
    int dir;
    matrix::Block O;
    matrix::CSR B;
    matrix::CSR N;
    std::vector<real> interior_c;

public:
    Derivative() = default;

    Derivative(int dir,
               const mesh::Mesh& m,
               const stencils::Stencil& stencil,
               const bcs::Grid& grid_bcs,
               const bcs::Object& object_bcs);

    // operator for when neumann conditions are not needed
    template <typename Op = eq_t>
    requires(!field::tuple::traits::ScalarType<Op>) void
    operator()(field::ScalarView_Const, field::ScalarView_Mutable, Op op = {}) const;

    // operaotr for when neumann conditions may be applied
    template <typename Op = eq_t>
    requires(!field::tuple::traits::ScalarType<Op>) void
    operator()(field::ScalarView_Const field_values,
               field::ScalarView_Const derivative_values,
               field::ScalarView_Mutable,
               Op op = {}) const;
};
} // namespace ccs::operators