#pragma once

#include "Derivative.hpp"
#include "fields/SystemField.hpp"

namespace ccs::operators
{
class Laplacian
{
    Derivative dx;
    Derivative dy;
    Derivative dz;

public:
    Laplacian() = default;

    Laplacian(const mesh::Mesh&,
              const stencils::Stencil&,
              const bcs::Grid&,
              const bcs::Object&);

    // when there are no neumann conditions in the problem
    std::function<void(field::ScalarView_Mutable)>
    operator()(field::ScalarView_Const) const;

    std::function<void(field::ScalarView_Mutable)>
    operator()(field::ScalarView_Const field_values,
               field::ScalarView_Const derivative_values) const;
};
} // namespace ccs::operators