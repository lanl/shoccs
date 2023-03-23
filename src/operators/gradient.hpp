#pragma once

#include "derivative.hpp"
#include "fields/vector.hpp"
#include "operator_visitor.hpp"

namespace ccs
{

class gradient
{
    derivative dx;
    derivative dy;
    derivative dz;
    index_extents ex;

public:
    gradient() = default;

    gradient(const mesh&,
             const stencil&,
             const bcs::Grid&,
             const bcs::Object&,
             const logs& = {});

    std::function<void(vector_span)> operator()(scalar_view) const;

    void visit(operator_visitor& v) const { return v.visit(dx); }
};
} // namespace ccs
