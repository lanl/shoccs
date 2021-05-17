#pragma once

#include "derivative.hpp"

namespace ccs
{
class laplacian
{
    derivative dx;
    derivative dy;
    derivative dz;

public:
    laplacian() = default;

    laplacian(const mesh&, const stencil&, const bcs::Grid&, const bcs::Object&);

    // when there are no neumann conditions in the problem
    std::function<void(scalar_span)> operator()(scalar_view) const;

    std::function<void(scalar_span)> operator()(scalar_view field_values,
                                                scalar_view derivative_values) const;
};
} // namespace ccs