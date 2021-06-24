#pragma once

#include "derivative.hpp"
#include "fields/vector.hpp"

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

    gradient(const mesh&, const stencil&, const bcs::Grid&, const bcs::Object&);

    std::function<void(vector_span)> operator()(scalar_view) const;
};
} // namespace ccs
