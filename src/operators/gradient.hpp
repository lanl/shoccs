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

public:
    gradient() = default;

    std::function<void(vector_span)> operator()(scalar_view);
};
} // namespace ccs