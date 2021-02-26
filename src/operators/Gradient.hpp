#pragma once

#include "Derivative.hpp"

#include "fields/SystemField.hpp"

namespace ccs::operators
{

class Gradient
{
    Derivative dx;
    Derivative dy;
    Derivative dz;

public:
    Gradient() = default;

    std::function<void(field::VectorView_Mutable)> operator()(field::ScalarView_Const);
};
} // namespace ccs::operators