#pragma once

#include "types.hpp"

#include "fields/Scalar.hpp"

namespace ccs::operators
{
struct Derivative {
    void operator()(field::ScalarView_Const, field::ScalarView_Mutable) {}
};
} // namespace ccs::operators