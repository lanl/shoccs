#pragma once

#include "fields/SystemField.hpp"

namespace ccs::operators
{

struct Gradient {

    auto operator()(field::SimpleScalar<std::span<const real>>)
    {
        return field::SimpleVector<std::vector<real>>{};
    }
};
} // namespace ccs::operators