#pragma once

#include "fields/SystemField.hpp"

namespace ccs::operators
{

struct Gradient {

    template<field::tuple::traits::ScalarType U>
    constexpr auto operator()(U&& u) {
        return field::SimpleVector<std::vector<real>>{};
    }

};
} // namespace ccs::operators