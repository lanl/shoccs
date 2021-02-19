#pragma once

#include "fields/SystemField.hpp"

namespace ccs::operators
{

struct Gradient {

    template<typename U, typename V>
    constexpr auto operator()(U&&, V&&) {
        return SystemField{};
    }

};
} // namespace ccs::operators