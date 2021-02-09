#pragma once

#include "Scalar.hpp"
#include <range/v3/view/transform.hpp>

namespace ccs::field
{

template <typename Fn>
constexpr auto lift(Fn fn)
{
    return [t = vs::transform(fn)](auto&& rng) {
        return Scalar{tuple::Tuple{rng | selector::D | t},
                      tuple::Tuple{rng | selector::Rx | t,
                                   rng | selector::Ry | t,
                                   rng | selector::Rz | t}};
    };
}
} // namespace ccs::field