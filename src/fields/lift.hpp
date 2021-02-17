#pragma once

#include "Scalar_fwd.hpp"
#include <range/v3/view/zip_with.hpp>

namespace ccs::field
{

template <typename Fn>
constexpr auto lift(Fn fn)
{
    return [fn](auto&&... rngs) {
        constexpr auto location = [](auto&& first, auto&&...) {
            return first.location();
        };

        return Scalar(location(rngs...),
                      Tuple{vs::zip_with(fn, selector::D(rngs)...)},
                      Tuple{vs::zip_with(fn, selector::Rx(rngs)...),
                            vs::zip_with(fn, selector::Ry(rngs)...),
                            vs::zip_with(fn, selector::Rz(rngs)...)});
    };
}
} // namespace ccs::field