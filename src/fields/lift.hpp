#pragma once

#include "Scalar_fwd.hpp"
#include "Vector_fwd.hpp"
#include <range/v3/view/zip_with.hpp>

namespace ccs::field
{

template <typename Fn>
constexpr auto lift(Fn fn)
{
    return [fn]<typename... Args>(Args && ... rngs)
    {
        constexpr auto location = [](auto&& first, auto&&...) {
            return first.location();
        };

        if constexpr ((tuple::traits::ScalarType<Args> && ...))
            return Scalar(location(rngs...),
                          Tuple{vs::zip_with(fn, selector::D(rngs)...)},
                          Tuple{vs::zip_with(fn, selector::Rx(rngs)...),
                                vs::zip_with(fn, selector::Ry(rngs)...),
                                vs::zip_with(fn, selector::Rz(rngs)...)});
        else
            return Vector(location(rngs...),
                          Tuple{vs::zip_with(fn, selector::Dx(rngs)...),
                                vs::zip_with(fn, selector::Dy(rngs)...),
                                vs::zip_with(fn, selector::Dz(rngs)...)},
                          Tuple{vs::zip_with(fn, selector::Rx(rngs)...),
                                vs::zip_with(fn, selector::Ry(rngs)...),
                                vs::zip_with(fn, selector::Rz(rngs)...)});
    };
}
} // namespace ccs::field