#pragma once

#include "Selector_fwd.hpp"
#include "types.hpp"
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/view.hpp>

namespace ccs::field::tuple
{

struct Location {
    std::span<const real> x;
    std::span<const real> y;
    std::span<const real> z;
    std::span<const real3> rx;
    std::span<const real3> ry;
    std::span<const real3> rz;

    constexpr auto view() const
    {
        return rs::make_view_closure([this]<traits::SelectionType S>(S&&) {
            constexpr auto J = traits::last_selection_index_v<S>;

            if constexpr (J == 0) {
                return vs::cartesian_product(x, y, z);
            } else if constexpr (traits::selection_index_length_v<S> == 2) {
                constexpr auto I = traits::selection_index_v<S, 0>;
                return get<I>(std::tuple{vs::all(rx), vs::all(ry), vs::all(rz)});
            } else {
                return Tuple{rx, ry, rz};
            }
        }); // detail::location_fn{x, y, z, rx, ry, rz});
    };

    // template <typename R>
    // auto operator()(Selection<R, 0>) const
    // {
    //     return vs::cartesian_product(x, y, z);
    // }
};

} // namespace ccs::field::tuple