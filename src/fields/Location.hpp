#pragma once

#include "Selector_fwd.hpp"
#include "types.hpp"
#include <range/v3/view/all.hpp>
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

    // constexpr auto

    constexpr auto view() const
    {
        return rs::make_view_closure([this]<traits::SelectionType S>(S&&) {
            if constexpr (traits::is_domain_selection_v<S>)
                return vs::cartesian_product(x, y, z);
            else if constexpr (traits::is_Rx_selection_v<S>)
                return vs::all(rx);
            else if constexpr (traits::is_Ry_selection_v<S>)
                return vs::all(ry);
            else if constexpr (traits::is_Rz_selection_v<S>)
                return vs::all(rz);
            else
                static_assert("unaccounted selection type");
        });
    };

    // template <typename R>
    // auto operator()(Selection<R, 0>) const
    // {
    //     return vs::cartesian_product(x, y, z);
    // }
};

} // namespace ccs::field::tuple