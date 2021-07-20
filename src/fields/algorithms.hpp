#pragma once

#include "vector.hpp"

#include <range/v3/algorithm/max.hpp>
#include <range/v3/algorithm/min.hpp>
#include <range/v3/algorithm/minmax.hpp>

namespace ccs
{

template <TupleLike T>
constexpr auto minmax(T&& t)
{
    using Rng = underlying_range_t<T>;
    using V = rs::range_value_t<Rng>;
    using R = rs::minmax_result<V>;
    return transform_reduce(
        [](auto&& rng) {
            // Need to safeguard the call to minmax in case of an empty range.  Empty
            // ranges are common with boundary condition based selection
            if (rs::begin(rng) != rs::end(rng))
                return rs::minmax(FWD(rng));
            else
                return R{std::numeric_limits<V>::max(), std::numeric_limits<V>::lowest()};
        },
        FWD(t),
        [](auto&& acc, auto&& item) {
            return R{rs::min(acc.min, item.min), rs::max(acc.max, item.max)};
        },
        R{std::numeric_limits<V>::max(), std::numeric_limits<V>::lowest()});
}

template <TupleLike T>
constexpr auto max(T&& t)
{
    using Rng = underlying_range_t<T>;
    using V = rs::range_value_t<Rng>;
    return transform_reduce(
        [](auto&& rng) {
            if (rs::begin(rng) != rs::end(rng))
                return rs::max(FWD(rng));
            else
                return std::numeric_limits<V>::lowest();
        },
        FWD(t),
        [](auto&& acc, auto&& item) { return rs::max(acc, item); },
        V{std::numeric_limits<V>::lowest()});
}

template <Vector T, Vector U>
constexpr auto dot(T&& t, U&& u)
{
    auto [x, y, z] = t * u;
    return x + y + z;
}

} // namespace ccs
