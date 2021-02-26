#pragma once

#include "Tuple_fwd.hpp"

#include <tuple>

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/copy_n.hpp>
#include <range/v3/algorithm/fill.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/common.hpp>
#include <range/v3/view/view.hpp>

namespace ccs::field::tuple
{

// Map a function `f` over the elements in tuple `t`.  Returns a new tuple.
template <typename F, typename T>
requires requires(F f, T t)
{
    f(get<0>(t));
}
constexpr auto tuple_map(F&& f, T&& t)
{
    return [&f, &t ]<auto... Is>(std::index_sequence<Is...>)
    {
        return std::tuple{f(get<Is>(t))...};
    }
    (std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>{});
}

// Given a container and and input range, attempt to resize the container.
// Either way, copy the input range into the container - may not be safe.
template <typename C, traits::Range R>
requires requires(C container, R r)
{
    rs::copy(r, rs::begin(container));
}
void resize_and_copy(C& container, R&& r)
{
    constexpr bool can_resize_member = requires(C c, R r) { c.resize(r.size()); };
    constexpr bool can_resize_rs = requires(C c, R r) { c.resize(rs::size(r)); };
    constexpr bool compare_sizes = requires(C c, R r) { rs::size(c) < rs::size(r); };
    if constexpr (can_resize_member) {
        container.resize(r.size());
        rs::copy(FWD(r), rs::begin(container));
    } else if constexpr (can_resize_rs) {
        container.resize(rs::size(r));
        rs::copy(FWD(r), rs::begin(container));
    } else if constexpr (compare_sizes) {
        auto min_sz = std::min(rs::size(container), rs::size(r));
        // note that copy_n takes an input iterator rather than a range
        rs::copy_n(rs::begin(r), min_sz, rs::begin(container));
    } else {
        rs::copy(FWD(r), rs::begin(container));
    }
}

// Construct and return a new container from an input container.  Preference
// is given to delegating to direct construction.  Falls back to construction
// via input ranges.
template <typename... Args, typename T>
auto container_from_container(const T& t)
{
    return [&t]<auto... Is>(std::index_sequence<Is...>)
    {
        // prefer to delegate to direct construction rather than building
        // from ranges.
        // Use () instead of {} to allow constructing vectors from from sizes
        constexpr bool direct = requires(T t)
        {
            std::tuple<Args...>{Args(get<Is>(t))...};
        };
        constexpr bool fromRange = requires(T t)
        {
            std::tuple<Args...>{
                Args{rs::begin(get<Is>(t)), rs::end(get<Is>(t))}...};
        };

        if constexpr (direct)
            return std::tuple<Args...>{Args(get<Is>(t))...};
        else if constexpr (fromRange)
            return std::tuple<Args...>{
                Args{rs::begin(get<Is>(t)), rs::end(get<Is>(t))}...};
        else
            static_assert(true, "Cannot construct container");
    }
    (std::make_index_sequence<sizeof...(Args)>{});
}

// Construct a container from a view by converting the view to a
// common range.  As above, prefer to delegate construction directly
// if possible.  This is helpful for constructing owning r_tuples with sizes.
template <typename... Args, typename T>
auto container_from_view(const T& t)
{
    return [&t]<auto... Is>(std::index_sequence<Is...>)
    {
        constexpr bool direct = requires(T t)
        {
            std::tuple<Args...>{Args(view<Is>(t))...};
        };

        if constexpr (direct) {
            return std::tuple<Args...>{Args(view<Is>(t))...};
        } else {
            auto x = std::tuple{vs::common(view<Is>(t))...};
            return std::tuple<Args...>{
                Args{rs::begin(get<Is>(x)), rs::end(get<Is>(x))}...};
        }
    }
    (std::make_index_sequence<sizeof...(Args)>{});
}
} // namespace ccs::field::tuple