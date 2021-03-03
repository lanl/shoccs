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

template <traits::TupleLike T>
constexpr auto TupleIndex =
    std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>{};

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
    (TupleIndex<T>);
}

template <traits::TupleLike T, traits::InvocableOver<T> F>
void for_each(T&& t, F&& f)
{
    [&]<auto... Is>(std::index_sequence<Is...>) { (std::invoke(f, get<Is>(t)), ...); }
    (TupleIndex<T>);
}

template <traits::TupleLike T, traits::TemplatedInvocableOver<T> F>
void for_each(T&& t, F&& f)
{
    [&]<auto... Is>(std::index_sequence<Is...>)
    {
        (f.template operator()<Is>(get<Is>(t)), ...);
    }
    (TupleIndex<T>);
}

template <traits::TupleLike T, traits::TupleInvocableOver<T> F>
void for_each(T&& t, F&& f)
{
    [&]<auto... Is>(std::index_sequence<Is...>)
    {
        (std::invoke(get<Is>(f), get<Is>(t)), ...);
    }
    (TupleIndex<T>);
}

// Given a container and and input range, attempt to resize the container.
// Either way, copy the input range into the container - may not be safe.
template <traits::Range R, traits::OutputRange<R> C>
void resize_and_copy(C&& container, R&& r)
{
    constexpr bool can_resize = requires(C c, R r) { c.resize(rs::size(r)); };
    constexpr bool compare_sizes = requires(C c, R r) { rs::size(c) < rs::size(r); };
    if constexpr (can_resize) {
        container.resize(rs::size(r));
        rs::copy(FWD(r), rs::begin(container));
    } else if constexpr (compare_sizes) {
        auto min_sz =
            rs::size(container) < rs::size(r) ? rs::size(container) : rs::size(r);
        // note that copy_n takes an input iterator rather than a range
        rs::copy_n(rs::begin(r), min_sz, rs::begin(container));
    } else {
        rs::copy(FWD(r), rs::begin(container));
    }
}

template <Numeric N, traits::OutputRange<N> T>
void resize_and_copy(T&& t, N n)
{
    rs::fill(FWD(t), n);
}

template <typename R, traits::OutputTuple<R> T>
requires(!traits::TupleLike<R>) void resize_and_copy(T&& t, R&& r)
{
    for_each(FWD(t), [r = FWD(r)](auto&& e) { resize_and_copy(FWD(e), r); });
}

template <traits::TupleLike R, traits::OutputTuple<R> T>
void resize_and_copy(T&& t, R&& r)
{
    for_each(
        FWD(t), [r = FWD(r)]<auto I>(auto&& e) { resize_and_copy(FWD(e), get<I>(r)); });
}

// Construct and return a new container from an input container.  Preference
// is given to delegating to direct construction.  Falls back to construction
// via input ranges.
template <typename To, traits::FromTupleDirect<To> T>
auto to_tuple(T&& t)
{
    return [&]<auto... Is>(std::index_sequence<Is...>)
    {
        return std::tuple<std::tuple_element_t<Is, To>...>{
            std::tuple_element_t<Is, To>(get<Is>(t))...};
    }
    (TupleIndex<T>);
}

template <typename To, traits::FromTupleRange<To> T>
requires(!traits::FromTupleDirect<T, To>) auto to_tuple(T&& t)
{
    return [&]<auto... Is>(std::index_sequence<Is...>)
    {
        return std::tuple<std::tuple_element_t<Is, To>...>{
            std::tuple_element_t<Is, To>(rs::begin(get<Is>(t)), rs::end(get<Is>(t)))...};
    }
    (TupleIndex<T>);
}

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
            std::tuple<Args...>{Args{rs::begin(get<Is>(t)), rs::end(get<Is>(t))}...};
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