#pragma once

#include "r_tuple_fwd.hpp"
#include <tuple>

#include <range/v3/view/all.hpp>

namespace ccs
{
namespace detail
{
template <typename T>
using all_t = decltype(vs::all(std::declval<T>()));

template <typename F, typename T>
requires requires(F f, T t)
{
    f(std::get<0>(t));
}
constexpr auto tuple_map(F&& f, T&& t)
{
    return [&f, &t ]<auto... Is>(std::index_sequence<Is...>)
    {
        return std::tuple{f(std::get<Is>(t))...};
    }
    (std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>{});
}
} // namespace detail

template <typename T>
concept All = rs::range<T&>&& rs::viewable_range<T>;

namespace detail
{
template <typename... Args>
struct owning_tuple {
    using Type = owning_tuple<Args...>;
    static constexpr int N = sizeof...(Args);
    std::tuple<Args...> t;

    owning_tuple(Args&&... args) : t{FWD(args)...} {}

    template <rs::sized_range... Ranges>
    // requires std::constructible_from <
    owning_tuple(Ranges&&... r) : t{Args{rs::begin(r), rs::end(r)}...}
    {
    }

    template <rs::input_range... Ranges>
    owning_tuple& operator=(Ranges&&... r)
    {
        t = std::tuple{Args{rs::begin(r), rs::end(r)}...};
        return *this;
    }
};

template <typename... Args>
owning_tuple(Args&&...) -> owning_tuple<std::remove_reference_t<Args>...>;

} // namespace detail

template <typename... Args>
struct r_tuple : detail::owning_tuple<Args...> {
    using Own = detail::owning_tuple<Args...>;
    using Type = r_tuple<Args...>;

    std::tuple<detail::all_t<Args&>...> r;

    r_tuple(Args&&... args) : Own{FWD(args)...}, r{detail::tuple_map(vs::all, this->t)} {}
};

// specialize for one arg to treat as a range more easily;
template <typename T>
struct r_tuple<T> : detail::owning_tuple<T>, detail::all_t<T&> {
    using Own = detail::owning_tuple<T>;
    using Range = detail::all_t<T&>;
    using Type = r_tuple<T>;

    r_tuple(T&& t) : Own{FWD(t)}, Range{vs::all(std::get<0>(this->t))} {}

    template <rs::input_range R>
    requires(!std::same_as<Type, std::remove_cvref_t<R>>) r_tuple(R&& r)
        : Own{FWD(r)}, Range{vs::all(std::get<0>(this->t))}
    {
    }

    template <rs::input_range R>
    requires(!std::same_as<Type, std::remove_cvref_t<R>>) r_tuple& operator=(R&& r)
    {
        static_cast<Own&>(*this) = FWD(r);
        // is there a better way to do this?
        static_cast<Range&>(*this) = vs::all(get<0>(this->t));
        return *this;
    }
};

template <All... Args>
struct r_tuple<Args...> {
    static constexpr int N = sizeof...(Args);
    std::tuple<detail::all_t<Args>...> r;

    r_tuple(Args&&... args) : r{vs::all(FWD(args))...} {};
};

// specialize for one arg to treat as a range more easily;

template <All R>
struct r_tuple<R> : detail::all_t<R> {
    using Range = detail::all_t<R>;
    static constexpr int N = 1;
};

template <typename... Args>
r_tuple(Args&&...) -> r_tuple<Args...>;

template <int I, typename R>
constexpr decltype(auto) get(R&& r)
{
    if constexpr (r.N == 1) {
        using Base = typename std::remove_cvref_t<R>::Range;
        return static_cast<Base&>(r);
    } else {
        return std::get<I>(FWD(r).r);
    }
}

} // namespace ccs