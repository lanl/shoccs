#pragma once

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/all.hpp>

#include "tuple_math.hpp"
#include "tuple_pipe.hpp"
#include "tuple_utils.hpp"

#include <iostream>

namespace ccs
{
// this base class stores the view of (or pointer to) the data in a tuple.  It should be
// inherited before single_view to ensure the tuple is initialized correctly.
template <typename... Args>
struct view_tuple_base {

    view_tuple_base() = default;
    constexpr view_tuple_base(Args&&... args){};

    template <typename... C>
    constexpr view_tuple_base(const container_tuple<C...>& x)
    {
    }
};

template <All... Args>
struct view_tuple_base<Args...> {
private:
    static constexpr bool Output = (OutputRange<Args> && ...);

public:
    std::tuple<vs::all_t<Args>...> v;

    view_tuple_base() = default;
    view_tuple_base(const view_tuple_base&) = default;
    view_tuple_base(view_tuple_base&&) = default;
    // defaults are triggered when Output is false
    view_tuple_base& operator=(const view_tuple_base&) = default;
    view_tuple_base& operator=(view_tuple_base&&) = default;

    explicit constexpr view_tuple_base(Args&&... args) requires(sizeof...(Args) > 0)
        : v{vs::all(FWD(args))...} {};

    template <NonTupleRange... Ranges>
        requires(std::constructible_from<Args, Ranges>&&...)
    explicit constexpr view_tuple_base(Ranges&&... args) : v{vs::all(Args{FWD(args)})...}
    {
    }

    template <TupleLike T>
    explicit constexpr view_tuple_base(T&& t) : v{tuple_map(vs::all, FWD(t))}
    {
    }

    constexpr view_tuple_base& operator=(const view_tuple_base& x) requires Output
    {
        resize_and_copy(*this, x);
        return *this;
    }

    constexpr view_tuple_base& operator=(view_tuple_base&& x) requires Output
    {
        resize_and_copy(*this, MOVE(x));
        return *this;
    }

    // When assigning from a container_tuple, make sure we don't copy anything and instead
    // just take news views
    template <OwningTuple C>
        requires(!ViewClosures<C>)
    constexpr view_tuple_base& operator=(C&& c)
    {
        v = tuple_map(vs::all, FWD(c));
        return *this;
    }

    // This overload is only reached when called on a Tuple that does not
    // have a corresponding container.  Thus, this call results in a copy operation
    // rather than simply resetting the view to the container
    template <typename T>
        requires(!OwningTuple<T> && OutputTuple<view_tuple_base, T>)
    constexpr view_tuple_base& operator=(T&& t)
    {
        // calling here
        resize_and_copy(*this, FWD(t));
        return *this;
    }

    // Check for direct assignment betweeen components.  Needed for Tuples of Selections
    template <typename T>
        requires(!OwningTuple<T> &&
                 (!OutputTuple<view_tuple_base, T>)&&AssignableDirect<view_tuple_base, T>)
    constexpr view_tuple_base& operator=(T&& t)
    {
        for_each([t = FWD(t)](auto&& x) { x = t; }, *this);
        return *this;
    }

    template <typename T>
        requires(!OutputTuple<view_tuple_base, T> &&
                 AssignableDirectTuple<view_tuple_base, T>)
    constexpr view_tuple_base& operator=(T&& t)
    {
        for_each([](auto&& x, auto&& y) { x = FWD(y); }, *this, FWD(t));
        return *this;
    }

    template <NonTupleRange T>
    friend constexpr bool operator==(const view_tuple_base& x, const T& y)
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return (rs::equal(get<Is>(x), y) && ...);
        }
        (sequence<view_tuple_base>);
    }

    template <SimilarTuples<view_tuple_base> T>
    friend constexpr bool operator==(const view_tuple_base& x, const T& y)
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return (rs::equal(get<Is>(x), get<Is>(y)) && ...);
        }
        (sequence<view_tuple_base>);
    }
};

template <typename... Args>
view_tuple_base(Args&&...) -> view_tuple_base<viewable_range_by_value<Args>...>;

template <std::size_t I, ViewTupleBase V>
constexpr decltype(auto) get(V&& v) noexcept
{
    return std::get<I>(FWD(v).v);
}

} // namespace ccs

// specialize tuple_size
namespace std
{
template <typename... Args>
struct tuple_size<ccs::view_tuple_base<Args...>>
    : std::integral_constant<size_t, sizeof...(Args)> {
};

template <size_t I, typename... Args>
struct tuple_element<I, ccs::view_tuple_base<Args...>>
    : tuple_element<I, decltype(declval<ccs::view_tuple_base<Args...>>().v)> {
};
} // namespace std

namespace ccs
{

// this base class allows for treating tuples of one parameters directly as ranges
template <typename... Args>
struct single_view {

    single_view() = default;

    single_view(const single_view&) = default;
    single_view(single_view&&) = default;
    single_view& operator=(const single_view&) = default;
    single_view& operator=(single_view&&) = default;

    template <typename... T>
    constexpr single_view(T&&...)
    {
    }

    template <typename... T>
    constexpr single_view& operator=(T&&...)
    {
        return *this;
    }
};

template <All A>
struct single_view<A> : vs::all_t<A> {
private:
    using view = vs::all_t<A>;

public:
    single_view() = default;
    single_view(const single_view&) = default;
    single_view(single_view&&) = default;
    single_view& operator=(const single_view&) = default;
    single_view& operator=(single_view&&) = default;

    constexpr single_view(A&& a) : view(vs::all(FWD(a))) {}

    template <TupleLike T>
    constexpr single_view(T&& t) : view(get<0>(t))
    {
    }

    constexpr single_view& operator=(A&& a)
    {
        view::operator=(vs::all(FWD(a)));
        return *this;
    }

    template <TupleLike T>
    constexpr single_view& operator=(T&& t)
    {
        view::operator=(get<0>(t));
        return *this;
    }
};

template <typename... Args>
single_view(Args&&...) -> single_view<Args...>;

template <std::size_t I, ViewTuple V>
constexpr decltype(auto) get(V&& v) noexcept;

// r_tuple's inherit from this base class which combines single_view/base_view_tuple
// into a workable unified abstraction
template <typename... Args>
struct view_tuple : view_tuple_base<Args...>,
                    single_view<Args...>,
                    detail::tuple_math<view_tuple<Args...>>,
                    detail::tuple_pipe<view_tuple<Args...>> {
private:
    using view = single_view<Args...>;
    static constexpr bool Output = (OutputRange<Args> && ...);

    friend class tuple_math_access;
    friend class tuple_pipe_access;

public:
    using base = view_tuple_base<Args...>;

    explicit view_tuple() = default;
    explicit constexpr view_tuple(Args&&... args) requires(sizeof...(Args) > 0)
        : base{FWD(args)...}, view{*this}
    {
    }

    // Forward all construction to view_tuple_base
    template <typename... T>
        requires std::constructible_from<base, T...>
    explicit constexpr view_tuple(T&&... t) : base{FWD(t)...}, view{*this} {}

    // Foward all assignment to view_tuple_base
    template <typename T>
        requires std::is_assignable_v<base&, T>
    constexpr view_tuple& operator=(T&& t)
    {
        base::operator=(FWD(t));
        view::operator=(*this);
        return *this;
    }
};

template <typename... Args>
view_tuple(Args&&...) -> view_tuple<viewable_range_by_value<Args>...>;

template <std::size_t I, ViewTuple V>
constexpr decltype(auto) get(V&& v) noexcept
{
    using B = typename std::remove_cvref_t<V>::base;
    return get<I>(static_cast<boost::copy_cv_ref_t<B, V&&>>(FWD(v)));
}

} // namespace ccs

// specialize tuple_size
namespace std
{
template <typename... Args>
struct tuple_size<ccs::view_tuple<Args...>>
    : std::integral_constant<size_t, sizeof...(Args)> {
};

template <size_t I, typename... Args>
struct tuple_element<I, ccs::view_tuple<Args...>>
    : tuple_element<I, ccs::view_tuple_base<Args...>> {
};
} // namespace std
