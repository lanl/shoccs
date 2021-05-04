#pragma once

#include "tuple_fwd.hpp"

#include "container_tuple.hpp"
#include "tuple_utils.hpp"
#include "view_tuple.hpp"

namespace ccs
{

// r_tuple for viewable ref components
template <typename... Args>
struct tuple : container_tuple<Args...>, view_tuple<Args&...> {
    using container = container_tuple<Args...>;
    using view = view_tuple<Args&...>;

    tuple() : container{}, view{*this} {}

    tuple(Args&&... args) : container{FWD(args)...}, view{*this} {}

    template <typename... T>
        requires((sizeof...(Args) > 0) && std::constructible_from<container, T...>)
    tuple(T&&... args) : container{FWD(args)...}, view{*this} {}

    template <typename T>
    requires std::is_assignable_v<container&, T> tuple& operator=(T&& t)
    {
        container::operator=(FWD(t));
        // need to adjust the views incase the container has resized
        view::operator=(*this);
        return *this;
    }

    template <std::invocable<tuple&> Fn>
    tuple& operator=(Fn fn)
    {
        std::invoke(fn, *this);
        return *this;
    }

    // need to define custom copy and move construction/assignment here since the
    // component-wise approach is not correct
    tuple(const tuple& r) : container{r}, view{*this} {}
    tuple& operator=(const tuple& r)
    {
        container::operator=(r);
        view::operator=(*this);
        return *this;
    }

    tuple(tuple&& r) noexcept : container{MOVE(r)}, view{*this} {}

    tuple& operator=(tuple&& r) noexcept
    {
        container::operator=(MOVE(r));
        view::operator=(*this);
        return *this;
    }

    template <NonTupleRange T>
    friend bool operator==(const tuple& x, const T& y) requires NestedTuple<tuple>
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return ((get<Is>(x) == y) && ...);
        }
        (sequence<tuple>);
    }

    template <SimilarTuples<tuple> T>
    friend bool operator==(const tuple& x, const T& y) requires NestedTuple<tuple>
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return ((get<Is>(x) == get<Is>(y)) && ...);
        }
        (sequence<tuple>);
    }

    tuple& as_tuple() & { return *this; }
    const tuple& as_tuple() const& { return *this; }
    tuple&& as_tuple() && { return MOVE(*this); }
};

//
// Non Owning Tuple
//
template <All... Args>
struct tuple<Args...> : view_tuple<Args...> {
    using view = view_tuple<Args...>;

    explicit tuple(Args&&... args) : view{FWD(args)...} {};

    tuple() = default;

    template <typename... T>
        requires(std::constructible_from<view, T...>)
    tuple(T&&... t) : view{FWD(t)...} {}

    template <typename T>
    requires std::is_assignable_v<view&, T> tuple& operator=(T&& t)
    {
        view::operator=(FWD(t));
        return *this;
    }

    template <std::invocable<tuple&> Fn>
    tuple& operator=(Fn fn)
    {
        std::invoke(fn, *this);
        return *this;
    }

    tuple& as_tuple() & { return *this; }
    const tuple& as_tuple() const& { return *this; }
    tuple&& as_tuple() && { return MOVE(*this); }
};

template <typename... Args>
tuple(Args&&...) -> tuple<viewable_range_by_value<Args>...>;
// tuple(Args&&...) -> tuple<Args...>;

// need to caputre view closures by value to meet range-v3 concepts
template <typename... ViewFn>
tuple(vs::view_closure<ViewFn>&...) -> tuple<vs::view_closure<ViewFn>...>;
template <typename... ViewFn>
tuple(const vs::view_closure<ViewFn>&...) -> tuple<vs::view_closure<ViewFn>...>;

template <std::size_t I, Tuple C>
constexpr decltype(auto) get(C&& c)
{
    if constexpr (OwningTuple<C>)
        return get<I>(FWD(c).as_container());
    else {
        using B = typename std::remove_cvref_t<C>::view;
        return get<I>(static_cast<boost::copy_cv_ref_t<B, C&&>>(FWD(c)));

        // return get<I>(FWD(c).as_ViewTuple());
    }
}

template <std::size_t I, std::size_t J, std::size_t... Rest, Tuple C>
constexpr decltype(auto) get(C&& c)
{
    return get<J, Rest...>(get<I>(FWD(c)));
}

template <ListIndex L, Tuple C>
constexpr decltype(auto) get(C&& c)
{
    // this really should only require one lambda but I can't do it without triggering a
    // parse error
    return []<auto... Ls>(std::index_sequence<Ls...>, auto&& c)->decltype(auto)
    {
        return []<auto... Is>(auto&& c, mp_size_t<Is>...)->decltype(auto)
        {
            return get<Is...>(FWD(c));
        }
        (FWD(c), mp_at_c<L, Ls>{}...);
    }
    (std::make_index_sequence<mp_size<L>::value>(), FWD(c));
}

} // namespace ccs

// specialize tuple_size
namespace std
{
template <typename... Args>
struct tuple_size<ccs::tuple<Args...>> : std::integral_constant<size_t, sizeof...(Args)> {
};

template <size_t I, typename... Args>
struct tuple_element<I, ccs::tuple<Args...>>
    : tuple_element<I, ccs::container_tuple<Args...>> {
};

template <size_t I, ccs::All... Args>
struct tuple_element<I, ccs::tuple<Args...>>
    : tuple_element<I, ccs::view_tuple<Args...>> {
};
} // namespace std

namespace ccs
{

template <typename T, auto N>
concept NTuple = Tuple<T> && std::tuple_size_v<std::remove_cvref_t<T>>
== N;

template <typename T>
concept OneTuple = NTuple<T, 1u>;

template <typename T>
concept ThreeTuple = NTuple<T, 3u>;

} // namespace ccs
