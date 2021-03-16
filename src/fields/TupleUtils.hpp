#pragma once

#include "Tuple_fwd.hpp"

#include <tuple>

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/copy_n.hpp>
#include <range/v3/algorithm/fill.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/common.hpp>
#include <range/v3/view/view.hpp>
#include <range/v3/view/zip_with.hpp>

namespace ccs::field::tuple
{

template <traits::TupleLike T>
constexpr auto TupleIndex =
    std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>{};

namespace detail
{
template <typename>
struct makeTuple_Fn;

template <template <typename...> typename T, typename... Ts>
struct makeTuple_Fn<T<Ts...>> {
    template <typename... Args>
    constexpr auto operator()(Args&&... args) const
    {
        return T{FWD(args)...};
    }
};
} // namespace detail
template <traits::TupleLike T>
constexpr auto makeTuple = detail::makeTuple_Fn<std::remove_cvref_t<T>>{};

template <auto I, traits::TupleLike... T>
constexpr auto get_all(T&&... t)
{
    return std::forward_as_tuple(get<I>(FWD(t))...);
}

template <auto I, traits::TupleLike... T>
constexpr auto get_all_with_index(T&&... t)
{
    return std::forward_as_tuple(traits::mp_size_t<I>{}, get<I>(FWD(t))...);
}

//
// Return a tuple of F applied to the elements of T...
//
template <traits::TupleLike... T, traits::InvocableOver<T...> F>
constexpr auto transform(F&& f, T&&... t)
{
    static_assert(sizeof...(T) > 0);
    using namespace traits;

    using U = mp_front<mp_list<std::remove_cvref_t<T>...>>;

    return [&f]<auto... Is>(std::index_sequence<Is...>, auto&&... ts)
    {
        return makeTuple<U>(std::apply(f, get_all<Is>(ts...))...);
    }
    (TupleIndex<U>, FWD(t)...);
}

//
// Return a tuple of F applied to the elements of T... where the first
// argument of F is an mp_size_t which can be used as constant expression to index a tuple
//
template <traits::TupleLike... T, traits::IndexedInvocableOver<T...> F>
constexpr auto transform(F&& f, T&&... t)
{
    static_assert(sizeof...(T) > 0);
    using namespace traits;

    using U = mp_front<mp_list<std::remove_cvref_t<T>...>>;

    return [&f]<auto... Is>(std::index_sequence<Is...>, auto&&... ts)
    {
        return makeTuple<U>(std::apply(f, get_all_with_index<Is>(ts...))...);
    }
    (TupleIndex<U>, FWD(t)...);
}

//
// Return a tuple of apply the tuple-of-functions F, to the elements of tuples T...
//
template <traits::TupleLike... T, traits::TupleInvocableOver<T...> F>
constexpr auto transform(F&& f, T&&... t)
{
    static_assert(sizeof...(T) > 0);
    using namespace traits;

    using U = mp_front<mp_list<std::remove_cvref_t<T>...>>;

    return [&f]<auto... Is>(std::index_sequence<Is...>, auto&&... ts)
    {
        return makeTuple<U>(std::apply(get<Is>(f), get_all<Is>(ts...))...);
    }
    (TupleIndex<U>, FWD(t)...);
}

//
// Return a tuple of apply the tuple-of-functions F, to the elements of tuples T...
//
template <traits::NestedTupleLike... T, traits::NestedInvocableOver<T...> F>
constexpr auto transform(F&& f, T&&... t)
{
    static_assert(sizeof...(T) > 0);
    using namespace traits;

    using U = mp_front<mp_list<std::remove_cvref_t<T>...>>;

    return [&f]<auto... Is>(std::index_sequence<Is...>, auto&&... ts)
    {
        return makeTuple<U>(
            std::apply([&f](auto&&... args) { return transform(f, FWD(args)...); },
                       get_all<Is>(ts...))...);
    }
    (TupleIndex<U>, FWD(t)...);
}

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

template <traits::TupleLike... T, traits::InvocableOver<T...> F>
void for_each(F&& f, T&&... t)
{
    static_assert(sizeof...(T) > 0);
    using namespace traits;

    using U = mp_front<mp_list<std::remove_cvref_t<T>...>>;

    [&f]<auto... Is>(std::index_sequence<Is...>, auto&&... ts)
    {
        (std::apply(f, get_all<Is>(ts...)), ...);
    }
    (TupleIndex<U>, FWD(t)...);
}

template <traits::TupleLike... T, traits::IndexedInvocableOver<T...> F>
void for_each(F&& f, T&&... t)
{
    static_assert(sizeof...(T) > 0);
    using namespace traits;

    using U = mp_front<mp_list<std::remove_cvref_t<T>...>>;

    [&f]<auto... Is>(std::index_sequence<Is...>, auto&&... ts)
    {
        (std::apply(f, get_all_with_index<Is>(ts...)), ...);
    }
    (TupleIndex<U>, FWD(t)...);
}

template <traits::TupleLike... T, traits::TupleInvocableOver<T...> F>
void for_each(F&& f, T&&... t)
{
    static_assert(sizeof...(T) > 0);
    using namespace traits;

    using U = mp_front<mp_list<std::remove_cvref_t<T>...>>;

    [&f]<auto... Is>(std::index_sequence<Is...>, auto&&... ts)
    {
        (std::apply(get<Is>(f), get_all<Is>(ts...)), ...);
    }
    (TupleIndex<U>, FWD(t)...);
}

template <traits::NestedTupleLike... T, traits::NestedInvocableOver<T...> F>
void for_each(F&& f, T&&... t)
{
    static_assert(sizeof...(T) > 0);
    using namespace traits;

    using U = mp_front<mp_list<std::remove_cvref_t<T>...>>;

    [&f]<auto... Is>(std::index_sequence<Is...>, auto&&... ts)
    {
        (std::apply([&f](auto&&... args) { for_each(f, FWD(args)...); },
                    get_all<Is>(ts...)),
         ...);
    }
    (TupleIndex<U>, FWD(t)...);
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

// Given a container and and input range, attempt to resize the container.
// Either way, copy the input range into the container - may not be safe.
template <traits::Range R, traits::OutputRange<R> C>
requires traits::RefView<C> void resize_and_copy(C&& container, R&& r)
{
    resize_and_copy(FWD(container).base(), FWD(r));
}

template <Numeric N, traits::OutputRange<N> T>
void resize_and_copy(T&& t, N n)
{
    rs::fill(FWD(t), n);
}

template <typename R, traits::OutputTuple<R> T>
requires(!traits::TupleLike<R>) void resize_and_copy(T&& t, R&& r)
{
    for_each([&r](auto&& e) { resize_and_copy(FWD(e), r); }, FWD(t));
}

template <traits::TupleLike R, traits::OutputTuple<R> T>
void resize_and_copy(T&& t, R&& r)
{
    // here
    for_each([](auto&& e, auto&& r) { resize_and_copy(FWD(e), FWD(r)); }, FWD(t), FWD(r));
}

//
// `to` is used to recursively construct one tuple from another tuple where the elements
// must be either directly constructible from one another or constructible from begin/end
//
template <typename R, typename Arg>
requires std::constructible_from<R, Arg> constexpr R to(Arg&& arg)
{
    return R(FWD(arg));
}

template <typename R, typename Arg>
    requires(!std::constructible_from<R, Arg>) &&
    traits::ConstructibleFromRange<R, Arg> constexpr R to(Arg&& arg)
{
    if constexpr (rs::common_range<Arg>) {
        return R(rs::begin(arg), rs::end(arg));
    } else {
        auto rng = vs::common(arg);
        return R(rs::begin(rng), rs::end(rng));
    }
}

template <typename R, typename Arg>
    requires(!std::constructible_from<R, Arg>) &&
    traits::TupleFromTuple<R, Arg> constexpr R to(Arg&& arg)
{
    return []<auto... Is>(std::index_sequence<Is...>, auto&& x)
    {
        return makeTuple<R>(to<std::tuple_element_t<Is, R>>(get<Is>(x))...);
    }
    (TupleIndex<R>, FWD(arg));
}

//
// lifting a function allows us to more easily call the function on each element
// of the ranges of the tuple
//
template <typename Fn>
constexpr auto lift(Fn fn)
{
    return [fn](auto&&... tup) {
        return transform(
            [fn]<rs::range... Args>(Args && ... rngs) {
                return vs::zip_with(fn, FWD(rngs)...);
            },
            FWD(tup)...);
    };
}
} // namespace ccs::field::tuple