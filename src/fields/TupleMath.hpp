#pragma once

#include "TupleUtils.hpp"

#include <range/v3/view/repeat.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/zip.hpp>
#include <range/v3/view/zip_with.hpp>
#include <tuple>

namespace ccs::field::tuple::lazy
{

template <typename T>
struct ViewMath;

class ViewMathAccess
{
    template <typename T>
    friend class ViewMath;
};

#define SHOCCS_GEN_OPERATORS(op, f)                                                      \
    template <typename U, Numeric V>                                                     \
    requires std::derived_from<std::remove_cvref_t<U>, T>&&                              \
        From_View<U> friend constexpr auto                                               \
        op(U&& u, V v)                                                                   \
    {                                                                                    \
        constexpr bool nested = requires(U u, V v) { op(get<0>(u), v); };                \
                                                                                         \
        if constexpr (nested) {                                                          \
            static_assert(true, "should not be here");                                   \
            return []<auto... Is>(std::index_sequence<Is...>, auto&& u, auto v)          \
            {                                                                            \
                return std::invoke(create_from_view<U>, u, op(get<Is>(u), v)...);        \
            }                                                                            \
            (std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<U>>>{},      \
             FWD(u),                                                                     \
             v);                                                                         \
        } else {                                                                         \
            return []<auto... Is>(std::index_sequence<Is...>, auto&& u, auto v)          \
            {                                                                            \
                return std::invoke(                                                      \
                    create_from_view<U>,                                                 \
                    u,                                                                   \
                    vs::zip_with(f, get<Is>(u), vs::repeat_n(v, get<Is>(u).size()))...); \
            }                                                                            \
            (std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<U>>>{},      \
             FWD(u),                                                                     \
             v);                                                                         \
        }                                                                                \
    }                                                                                    \
                                                                                         \
    template <typename U, Numeric V>                                                     \
    requires std::derived_from<std::remove_cvref_t<U>, T>&&                              \
        From_View<U> friend constexpr auto                                               \
        op(V v, U&& u)                                                                   \
    {                                                                                    \
        constexpr bool nested = requires(U u, V v)                                       \
        {                                                                                \
            u.template get<0>();                                                         \
            op(v, u.template get<0>());                                                  \
        };                                                                               \
                                                                                         \
        if constexpr (nested) {                                                          \
            static_assert(true, "should not be here");                                   \
            return []<auto... Is>(std::index_sequence<Is...>, auto v, auto&& u)          \
            {                                                                            \
                return std::invoke(                                                      \
                    create_from_view<U>, u, op(v, u.template get<Is>())...);             \
            }                                                                            \
            (std::make_index_sequence<u.N>{}, v, FWD(u));                                \
        } else {                                                                         \
            return []<auto... Is>(std::index_sequence<Is...>, auto v, auto&& u)          \
            {                                                                            \
                return std::invoke(create_from_view<U>,                                  \
                                   u,                                                    \
                                   vs::zip_with(f,                                       \
                                                vs::repeat_n(v, view<Is>(u).size()),     \
                                                view<Is>(u))...);                        \
            }                                                                            \
            (std::make_index_sequence<u.N>{}, v, FWD(u));                                \
        }                                                                                \
    }                                                                                    \
                                                                                         \
    template <typename U, typename V>                                                    \
    requires std::derived_from<std::remove_cvref_t<U>, T>&& From_View<U, V>&& requires(  \
        V v)                                                                             \
    {                                                                                    \
        v.view();                                                                        \
    }                                                                                    \
    friend constexpr auto op(U&& u, V&& v)                                               \
    {                                                                                    \
        constexpr bool nested = requires(U u, V v)                                       \
        {                                                                                \
            u.template get<0>();                                                         \
            v.template get<0>();                                                         \
            op(u.template get<0>(), v.template get<0>());                                \
        };                                                                               \
                                                                                         \
        if constexpr (nested) {                                                          \
            return []<auto... Is>(std::index_sequence<Is...>, auto&& u, auto&& v)        \
            {                                                                            \
                return std::invoke(create_from_view<U, V>,                               \
                                   u,                                                    \
                                   v,                                                    \
                                   op(u.template get<Is>(), v.template get<Is>())...);   \
            }                                                                            \
            (std::make_index_sequence<u.N>{}, v, FWD(u));                                \
        } else {                                                                         \
            return []<auto... Is>(std::index_sequence<Is...>, auto&& u, auto&& v)        \
            {                                                                            \
                return std::invoke(create_from_view<U, V>,                               \
                                   u,                                                    \
                                   v,                                                    \
                                   vs::zip_with(f, view<Is>(u), view<Is>(v))...);        \
            }                                                                            \
            (std::make_index_sequence<std::max(u.N, v.N)>{}, FWD(u), FWD(v));            \
        }                                                                                \
    }

template <typename T>
struct ViewMath {

private:
    template <std::derived_from<T> U, Numeric V>
    requires traits::OutputTuple<U, V> friend U& operator+=(U& u, V v)
    {
        for_each(
            [v](auto&& rng) {
                for (auto&& x : rng) x += v;
            },
            u);

        return u;
    }

    template <std::derived_from<T> U, traits::TupleLike V>
    requires traits::OutputTuple<U, V> friend U& operator+=(U& u, V&& v)
    {
        for_each(
            [](auto&& out, auto&& in) {
                for (auto&& [o, i] : vs::zip(out, in)) o += i;
            },
            u,
            FWD(v));

        return u;
    }

    template <typename U, Numeric V>
    requires std::derived_from<std::remove_cvref_t<U>, T> friend constexpr auto
    operator+(U&& u, V v)
    {
        return transform(
            [v](auto&& rng) {
                const auto sz = rs::size(rng);
                return vs::zip_with(std::plus{}, FWD(rng), vs::repeat_n(v, sz));
            },
            FWD(u));
    }

    template <typename U, Numeric V>
    requires std::derived_from<std::remove_cvref_t<U>, T> friend constexpr auto
    operator+(V v, U&& u)
    {
        return transform(
            [v](auto&& rng) {
                const auto sz = rs::size(rng);
                return vs::zip_with(std::plus{}, vs::repeat_n(v, sz), FWD(rng));
            },
            FWD(u));
    }

    template <typename U, typename V>
    requires std::derived_from<std::remove_cvref_t<U>, T>&&
        traits::mp_similar<std::remove_cvref_t<U>,
                           std::remove_cvref_t<V>>::value friend constexpr auto
        operator+(U&& u, V&& v)
    {
        return transform(
            [](auto&& a, auto&& b) { return vs::zip_with(std::plus{}, FWD(a), FWD(b)); },
            FWD(u),
            FWD(v));
    }

#if 0
    SHOCCS_GEN_OPERATORS(operator+, std::plus{})
    SHOCCS_GEN_OPERATORS(operator-, std::minus{})
    SHOCCS_GEN_OPERATORS(operator*, std::multiplies{})
    SHOCCS_GEN_OPERATORS(operator/, std::divides{})
#endif
};
#undef SHOCCS_GEN_OPERATORS

} // namespace ccs::field::tuple::lazy