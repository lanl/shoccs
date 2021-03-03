#pragma once

#include "Tuple_fwd.hpp"
#include "types.hpp"
#include <range/v3/view/repeat.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/zip.hpp>
#include <range/v3/view/zip_with.hpp>
#include <tuple>

namespace ccs::field::tuple::lazy
{

template <typename T>
struct ContainterMath;

class ContainterAccess
{
    template <typename T>
    friend class ContainterMath;
};

template <typename T>
struct ContainerMath {

private:
};

template <typename T>
struct ViewMath;

class ViewAccess
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
        constexpr auto N = std::tuple_size_v<U>;

        [&u, v ]<auto... Is>(std::index_sequence<Is...>)
        {
            auto f = [v](auto&& r) {
                for (auto&& x : r) x += v;
            };
            (f(get<Is>(u)), ...);
        }
        (std::make_index_sequence<N>{});

        return u;
    }

    template <std::derived_from<T> U, traits::TupleLike V>
    requires traits::OutputTuple<U, V> friend U& operator+=(U& u, V&& v)
    {
        constexpr auto N = std::tuple_size_v<U>;

        [&]<auto... Is>(std::index_sequence<Is...>)
        {
            auto f = [](auto&& self, auto&& other) {
                for (auto&& [x, y] : vs::zip(self, other)) x += y;
            };
            (f(get<Is>(u), get<Is>(v)), ...);
        }
        (std::make_index_sequence<N>{});

        return u;
    }

    template <typename U, Numeric V>
    requires std::derived_from<std::remove_cvref_t<U>, T>&&
        From_View<U> friend constexpr auto
        operator+(U&& u, V v)
    {
        constexpr auto N = std::tuple_size_v<std::remove_cvref_t<U>>;
        constexpr bool nested = requires(U u, V v) { operator+(get<0>(u), v); };

        if constexpr (nested) {
            return []<auto... Is>(std::index_sequence<Is...>, auto&& u, auto v)
            {
                return std::invoke(create_from_view<U>, u, operator+(get<Is>(u), v)...);
            }
            (std::make_index_sequence<N>{}, FWD(u), v);
        } else {
            return []<auto... Is>(std::index_sequence<Is...>, auto&& u, auto v)
            {
                return std::invoke(create_from_view<U>,
                                   u,
                                   vs::zip_with(std::plus{},
                                                get<Is>(u),
                                                vs::repeat_n(v, get<Is>(u).size()))...);
            }
            (std::make_index_sequence<N>{}, FWD(u), v);
        }
    }

    template <typename U, Numeric V>
    requires std::derived_from<std::remove_cvref_t<U>, T>&&
        From_View<U> friend constexpr auto
        operator+(V v, U&& u)
    {
        constexpr auto N = std::tuple_size_v<std::remove_cvref_t<U>>;
        constexpr bool nested = requires(U u, V v) { operator+(v, get<0>(u)); };

        if constexpr (nested) {
            return []<auto... Is>(std::index_sequence<Is...>, auto v, auto&& u)
            {
                return std::invoke(create_from_view<U>, u, operator+(v, get<Is>(u))...);
            }
            (std::make_index_sequence<N>{}, v, FWD(u));
        } else {
            return []<auto... Is>(std::index_sequence<Is...>, auto v, auto&& u)
            {
                return std::invoke(create_from_view<U>,
                                   u,
                                   vs::zip_with(std::plus{},
                                                vs::repeat_n(v, get<Is>(u).size()),
                                                get<Is>(u))...);
            }
            (std::make_index_sequence<N>{}, v, FWD(u));
        }
    }

    template <typename U, typename V>
    requires std::derived_from<std::remove_cvref_t<U>, T>&&
        From_View<U, V> friend constexpr auto
        operator+(U&& u, V&& v)
    {
        constexpr auto N = std::tuple_size_v<std::remove_cvref_t<U>>;
        constexpr bool nested = requires(U u, V v) { operator+(get<0>(u), get<0>(v)); };

        if constexpr (nested) {
            return []<auto... Is>(std::index_sequence<Is...>, auto&& u, auto&& v)
            {
                return std::invoke(
                    create_from_view<U, V>, u, v, operator+(get<Is>(u), get<Is>(v))...);
            }
            (std::make_index_sequence<N>{}, v, FWD(u));
        } else {
            return []<auto... Is>(std::index_sequence<Is...>, auto&& u, auto&& v)
            {
                return std::invoke(create_from_view<U, V>,
                                   u,
                                   v,
                                   vs::zip_with(std::plus{}, get<Is>(u), get<Is>(v))...);
            }
            (std::make_index_sequence<N>{}, FWD(u), FWD(v));
        }
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