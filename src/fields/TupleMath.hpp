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
    template <std::derived_from<T> U, Numeric V>
    friend U& operator+=(U& u, V v)
    {
        auto& c = u.container().c;

        [&c, v ]<auto... Is>(std::index_sequence<Is...>)
        {
            auto f = [v](auto&& r) {
                for (auto&& x : r) x += v;
            };
            (f(std::get<Is>(c)), ...);
        }
        (std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<decltype(c)>>>{});

        return u;
    }

    template <std::derived_from<T> U, typename V>
    requires requires(V v)
    {
        v.view();
    }
    friend U& operator+=(U& u, V&& v)
    {
        auto& c = u.container().c;

        [&c, &v ]<auto... Is>(std::index_sequence<Is...>)
        {
            auto f = [](auto&& self, auto&& other) {
                for (auto&& [x, y] : vs::zip(self, other)) x += y;
            };
            (f(std::get<Is>(c), view<Is>(v)), ...);
        }
        (std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<decltype(c)>>>{});

        return u;
    }
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
        constexpr bool nested = requires(U u, V v)                                       \
        {                                                                                \
            u.template get<0>();                                                         \
            op(u.template get<0>(), v);                                                  \
        };                                                                               \
                                                                                         \
        if constexpr (nested) {                                                          \
            return []<auto... Is>(std::index_sequence<Is...>, auto&& u, auto v)          \
            {                                                                            \
                return std::invoke(                                                      \
                    create_from_view<U>, u, op(u.template get<Is>(), v)...);             \
            }                                                                            \
            (std::make_index_sequence<u.N>{}, FWD(u), v);                                \
        } else {                                                                         \
            return []<auto... Is>(std::index_sequence<Is...>, auto&& u, auto v)          \
            {                                                                            \
                return std::invoke(                                                      \
                    create_from_view<U>,                                                 \
                    u,                                                                   \
                    vs::zip_with(                                                        \
                        f, view<Is>(u), vs::repeat_n(v, view<Is>(u).size()))...);        \
            }                                                                            \
            (std::make_index_sequence<u.N>{}, FWD(u), v);                                \
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
    SHOCCS_GEN_OPERATORS(operator+, std::plus{})
    SHOCCS_GEN_OPERATORS(operator-, std::minus{})
    SHOCCS_GEN_OPERATORS(operator*, std::multiplies{})
    SHOCCS_GEN_OPERATORS(operator/, std::divides{})
};
#undef SHOCCS_GEN_OPERATORS

} // namespace ccs::field::tuple::lazy