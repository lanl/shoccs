#pragma once

#include "r_tuple_fwd.hpp"
#include "types.hpp"
#include <range/v3/view/repeat.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/zip.hpp>
#include <range/v3/view/zip_with.hpp>
#include <tuple>

namespace ccs::lazy
{

template <typename T>
struct containter_math_crtp;

class containter_math_access
{
    template <typename T>
    friend class containter_math_crtp;
};

template <typename T>
struct container_math_crtp {

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
struct view_math_crtp;

class view_math_access
{
    template <typename T>
    friend class view_math_crtp;
};

template <typename T>
struct view_math_crtp {

private:
    template <typename U, Numeric V>
    requires std::derived_from<std::remove_cvref_t<U>, T>&&
        traits::From_View<U> friend constexpr auto
        operator+(U&& u, V v)
    {
        constexpr bool nested = requires(U u, V v)
        {
            u.template get<0>();
            u.template get<0>() + v;
        };

        if constexpr (nested) {
            return []<auto... Is>(std::index_sequence<Is...>, auto&& u, auto v)
            {
                return std::invoke(traits::create_from_view<U>,
                                   u,
                                   operator+(u.template get<Is>(), v)...);
            }
            (std::make_index_sequence<u.N>{}, FWD(u), v);
        } else {
            return []<auto... Is>(std::index_sequence<Is...>, auto&& u, auto v)
            {
                return std::invoke(traits::create_from_view<U>,
                                   u,
                                   vs::zip_with(std::plus{},
                                                view<Is>(u),
                                                vs::repeat_n(v, view<Is>(u).size()))...);
            }
            (std::make_index_sequence<u.N>{}, FWD(u), v);
        }
    }

    template <typename U, Numeric V>
    requires std::derived_from<std::remove_cvref_t<U>, T>&&
        traits::From_View<U> friend constexpr auto
        operator+(V v, U&& u)
    {
        constexpr bool nested = requires(U u, V v)
        {
            u.template get<0>();
            v + u.template get<0>();
        };

        if constexpr (nested) {
            return []<auto... Is>(std::index_sequence<Is...>, auto v, auto&& u)
            {
                return std::invoke(traits::create_from_view<U>,
                                   u,
                                   operator+(v, u.template get<Is>())...);
            }
            (std::make_index_sequence<u.N>{}, v, FWD(u));
        } else {
            return []<auto... Is>(std::index_sequence<Is...>, auto v, auto&& u)
            {
                return std::invoke(traits::create_from_view<U>,
                                   u,
                                   vs::zip_with(std::plus{},
                                                vs::repeat_n(v, view<Is>(u).size()),
                                                view<Is>(u))...);
            }
            (std::make_index_sequence<u.N>{}, v, FWD(u));
        }
    }

    template <typename U, typename V>
    requires std::derived_from<std::remove_cvref_t<U>, T>&&
        traits::From_View<U, V>&& requires(V v)
    {
        v.view();
    }
    friend constexpr auto operator+(U&& u, V&& v)
    {
        constexpr bool nested = requires(U u, V v)
        {
            u.template get<0>();
            v.template get<0>();
            u.template get<0>() + v.template get<0>();
        };

        if constexpr (nested) {
            return []<auto... Is>(std::index_sequence<Is...>, auto&& u, auto&& v)
            {
                return std::invoke(traits::create_from_view<U, V>,
                                   u,
                                   v,
                                   operator+(u.template get<Is>(), v.template get<Is>())...);
            }
            (std::make_index_sequence<u.N>{}, v, FWD(u));
        } else {
            return []<auto... Is>(std::index_sequence<Is...>, auto&& u, auto&& v)
            {
                return std::invoke(
                    traits::create_from_view<U, V>,
                    u,
                    v,
                    vs::zip_with(std::plus{}, view<Is>(u), view<Is>(v))...);
            }
            (std::make_index_sequence<std::max(u.N, v.N)>{}, FWD(u), FWD(v));
        }
    }
};

} // namespace ccs::lazy