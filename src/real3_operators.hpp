#pragma once
#include "types.hpp"
#include <cmath>

namespace ccs
{
namespace detail
{
template <TupleLike T>
constexpr auto tuple_sz = std::tuple_size_v<std::remove_cvref_t<T>>;
}

#define SHOCCS_GEN_OPERATORS(op, acc)                                                    \
    template <TupleLike U, TupleLike V>                                                  \
    requires(detail::tuple_sz<U> ==                                                      \
             detail::tuple_sz<V>) constexpr std::array<real, detail::tuple_sz<U>>        \
    op(U&& u, V&& v)                                                                     \
    {                                                                                    \
        return []<auto... Is>(std::index_sequence<Is...>, auto&& a, auto&& b)            \
        {                                                                                \
            return std::array<real, detail::tuple_sz<U>>{                                \
                (std::get<Is>(a) acc std::get<Is>(b))...};                               \
        }                                                                                \
        (std::make_index_sequence<detail::tuple_sz<U>>{}, FWD(u), FWD(v));               \
    }                                                                                    \
    template <TupleLike U>                                                               \
    constexpr std::array<real, detail::tuple_sz<U>> op(U&& u, Numeric auto v)            \
    {                                                                                    \
        return [b = v]<auto... Is>(std::index_sequence<Is...>, auto&& a)                 \
        {                                                                                \
            return std::array<real, detail::tuple_sz<U>>{(std::get<Is>(a) acc b)...};    \
        }                                                                                \
        (std::make_index_sequence<detail::tuple_sz<U>>{}, FWD(u));                       \
    }                                                                                    \
    template <TupleLike U>                                                               \
    constexpr std::array<real, detail::tuple_sz<U>> op(Numeric auto v, U&& u)            \
    {                                                                                    \
        return [b = v]<auto... Is>(std::index_sequence<Is...>, auto&& a)                 \
        {                                                                                \
            return std::array<real, detail::tuple_sz<U>>{(b acc std::get<Is>(a))...};    \
        }                                                                                \
        (std::make_index_sequence<detail::tuple_sz<U>>{}, FWD(u));                       \
    }

SHOCCS_GEN_OPERATORS(operator*, *)
SHOCCS_GEN_OPERATORS(operator/, /)
SHOCCS_GEN_OPERATORS(operator+, +)
SHOCCS_GEN_OPERATORS(operator-, -)

#undef SHOCCS_GEN_OPERATORS

template <TupleLike U, TupleLike V>
requires(detail::tuple_sz<U> == detail::tuple_sz<V>) constexpr real dot(U&& u, V&& v)
{
    return []<auto... Is>(std::index_sequence<Is...>, auto&& a, auto&& b)
    {
        return ((std::get<Is>(a) * std::get<Is>(b)) + ...);
    }
    (std::make_index_sequence<detail::tuple_sz<U>>{}, FWD(u), FWD(v));
}

template <TupleLike U>
real length(U&& u)
{
    return std::sqrt(dot(u, u));
}

} // namespace ccs