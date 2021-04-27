#pragma once
#include "fields/Tuple_fwd.hpp"
#include <cmath>

namespace ccs
{
using field::tuple::traits::NumericTupleLike;

namespace detail
{

template <NumericTupleLike T>
constexpr auto tuple_sz = std::tuple_size_v<std::remove_cvref_t<T>>;
} // namespace detail

#define SHOCCS_GEN_OPERATORS(op, acc)                                                    \
    template <NumericTupleLike U, NumericTupleLike V>                                    \
    requires(detail::tuple_sz<U> ==                                                      \
             detail::tuple_sz<V>) constexpr std::array<real, detail::tuple_sz<U>>        \
    op(U&& u, V&& v)                                                                     \
    {                                                                                    \
        return []<auto... Is>(std::index_sequence<Is...>, auto&& a, auto&& b)            \
        {                                                                                \
            using std::get;                                                              \
            return std::array<real, detail::tuple_sz<U>>{                                \
                (get<Is>(a) acc get<Is>(b))...};                                         \
        }                                                                                \
        (std::make_index_sequence<detail::tuple_sz<U>>{}, FWD(u), FWD(v));               \
    }                                                                                    \
    template <NumericTupleLike U>                                                        \
    constexpr std::array<real, detail::tuple_sz<U>> op(U&& u, Numeric auto v)            \
    {                                                                                    \
        return [b = v]<auto... Is>(std::index_sequence<Is...>, auto&& a)                 \
        {                                                                                \
            using std::get;                                                              \
            return std::array<real, detail::tuple_sz<U>>{(get<Is>(a) acc b)...};         \
        }                                                                                \
        (std::make_index_sequence<detail::tuple_sz<U>>{}, FWD(u));                       \
    }                                                                                    \
    template <NumericTupleLike U>                                                        \
    constexpr std::array<real, detail::tuple_sz<U>> op(Numeric auto v, U&& u)            \
    {                                                                                    \
        return [b = v]<auto... Is>(std::index_sequence<Is...>, auto&& a)                 \
        {                                                                                \
            using std::get;                                                              \
            return std::array<real, detail::tuple_sz<U>>{(b acc get<Is>(a))...};         \
        }                                                                                \
        (std::make_index_sequence<detail::tuple_sz<U>>{}, FWD(u));                       \
    }

SHOCCS_GEN_OPERATORS(operator*, *)
SHOCCS_GEN_OPERATORS(operator/, /)
SHOCCS_GEN_OPERATORS(operator+, +)
SHOCCS_GEN_OPERATORS(operator-, -)

#undef SHOCCS_GEN_OPERATORS

template <NumericTupleLike U, NumericTupleLike V>
requires(detail::tuple_sz<U> == detail::tuple_sz<V>) constexpr real dot(U&& u, V&& v)
{
    return []<auto... Is>(std::index_sequence<Is...>, auto&& a, auto&& b)
    {
        using std::get;
        return ((get<Is>(a) * get<Is>(b)) + ...);
    }
    (std::make_index_sequence<detail::tuple_sz<U>>{}, FWD(u), FWD(v));
}

template <NumericTupleLike U>
real length(U&& u)
{
    return std::sqrt(dot(u, u));
}

} // namespace ccs