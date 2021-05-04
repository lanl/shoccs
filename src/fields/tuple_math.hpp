#pragma once

#include "tuple_utils.hpp"

#include <range/v3/view/repeat.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/zip.hpp>
#include <range/v3/view/zip_with.hpp>
#include <tuple>

namespace ccs::detail
{

template <typename T>
struct tuple_math;

class tuple_math_access
{
    template <typename T>
    friend class tuple_math;
};

template <typename T>
struct tuple_math {

private:
#define SHOCCS_GEN_OPERATORS(op, f)                                                      \
    template <std::derived_from<T> U, Numeric V>                                         \
        requires OutputTuple<U, V>                                                       \
    friend U& op(U& u, V v)                                                              \
    {                                                                                    \
        for_each(                                                                        \
            [v](auto&& rng) {                                                            \
                for (auto&& x : rng) x f v;                                              \
            },                                                                           \
            u);                                                                          \
                                                                                         \
        return u;                                                                        \
    }                                                                                    \
                                                                                         \
    template <std::derived_from<T> U, TupleLike V>                                       \
        requires OutputTuple<U, V>                                                       \
    friend U& op(U& u, V&& v)                                                            \
    {                                                                                    \
        for_each(                                                                        \
            [](auto&& out, auto&& in) {                                                  \
                for (auto&& [o, i] : vs::zip(out, in)) o f i;                            \
            },                                                                           \
            u,                                                                           \
            FWD(v));                                                                     \
                                                                                         \
        return u;                                                                        \
    }

    SHOCCS_GEN_OPERATORS(operator+=, +=)
    SHOCCS_GEN_OPERATORS(operator*=, *=)
    SHOCCS_GEN_OPERATORS(operator-=, -=)
    SHOCCS_GEN_OPERATORS(operator/=, /=)

#undef SHOCCS_GEN_OPERATORS

#define SHOCCS_GEN_OPERATORS(op, f)                                                      \
    template <typename U, Numeric V>                                                     \
        requires std::derived_from<std::remove_cvref_t<U>, T>                            \
    friend constexpr auto op(U&& u, V v)                                                 \
    {                                                                                    \
        return transform(                                                                \
            [v](auto&& rng) {                                                            \
                const auto sz = rs::size(rng);                                           \
                return vs::zip_with(f, FWD(rng), vs::repeat_n(v, sz));                   \
            },                                                                           \
            FWD(u));                                                                     \
    }                                                                                    \
                                                                                         \
    template <typename U, Numeric V>                                                     \
        requires std::derived_from<std::remove_cvref_t<U>, T>                            \
    friend constexpr auto op(V v, U&& u)                                                 \
    {                                                                                    \
        return transform(                                                                \
            [v](auto&& rng) {                                                            \
                const auto sz = rs::size(rng);                                           \
                return vs::zip_with(f, vs::repeat_n(v, sz), FWD(rng));                   \
            },                                                                           \
            FWD(u));                                                                     \
    }                                                                                    \
                                                                                         \
    template <typename U, typename V>                                                    \
    requires std::derived_from<std::remove_cvref_t<U>, T> &&                             \
        mp_similar<std::remove_cvref_t<U>,                                               \
                   std::remove_cvref_t<V>>::value friend constexpr auto                  \
        op(U&& u, V&& v)                                                                 \
    {                                                                                    \
        return transform(                                                                \
            [](auto&& a, auto&& b) { return vs::zip_with(f, FWD(a), FWD(b)); },          \
            FWD(u),                                                                      \
            FWD(v));                                                                     \
    }

    SHOCCS_GEN_OPERATORS(operator+, std::plus{})
    SHOCCS_GEN_OPERATORS(operator-, std::minus{})
    SHOCCS_GEN_OPERATORS(operator*, std::multiplies{})
    SHOCCS_GEN_OPERATORS(operator/, std::divides{})

#undef SHOCCS_GEN_OPERATORS
};

} // namespace ccs::detail
