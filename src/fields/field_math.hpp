#pragma once

#include "field_fwd.hpp"
#include "field_utils.hpp"
#include "types.hpp"
#include <concepts>
#include <functional>

namespace ccs::detail
{
template <typename T>
struct field_math;

class field_math_access
{
    template <typename T>
    friend class field_math;
};

#define SHOCCS_GEN_OPERATORS(op, f)                                                      \
    template <typename U, Numeric V>                                                     \
        requires std::derived_from<std::remove_cvref_t<U>, T>                            \
    friend constexpr auto op(U&& u, V) { return u; }                                     \
                                                                                         \
    template <typename U, Numeric V>                                                     \
        requires std::derived_from<std::remove_cvref_t<U>, T>                            \
    friend constexpr auto op(V, U&& u) { return u; }                                     \
    template <typename U, typename V>                                                    \
        requires std::derived_from<std::remove_cvref_t<U>, T>                            \
    friend constexpr auto op(U&& u, V&&) { return u; }
#undef SHOCCS_GEN_OPERATORS

template <typename T>
struct field_math {

private:
#define SHOCCS_GEN_OPERATORS(op, f)                                                      \
    template <std::derived_from<T> U, Numeric V>                                         \
        requires OutputField<U, V>                                                       \
    constexpr friend U& op(U& u, V v)                                                    \
    {                                                                                    \
        for_each([v](auto&& e) { e f v; }, u);                                           \
        return u;                                                                        \
    }                                                                                    \
                                                                                         \
    template <std::derived_from<T> U, Field V>                                           \
        requires OutputField<U, V>                                                       \
    constexpr friend U& op(U& u, V&& v)                                                  \
    {                                                                                    \
        for_each([](auto&& ui, auto&& vi) { ui f vi; }, u, FWD(v));                      \
        return u;                                                                        \
    }

    SHOCCS_GEN_OPERATORS(operator+=, +=)
    SHOCCS_GEN_OPERATORS(operator*=, *=)
    SHOCCS_GEN_OPERATORS(operator-=, -=)
    SHOCCS_GEN_OPERATORS(operator/=, /=)

#undef SHOCCS_GEN_OPERATORS

#define SHOCCS_GEN_OPERATORS(op, f)                                                      \
    template <typename U, Numeric V>                                                     \
        requires std::derived_from<T, std::remove_cvref_t<U>>                            \
    friend constexpr auto op(U&& u, V v)                                                 \
    {                                                                                    \
        return transform([v](auto&& e) { return e f v; }, FWD(u));                       \
    }                                                                                    \
                                                                                         \
    template <typename U, Numeric V>                                                     \
        requires std::derived_from<T, std::remove_cvref_t<U>>                            \
    friend constexpr auto op(V v, U&& u)                                                 \
    {                                                                                    \
        return transform([v](auto&& e) { return v f e; }, FWD(u));                       \
    }                                                                                    \
                                                                                         \
    template <typename U, Field V>                                                       \
        requires std::derived_from<T, std::remove_cvref_t<U>>                            \
    friend constexpr auto op(U&& u, V&& v)                                               \
    {                                                                                    \
        return transform([](auto&& ui, auto&& vi) { return ui f vi; }, FWD(u), FWD(v));  \
    }

    SHOCCS_GEN_OPERATORS(operator+, +)
    SHOCCS_GEN_OPERATORS(operator*, *)
    SHOCCS_GEN_OPERATORS(operator-, -)
    SHOCCS_GEN_OPERATORS(operator/, /)

#undef SHOCCS_GEN_OPERATORS
};

} // namespace ccs::detail
