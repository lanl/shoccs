#pragma once
#include "types.hpp"
#include <cmath>

namespace ccs
{

#define gen_operators(op, acc)                                                           \
    constexpr real3 op(const real3& a, const real3& b)                                   \
    {                                                                                    \
        return {a[0] acc b[0], a[1] acc b[1], a[2] acc b[2]};                            \
    }                                                                                    \
    constexpr real3 op(const real3& a, Numeric auto n)                                   \
    {                                                                                    \
        return {a[0] acc n, a[1] acc n, a[2] acc n};                                     \
    }                                                                                    \
    constexpr real3 op(Numeric auto n, const real3& a)                                   \
    {                                                                                    \
        return {n acc a[0], n acc a[1], n acc a[2]};                                     \
    }

// clang-format off
gen_operators(operator*, *)
gen_operators(operator/, /)
gen_operators(operator+, +)
gen_operators(operator-, -)
// clang-format on

#undef gen_operators

        constexpr real dot(const real3& a, const real3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

real length(const real3& a) { return std::sqrt(dot(a, a)); }

} // namespace ccs