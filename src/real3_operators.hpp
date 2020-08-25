#pragma once
#include "types.hpp"

namespace ccs
{

constexpr real3 operator+(const real3& a, const real3& b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

constexpr real3 operator-(const real3& a, const real3& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

constexpr real dot(const real3& a, const real3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}



} // namespace shocss