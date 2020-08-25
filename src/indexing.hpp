#pragma once

// Orgainization of slow/fast indices for things written as (i, j, k) tuples
// Generally, operators meant to be applied in a particular direction are applied
// to data for which that direction is contiguous and the other two are have non-unit
// strides.  The direction with the biggest strides is `slow` while the direction
// with the smaller strides is `fast`
namespace ccs::index {

template <int I>
struct dir;

template <>
struct dir<0> {
    static constexpr int slow = 1;
    static constexpr int fast = 2;
};

template <>
struct dir<1> {
    static constexpr int slow = 0;
    static constexpr int fast = 2;
};

template <>
struct dir<2> {
    static constexpr int slow = 0;
    static constexpr int fast = 1;
};
}

