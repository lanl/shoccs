#pragma once

#include "common.hpp"
#include <vector>

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/take.hpp>

namespace ccs::matrix
{

// Simple contiguous storage for dense matrix with lazy operators
class dense : public matrix_base
{
    std::vector<real> v;

public:
    dense() = default;

    template <rs::input_range R>
    dense(integer rows, integer columns, R&& rng)
        : matrix_base{rows, columns}, v(rows * columns)
    {
        rs::copy(rng | vs::take(v.size()), v.begin());
    }

    template <rs::input_range R>
    dense(integer rows,
          integer columns,
          integer row_offset,
          integer col_offset,
          integer stride,
          R&& rng)
        : matrix_base{rows, columns, row_offset, col_offset, stride}, v(rows * columns)
    {
        rs::copy(rng | vs::take(v.size()), v.begin());
    }

    auto size() const noexcept { return v.size(); }

    template <typename Op = eq_t>
    void operator()(std::span<const real> x, std::span<real> b, Op op = {}) const;
};
} // namespace ccs::matrix
