#pragma once

#include "circulant.hpp"
#include "dense.hpp"
#include "fields/result_field.hpp"
#include "types.hpp"

#include <range/v3/view/concat.hpp>

namespace ccs::matrix
{
// Block matrix arising from method-of-lines discretization along a line.  A full domain
// discretization requires many of these to be embedded at various starting rows.
class inner_block
{
    int row_start;
    dense left_boundary;
    circulant interior;
    dense right_boundary;

public:
    inner_block() = default;

    inner_block(int row_start, dense&& left, circulant&& i, dense&& right)
        : row_start{row_start},
          left_boundary{std::move(left)},
          interior{std::move(i)},
          right_boundary{std::move(right)}
    {
    }

    constexpr int first_row() const { return row_start; }

    constexpr int rows() const
    {
        return left_boundary.rows() + interior.rows() + right_boundary.rows();
    }

private:
    template <ranges::random_access_range R>
    friend constexpr auto operator*(const inner_block& mat, R&& rng)
    {
        namespace vs = ranges::views;
        auto x = rng | vs::drop(mat.row_start);
        int right_offset = mat.rows() - mat.right_boundary.columns();
        return result_range{vs::concat(
            (mat.left_boundary * x).range(),
            (mat.interior * x).range(),
            (mat.right_boundary * (x | vs::drop(right_offset))).range())};
    }
};
} // namespace ccs::matrix