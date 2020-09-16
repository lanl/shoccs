#pragma once

#include "circulant.hpp"
#include "dense.hpp"
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
        auto x = rng | ranges::views::drop(mat.row_start);
        int right_offset = mat.rows() - mat.right_boundary.columns();
        return ranges::views::concat(mat.left_boundary * x,
                                     mat.interior * x,
                                     mat.right_boundary *
                                         (x | ranges::views::drop(right_offset)));
    }
};
} // namespace ccs::matrix