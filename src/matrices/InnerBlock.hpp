#pragma once

#include "Circulant.hpp"
#include "Dense.hpp"
#include "types.hpp"

#include <range/v3/view/concat.hpp>

namespace ccs::matrix
{
// Block matrix arising from method-of-lines discretization along a line.  A full domain
// discretization requires many of these to be embedded at various starting rows.
class InnerBlock
{
    int row_start;
    Dense left_boundary;
    Circulant interior;
    Dense right_boundary;

public:
    InnerBlock() = default;

    InnerBlock(int row_start, Dense&& left, Circulant&& i, Dense&& right)
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
    template <rs::random_access_range R>
    friend constexpr auto operator*(const InnerBlock& mat, R&& rng)
    {
        auto x = rng | vs::drop(mat.row_start);
        int right_offset = mat.rows() - mat.right_boundary.columns();
        return field::Tuple{
            vs::concat(mat.left_boundary * x,
                       mat.interior * x,
                       mat.right_boundary * (x | vs::drop(right_offset)))};
    }
};
} // namespace ccs::matrix