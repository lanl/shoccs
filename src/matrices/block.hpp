#pragma once

#include "circulant.hpp"
#include "dense.hpp"
#include "types.hpp"

#include <range/v3/view/concat.hpp>

namespace ccs::matrix
{
// 1D block matrix arising from method-of-lines discretization.
class block
{
    dense left_boundary;
    circulant interior;
    dense right_boundary;

public:
    block() = default;

    block(dense&& left, circulant&& i, dense&& right)
        : left_boundary{std::move(left)},
          interior{std::move(i)},
          right_boundary{std::move(right)}
    {
    }

    constexpr int rows() const
    {
        return left_boundary.rows() + interior.rows() + right_boundary.rows();
    }

private:
    template <ranges::random_access_range R>
    friend constexpr auto operator*(const block& mat, R&& rng)
    {
        int right_offset = mat.rows() - mat.right_boundary.columns();
        return ranges::views::concat(mat.left_boundary * rng,
                                     mat.interior * rng,
                                     mat.right_boundary * (rng | ranges::views::drop(right_offset)));
    }
};
} // namespace ccs::matrix