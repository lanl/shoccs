#pragma once

#include "range_operators.hpp"
#include "types.hpp"
#include <vector>


// range includes
#include <range/v3/algorithm/copy.hpp>
#include <range/v3/numeric/inner_product.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/chunk.hpp>
#include <range/v3/view/repeat_n.hpp>

namespace ccs::matrix
{

// Simple contiguous storage for dense matrix with lazy operators
class dense
{
    int rows_;
    int columns_;
    std::vector<real> v;

public:
    dense() = default;

    template <ranges::input_range R>
    dense(int rows, int columns, R&& rng)
        : rows_{rows}, columns_{columns}, v(rows * columns)
    {
        ranges::copy(rng | ranges::view::take(v.size()), v.begin());
    }

    size_t size() const noexcept { return v.size(); }

    constexpr int rows() const { return rows_; }
    constexpr int columns() const { return columns_; }

private:
    template <ranges::random_access_range R>
    friend constexpr auto operator*(const dense& mat, R&& rng)
    {
        assert(rng.size() >= static_cast<unsigned>(mat.columns()));

        return ranges::views::zip_with(
            [](auto&& a, auto&& b) { return ranges::inner_product(a, b, 0.0); },
            ranges::views::chunk(mat.v, mat.columns()),
            ranges::views::repeat_n(rng, mat.rows()));
    }
};
} // namespace ccs::matrix