#pragma once

#include "fields/Tuple.hpp"
#include "types.hpp"

#include <cassert>
#include <span>

#include <range/v3/numeric/inner_product.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/sliding.hpp>
#include <range/v3/view/zip_with.hpp>

namespace ccs::matrix
{

class Circulant
{
    int offset_; // don't support negative for periodic domains yet
    int rows_;
    std::span<const real> v;

public:
    Circulant() = default;

    Circulant(int offset, int rows, std::span<const real> coeffs)
        : offset_{offset}, rows_{rows}, v{coeffs}
    {
        assert(offset_ > -1);
    }

    size_t size() const noexcept { return v.size(); }

    constexpr int rows() const { return rows_; }
    constexpr int offset() const { return offset_; }

private:
    template <ranges::random_access_range R>
    friend constexpr auto operator*(const Circulant& mat, R&& rng)
    {
        assert(rng.size() >= static_cast<unsigned>(mat.rows()));
        assert(mat.size() > 0);

        return field::Tuple{ranges::views::zip_with(
            [](auto&& a, auto&& b) { return ranges::inner_product(a, b, 0.0); },
            ranges::views::repeat_n(mat.v, mat.rows()),
            rng | ranges::views::drop(mat.offset()) |
                ranges::views::sliding(mat.size()))};
    }
};

} // namespace ccs::matrix