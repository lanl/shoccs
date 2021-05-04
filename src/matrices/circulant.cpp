#include "circulant.hpp"

#include <cassert>

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/numeric/inner_product.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/sliding.hpp>
#include <range/v3/view/stride.hpp>
#include <range/v3/view/zip.hpp>
#include <range/v3/view/zip_with.hpp>

namespace ccs::matrix
{

circulant::circulant(integer rows, std::span<const real> coeffs)
    : matrix_base{rows, rows + (integer)coeffs.size() - 1, (integer)coeffs.size() / 2},
      v{coeffs}
{
}

circulant::circulant(integer rows,
                     integer row_offset,
                     integer stride,
                     std::span<const real> coeffs)
    : matrix_base{rows, rows + (integer)coeffs.size() - 1, row_offset, -1, stride},
      v{coeffs}
{
}

template <typename Op>
void circulant::operator()(std::span<const real> x, std::span<real> b, Op op) const
{
    assert(row_offset() >= stride() * (size() / 2));
    assert((integer)b.size() >= row_offset() + rows() * stride());
    assert((integer)x.size() >= row_offset() + (rows() + (size() / 2) - 1) * stride());
    // move input and output spans to correct position
    const auto st = stride();
    x = x.subspan(row_offset() - st * (size() / 2));
    b = b.subspan(row_offset());

    if (st == 1) {
        auto rng =
            vs::zip_with([](auto&& a, auto&& b) { return rs::inner_product(a, b, 0.0); },
                         vs::repeat_n(v, rows()),
                         x | vs::sliding(size()));

        // rs::copy(rng, rs::begin(b));
        for (auto&& [y, z] : vs::zip(b, rng)) op(y, z);
    } else {
        auto in = x | vs::stride(st);
        auto out = b | vs::stride(st);

        auto rng =
            vs::zip_with([](auto&& a, auto&& b) { return rs::inner_product(a, b, 0.0); },
                         vs::repeat_n(v, rows()),
                         in | vs::sliding(size()));

        // rs::copy(rng, rs::begin(out));
        for (auto&& [y, z] : vs::zip(out, rng)) op(y, z);
    }
}

template void
circulant::operator()<eq_t>(std::span<const real>, std::span<real>, eq_t) const;
template void
circulant::operator()<plus_eq_t>(std::span<const real>, std::span<real>, plus_eq_t) const;

} // namespace ccs::matrix
