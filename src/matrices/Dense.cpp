#include "Dense.hpp"
// range includes
#include <range/v3/algorithm/copy.hpp>
#include <range/v3/numeric/inner_product.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/chunk.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/stride.hpp>
#include <range/v3/view/zip_with.hpp>

#include <cassert>

namespace ccs::matrix
{

void Dense::operator()(std::span<const real> x, std::span<real> b) const
{
    // sanity checks
    assert((integer)b.size() >= row_offset() + (rows() - 1) * stride());
    assert((integer)x.size() >= col_offset() + (columns() - 1) * stride());
    // move input and output spans to correct position
    x = x.subspan(col_offset());
    b = b.subspan(row_offset());
    const auto st = stride();

    if (st == 1) {
        auto rng =
            vs::zip_with([](auto&& a, auto&& b) { return rs::inner_product(a, b, 0.0); },
                         vs::chunk(v, columns()),
                         vs::repeat_n(x, rows()));
        rs::copy(rng, rs::begin(b));
    } else {
        auto in = x | vs::stride(st);
        auto out = b | vs::stride(st);

        auto rng =
            vs::zip_with([](auto&& a, auto&& b) { return rs::inner_product(a, b, 0.0); },
                         vs::chunk(v, columns()),
                         vs::repeat_n(in, rows()));
        rs::copy(rng, rs::begin(out));
    }
}

} // namespace ccs::matrix