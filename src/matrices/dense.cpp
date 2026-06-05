#include "dense.hpp"

#include <cassert>
#include <numeric>

namespace ccs::matrix
{

template <typename Op>
void dense::operator()(std::span<const real> x, std::span<real> b, Op op) const
{
    // sanity checks
    assert((integer)b.size() >= row_offset() + (rows() - 1) * stride());
    assert((integer)x.size() >= col_offset() + (columns() - 1) * stride());
    // move input and output spans to correct position
    x = x.subspan(col_offset());
    b = b.subspan(row_offset());
    const auto st = stride();
    const auto* vp = v_d.data();
    const auto nc = columns();

    if (st == 1) {
        for (integer i = 0; i < rows(); i++) {
            auto dot = std::inner_product(
                vp + i * nc, vp + (i + 1) * nc,
                x.data(), 0.0);
            op(b[i], dot);
        }
    } else {
        for (integer i = 0; i < rows(); i++) {
            real dot = 0.0;
            for (integer j = 0; j < nc; j++)
                dot += vp[i * nc + j] * x[j * st];
            op(b[i * st], dot);
        }
    }
}

//
template void dense::operator()<eq_t>(std::span<const real>, std::span<real>, eq_t) const;

template void
dense::operator()<plus_eq_t>(std::span<const real>, std::span<real>, plus_eq_t) const;

} // namespace ccs::matrix
