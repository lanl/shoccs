#include "eigenvalue_visitor.hpp"
#include "gradient.hpp"

#include <mkl.h>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <range/v3/view/chunk.hpp>
#include <range/v3/view/enumerate.hpp>

namespace ccs
{
void eigenvalue_visitor::visit(const derivative& d)
{
    // build a map of where to put the coefficients by first utilizing the
    // unit_stride_visitor
    d.visit(u);

    auto [rows, cols] = u.mapped_dims();
    assert(rows == cols);
    eigs_real.resize(rows);
    eigs_imag.resize(rows);

    fmt::print("Matrix sz: [{} x {}]\n", rows, cols);

    // canabalize the unit_stride_visitor to construct our coefficient_visitor
    v = matrix::coefficient_visitor{MOVE(u)};
    d.visit(v);
    for (auto&& [i, rng] : vs::enumerate(v.matrix() | vs::chunk(cols))) {
        fmt::print("{:2d}|\t{}\n", i, fmt::join(rng, ", "));
    }

    // compute eigenvalues given our now dense matrix
    real vl, vr;
    MKL_INT lda = rows;
    MKL_INT n = rows;
    MKL_INT ldvl = n;
    MKL_INT ldvr = n;

    auto ret = LAPACKE_dgeev(LAPACK_ROW_MAJOR,
                             'N',
                             'N',
                             n,
                             v.data(),
                             lda,
                             eigs_real.data(),
                             eigs_imag.data(),
                             &vl,
                             ldvl,
                             &vr,
                             ldvr);
    assert(ret == 0);
}

std::span<const real> eigenvalue_visitor::eigenvalues_real() const { return eigs_real; }
std::span<const real> eigenvalue_visitor::eigenvalues_imag() const { return eigs_imag; }
} // namespace ccs
