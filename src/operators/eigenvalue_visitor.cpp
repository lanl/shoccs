#include "eigenvalue_visitor.hpp"
#include "gradient.hpp"

#include <cstdint>
#include <lapack.hh>


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

    // canabalize the unit_stride_visitor to construct our coefficient_visitor
    v = matrix::coefficient_visitor{MOVE(u)};
    d.visit(v);

    // compute eigenvalues given our now dense matrix
    real vl, vr;
    int64_t lda = rows;
    int64_t n = rows;
    // according to the docs I should be able to leave ldvl and ldvr as 1 since
    // jobl=jobr='N', but then I get an error in the dgeev_work routine.
    int64_t ldvl = n;
    int64_t ldvr = n;

    std::vector<std::complex<double>> W(eigs_real.size());

    auto ret = lapack::geev(lapack::Job::NoVec,
                            lapack::Job::NoVec,
                            n,
                            v.data(),
                            lda,
                            W.data(),
                            &vl,
                            ldvl,
                            &vr,
                            ldvr);
    assert(ret == 0);

    for (int i = 0; i < (int)W.size(); i++) {
        eigs_real[i] = W[i].real();
        eigs_imag[i] = W[i].imag();
    }
}

std::span<const real> eigenvalue_visitor::eigenvalues_real() const { return eigs_real; }
std::span<const real> eigenvalue_visitor::eigenvalues_imag() const { return eigs_imag; }
} // namespace ccs
