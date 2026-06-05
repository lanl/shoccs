#include "coefficient_visitor.hpp"
#include "circulant.hpp"
#include "csr.hpp"
#include "dense.hpp"

#include <cassert>

namespace ccs::matrix
{
void coefficient_visitor::visit(const dense& mat)
{
    assert(m.size() > 0);
    integer r_off = mat.row_offset();
    integer r_n = mat.rows();
    integer c_off = mat.col_offset();
    integer c_n = mat.columns();

    auto ind = v.mapped(r_off, r_n, c_off, c_n);
    auto d = mat.data();

    if (auto f = mat.flags(); is_ldd(f)) {
        for (integer r = 0; r < r_n; r++)
            for (integer c = 1; c < c_n; c++)
                m[ind[r * c_n + c]] = d[r * c_n + c];
    } else if (is_rdd(f)) {
        for (integer r = 0; r < r_n; r++)
            for (integer c = 0; c < c_n - 1; c++)
                m[ind[r * c_n + c]] = d[r * c_n + c];
    } else {
        for (integer i = 0; i < (integer)ind.size(); i++)
            m[ind[i]] = d[i];
    }
}

void coefficient_visitor::visit(const circulant& mat)
{
    assert(m.size() > 0);

    integer r_off = mat.row_offset();
    integer r_n = mat.rows();
    integer c_off = r_off - (mat.size() / 2);

    for (integer row = 0; row < r_n; row++) {
        auto mapped = v.mapped(row + r_off, 1, c_off + row, mat.size());
        auto data = mat.data();
        for (integer k = 0; k < (integer)mapped.size(); k++)
            m[mapped[k]] = data[k];
    }
}

void coefficient_visitor::visit(const csr& mat)
{
    assert(m.size() > 0);

    for (integer row = 0; row < mat.rows(); row++) {
        auto mapped = v.mapped(row, mat);
        auto coeffs = mat.column_coefficients(row);
        for (integer k = 0; k < (integer)mapped.size(); k++)
            if (mapped[k] != -1) m[mapped[k]] = coeffs[k];
    }
}
} // namespace ccs::matrix
