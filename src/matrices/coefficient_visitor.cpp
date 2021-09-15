#include "coefficient_visitor.hpp"
#include "circulant.hpp"
#include "csr.hpp"
#include "dense.hpp"

#include <range/v3/view/chunk.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/zip.hpp>

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
        auto t = vs::chunk(c_n) | vs::for_each(vs::drop(1));
        for (auto&& [i, x] : vs::zip(ind | t, d | t)) { m[i] = x; }
    } else if (is_rdd(f)) {
        auto t = vs::chunk(c_n) | vs::for_each(vs::take(c_n - 1));
        for (auto&& [i, x] : vs::zip(ind | t, d | t)) m[i] = x;
    } else {
        for (auto&& [i, x] : vs::zip(ind, d)) m[i] = x;
    }
}

void coefficient_visitor::visit(const circulant& mat)
{
    assert(m.size() > 0);

    integer r_off = mat.row_offset();
    integer r_n = mat.rows();
    integer c_off = r_off - (mat.size() / 2);

    for (integer row = 0; row < r_n; row++) {
        for (auto&& [i, x] :
             vs::zip(v.mapped(row + r_off, 1, c_off + row, mat.size()), mat.data()))
            m[i] = x;
    }
}

void coefficient_visitor::visit(const csr& mat)
{
    assert(m.size() > 0);

    for (integer row = 0; row < mat.rows(); row++) {
        for (auto&& [i, x] : vs::zip(v.mapped(row, mat), mat.column_coefficients(row)))
            if (i != -1) m[i] = x;
    }
}
} // namespace ccs::matrix
