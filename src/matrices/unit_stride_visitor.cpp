#include "unit_stride_visitor.hpp"
#include "circulant.hpp"
#include "dense.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

namespace ccs::matrix
{

void unit_stride_visitor::adjust_size(integer rows,
                                      integer row_offset,
                                      integer cols,
                                      integer col_offset)
{
    if (row_offset + rows > nrows_in) {
        rows_out.insert(rows_out.end(), row_offset + rows - nrows_in, -1);
        nrows_in = row_offset + rows;
    }

    if (col_offset + cols > ncols_in) {
        cols_out.insert(cols_out.end(), col_offset + cols - ncols_in, -1);
        ncols_in = col_offset + cols;
    }
}

void unit_stride_visitor::add_rows(integer first, integer last)
{
    for (integer i = first; i < last; ++i) {
        if (rows_out[i] == -1) rows_out[i] = nrows_out++;
    }
}

void unit_stride_visitor::add_cols(integer first, integer last)
{
    for (integer i = first; i < last; ++i) {
        if (cols_out[i] == -1) cols_out[i] = ncols_out++;
    }
}

void unit_stride_visitor::visit(const dense& mat)
{

    assert(mat.stride() == 1);
    if (mat.stride() != 1) return;

    integer r_off = mat.row_offset();
    integer r_n = mat.rows();
    integer c_off = mat.col_offset();
    integer c_n = mat.columns();

    adjust_size(r_n, r_off, c_n, c_off);

    add_rows(r_off, r_off + r_n);
    fmt::print("rows_out: {}\n", fmt::join(rows_out, ", "));

    if (auto f = mat.flags(); is_ldd(f)) {
        add_cols(c_off + 1, c_off + c_n);
    } else if (is_rdd(f)) {
        add_cols(c_off, c_off + c_n - 1);
    } else {
        add_cols(c_off, c_off + c_n);
    }
    fmt::print("cols_out: {}\n", fmt::join(cols_out, ", "));
}

void unit_stride_visitor::visit(const circulant& mat)
{

    assert(mat.stride() == 1);
    if (mat.stride() != 1) return;

    integer r_off = mat.row_offset();
    integer r_n = mat.rows();
    integer c_off = r_off - (mat.size() / 2);
    integer c_n = mat.columns();

    adjust_size(r_n, r_off, c_n, c_off);

    add_rows(r_off, r_off + r_n);
    fmt::print("rows_out: {}\n", fmt::join(rows_out, ", "));

    add_cols(c_off, c_off + c_n);

    fmt::print("cols_out: {}\n", fmt::join(cols_out, ", "));
}

std::span<const integer> unit_stride_visitor::mapped(integer first_row,
                                                     integer rows,
                                                     integer first_col,
                                                     integer cols) const
{
    if (rows * cols > (integer)ic_.size()) ic_.resize(rows * cols);

    for (integer r = 0; r < rows; r++)
        for (integer c = 0; c < cols; c++) {
            auto r_in = first_row + r;
            auto c_in = first_col + c;
            assert(rows_out[r_in] != -1);
            assert(cols_out[c_in] != -1);
            ic_[r * cols + c] = rows_out[r_in] * ncols_out + cols_out[c_in];
        }

    return std::span(ic_.begin(), rows * cols);
}

} // namespace ccs::matrix
