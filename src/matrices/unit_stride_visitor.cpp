#include "unit_stride_visitor.hpp"
#include "circulant.hpp"
#include "csr.hpp"
#include "dense.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <range/v3/view/repeat.hpp>

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

void unit_stride_visitor::add_cols(integer offset, std::span<const integer> cols)
{
    for (auto&& col : cols) {
        integer i = col + offset;
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

void unit_stride_visitor::visit(const csr& mat)
{
    const auto col_flags = (colspace_rx | colspace_ry | colspace_rz) & mat.flags();
    const auto row_flags =
        ((rowspace_rx | rowspace_ry | rowspace_rz) & mat.flags()) >> row_shift;

    // fmt::print("row/col flags {} / {}\n", row_flags, col_flags);

    const integer row_offset =
        !!row_flags *
        (base_in + !is_rx(row_flags) * (rx.size() + !is_ry(row_flags) * ry.size()));
    const integer col_offset =
        !!col_flags *
        (base_in + !is_rx(col_flags) * (rx.size() + !is_ry(col_flags) * ry.size()));

    // fmt::print("row/col offsets {} / {}\n", row_offset, col_offset);

    std::vector<bool> empty{};
    // here we make explicit use of rx=1, ry=2, rz=4.  would probably be better to do this
    // indirectly
    auto selectors = std::array<std::vector<bool>*, 5>{&empty, &rx, &ry, &empty, &rz};
    const auto& row_skip = *selectors[row_flags];
    const auto& col_skip = *selectors[col_flags];

    const bool check_row = rs::size(row_skip) > 0;
    const bool check_col = rs::size(col_skip) > 0;

    // if (check_row) fmt::print("row selector: {}\n", fmt::join(row_skip, ", "));
    // if (check_col) fmt::print("col selector: {}\n", fmt::join(col_skip, ", "));

    for (integer row = 0; row < mat.rows(); row++) {
        // skip if dirichlet row
        if (check_row && row_skip[row]) continue;

        // skip if there is no data on this row
        auto indices = mat.column_indices(row);
        // fmt::print("row {} / indices {}\n", row, indices);
        if (indices.size() == 0) continue;

        // for the "B" matrix with a field rowspace and r columnspace we want to
        // exclude points in r that are associated with a dirichlet bc
        // Note that we do not want to include rows that have only skipped
        // points in their column space so add a boolean flag which will only be true if
        // we set at least one column
        bool wrote_column = false;
        for (auto col : indices) {
            if (check_col && col_skip[col]) continue;
            wrote_column = true;
            integer i = col + col_offset;
            // fmt::print("col {}, cols_out[{}] = {}\n", col, i, cols_out[i]);
            if (cols_out[i] == -1) cols_out[i] = ncols_out++;
        }

        // handle entry for row
        if (wrote_column) {
            auto row_i = row + row_offset;
            if (rows_out[row_i] == -1) rows_out[row_i] = nrows_out++;
        }
    }

    fmt::print("csr rows_out: {}\n", fmt::join(rows_out, ", "));
    fmt::print("csr cols_out: {}\n", fmt::join(cols_out, ", "));
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
