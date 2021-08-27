#pragma once

#include "matrix_visitor.hpp"

namespace ccs::matrix
{
class unit_stride_visitor : public visitor
{
    using ivec = std::vector<integer>;

    integer nrows_in, ncols_in, nrows_out, ncols_out;

    ivec rows_out, cols_out;

    mutable std::vector<integer> ic_;

    void adjust_size(integer rows, integer row_offset, integer cols, integer col_offset);
    void add_rows(integer, integer);
    void add_cols(integer, integer);

public:
    unit_stride_visitor() = default;

    unit_stride_visitor(integer rows, integer columns)
        : nrows_in(rows),
          ncols_in(columns),
          nrows_out{},
          ncols_out{},
          rows_out(nrows_in, -1),
          cols_out(ncols_in, -1)
    {
    }

    void visit(const dense&) override;
    void visit(const circulant&) override;

    std::span<const integer>
    mapped(integer first_row, integer rows, integer first_col, integer cols) const;
};

} // namespace ccs::matrix
