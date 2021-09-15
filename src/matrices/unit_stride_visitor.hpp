#pragma once

#include "common.hpp"
#include "fields/tuple_fwd.hpp"
#include "matrix_visitor.hpp"

namespace ccs::matrix
{
class unit_stride_visitor : public visitor
{
    using ivec = std::vector<integer>;

    integer nrows_in, ncols_in, nrows_out, ncols_out, base_in;

    ivec rows_out, cols_out;

    // flags to indicate which indices are being skipped
    std::vector<bool> rx, ry, rz;

    mutable std::vector<integer> ic_;

    void adjust_size(integer rows, integer row_offset, integer cols, integer col_offset);
    void add_rows(integer, integer);
    void add_cols(integer, integer);
    void add_cols(integer, std::span<const integer>);

    // I don't think these routines make a lot of sense as memebers of this class
    std::array<flag, 2> csr_flags(const csr&) const;
    std::array<integer, 2> csr_offsets(const csr&) const;

public:
    unit_stride_visitor() = default;

    unit_stride_visitor(integer rows, integer columns)
        : nrows_in(rows),
          ncols_in(columns),
          nrows_out{},
          ncols_out{},
          rows_out(nrows_in, -1),
          cols_out(ncols_in, -1),
          rx{},
          ry{},
          rz{}
    {
    }

    template <Range Rx, Range Ry, Range Rz>
    unit_stride_visitor(int3 nxyz, Rx&& rx, Ry&& ry, Rz&& rz)
        : nrows_in(nxyz[0] * nxyz[1] * nxyz[2] + rs::size(rx) + rs::size(ry) +
                   rs::size(rz)),
          ncols_in(nrows_in),
          nrows_out{},
          ncols_out{},
          base_in{nxyz[0] * nxyz[1] * nxyz[2]},
          rows_out(nrows_in, -1),
          cols_out(ncols_in, -1),
          rx(rs::begin(rx), rs::end(rx)),
          ry(rs::begin(ry), rs::end(ry)),
          rz(rs::begin(rz), rs::end(rz))
    {
    }

    void visit(const dense&) override;
    void visit(const circulant&) override;
    void visit(const csr&) override;

    integer mapped_size() const { return nrows_out * ncols_out; }

    // for dense and circulant matrices
    std::span<const integer>
    mapped(integer first_row, integer rows, integer first_col, integer cols) const;

    // for csr matrices - offset calculations can be handled internally for ease of use
    std::span<const integer> mapped(integer row, const csr&) const;
};

} // namespace ccs::matrix
