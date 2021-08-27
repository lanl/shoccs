#pragma once

#include "matrix_visitor.hpp"

#include "circulant.hpp"
#include "dense.hpp"

namespace ccs::matrix
{
// Block matrix arising from method-of-lines discretization along a line.  A full domain
// discretization requires many of these to be embedded at various starting rows.
class inner_block : public matrix_base
{
    dense left_boundary;
    circulant interior;
    dense right_boundary;

public:
    inner_block() = default;

    inner_block(dense&& left, circulant&& i, dense&& right);

    inner_block(integer columns,
                integer row_offset,
                integer col_offset,
                integer stride,
                dense&& left,
                circulant&& i,
                dense&& right);

    // don't allow changes to these after construction
    using matrix_base::col_offset;
    using matrix_base::row_offset;
    using matrix_base::stride;

    matrix_base& row_offset(integer) = delete;
    matrix_base& col_offset(integer) = delete;
    matrix_base& stride(integer) = delete;

    template <typename Op = eq_t>
    void operator()(std::span<const real> x, std::span<real> b, Op op = {}) const;

    void visit(visitor& v) const
    {
        v.visit(left_boundary);
        v.visit(interior);
        v.visit(right_boundary);
    }
};
} // namespace ccs::matrix
