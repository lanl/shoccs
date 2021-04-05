#pragma once

#include "Circulant.hpp"
#include "Dense.hpp"

namespace ccs::matrix
{
// Block matrix arising from method-of-lines discretization along a line.  A full domain
// discretization requires many of these to be embedded at various starting rows.
class InnerBlock : public Common
{
    Dense left_boundary;
    Circulant interior;
    Dense right_boundary;

public:
    InnerBlock() = default;

    InnerBlock(Dense&& left, Circulant&& i, Dense&& right);

    InnerBlock(integer columns,
               integer row_offset,
               integer col_offset,
               integer stride,
               Dense&& left,
               Circulant&& i,
               Dense&& right);

    // don't allow changes to these after construction
    using Common::col_offset;
    using Common::row_offset;
    using Common::stride;

    Common& row_offset(integer) = delete;
    Common& col_offset(integer) = delete;
    Common& stride(integer) = delete;

    void operator()(std::span<const real> x, std::span<real> b) const;
};
} // namespace ccs::matrix