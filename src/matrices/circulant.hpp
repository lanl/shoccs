#pragma once

#include "matrix_visitor.hpp"

#include "common.hpp"

namespace ccs::matrix
{

class circulant : public matrix_base
{
    std::span<const real> v;

public:
    circulant() = default;

    circulant(integer rows, std::span<const real> coeffs);

    circulant(integer rows,
              integer row_offset,
              integer stride,
              std::span<const real> coeffs);

    integer size() const noexcept { return v.size(); }

    template <typename Op = eq_t>
    void operator()(std::span<const real> x, std::span<real> b, Op op = {}) const;

    void visit(visitor& v) const { return v.visit(*this); }

    std::span<const real> data() const { return v; }
};

} // namespace ccs::matrix
