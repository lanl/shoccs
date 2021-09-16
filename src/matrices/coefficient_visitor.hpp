#pragma once

#include "fields/tuple_fwd.hpp"
#include "matrix_visitor.hpp"
#include "unit_stride_visitor.hpp"

namespace ccs::matrix
{
class coefficient_visitor : public visitor
{
    unit_stride_visitor v;

    std::vector<real> m; // full matrix of coefficients

public:
    coefficient_visitor() = default;

    coefficient_visitor(unit_stride_visitor&& v_) : v{MOVE(v_)}, m(v.mapped_size()) {}

    void visit(const dense&) override;
    void visit(const circulant&) override;
    void visit(const csr&) override;

    std::span<const real> matrix() const { return m; }

    const real* data() const { return m.data(); }
    real* data() { return m.data(); }
};
} // namespace ccs::matrix
