#pragma once

#include "matrix_visitor.hpp"

#include "common.hpp"
#include "kokkos_types.hpp"

namespace ccs::matrix
{

class circulant : public matrix_base
{
    device_view<real*> v_d;

public:
    circulant() = default;

    circulant(integer rows, std::span<const real> coeffs);

    circulant(integer rows,
              integer row_offset,
              integer stride,
              std::span<const real> coeffs);

    integer size() const noexcept { return v_d.extent(0); }

    const device_view<real*>& coeffs_view() const { return v_d; }

    template <typename Op = eq_t>
    void operator()(std::span<const real> x,
                    std::span<real> b,
                    Op op = {}) const;

    void visit(visitor& v) const { return v.visit(*this); }

    std::span<const real> data() const { return {v_d.data(), v_d.extent(0)}; }
};

} // namespace ccs::matrix
