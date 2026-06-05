#pragma once

#include "common.hpp"
#include "kokkos_types.hpp"
#include "matrix_visitor.hpp"

#include <algorithm>
#include <ranges>

namespace ccs::matrix
{

// Simple contiguous storage for dense matrix with lazy operators
class dense : public matrix_base
{
    device_view<real*> v_d;
    flag f;

public:
    dense() = default;

    template <std::ranges::input_range R>
    dense(integer rows, integer columns, R&& rng, flag boundary = 0)
        : matrix_base{rows, columns}, v_d("dense_coeffs", rows * columns), f{boundary}
    {
        std::vector<real> tmp(rows * columns);
        std::ranges::copy(rng | std::views::take(tmp.size()), tmp.begin());
        auto h_view = Kokkos::View<const real*, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            tmp.data(), tmp.size());
        Kokkos::deep_copy(v_d, h_view);
    }

    template <std::ranges::input_range R>
    dense(integer rows,
          integer columns,
          integer row_offset,
          integer col_offset,
          integer stride,
          R&& rng,
          flag boundary = 0)
        : matrix_base{rows, columns, row_offset, col_offset, stride},
          v_d("dense_coeffs", rows * columns),
          f{boundary}
    {
        std::vector<real> tmp(rows * columns);
        std::ranges::copy(rng | std::views::take(tmp.size()), tmp.begin());
        auto h_view = Kokkos::View<const real*, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            tmp.data(), tmp.size());
        Kokkos::deep_copy(v_d, h_view);
    }

    integer size() const noexcept { return v_d.extent(0); }

    const device_view<real*>& coeffs_view() const { return v_d; }

    template <typename Op = eq_t>
    void operator()(std::span<const real> x,
                    std::span<real> b,
                    Op op = {}) const;

    std::span<const real> data() const { return {v_d.data(), v_d.extent(0)}; }
    flag flags() const { return f; }
    void flags(flag f_) { f = f_; }
    void visit(visitor& v) const { v.visit(*this); };
};
} // namespace ccs::matrix
