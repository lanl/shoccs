#include "circulant.hpp"

#include "kokkos_types.hpp"

#include <cassert>
#include <numeric>

namespace ccs::matrix
{

circulant::circulant(integer rows, std::span<const real> coeffs)
    : matrix_base{rows, rows + (integer)coeffs.size() - 1, (integer)coeffs.size() / 2},
      v_d("circulant_coeffs", coeffs.size())
{
    auto h_view = Kokkos::View<const real*, Kokkos::HostSpace,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        coeffs.data(), coeffs.size());
    Kokkos::deep_copy(v_d, h_view);
}

circulant::circulant(integer rows,
                     integer row_offset,
                     integer stride,
                     std::span<const real> coeffs)
    : matrix_base{rows, rows + (integer)coeffs.size() - 1, row_offset, -1, stride},
      v_d("circulant_coeffs", coeffs.size())
{
    auto h_view = Kokkos::View<const real*, Kokkos::HostSpace,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        coeffs.data(), coeffs.size());
    Kokkos::deep_copy(v_d, h_view);
}

template <typename Op>
void circulant::operator()(std::span<const real> x, std::span<real> b, Op op) const
{
    assert(row_offset() >= stride() * (size() / 2));
    assert((integer)b.size() >= row_offset() + rows() * stride());
    assert((integer)x.size() >= row_offset() + (rows() + (size() / 2) - 1) * stride());
    // move input and output spans to correct position
    const auto st = stride();
    x = x.subspan(row_offset() - st * (size() / 2));
    b = b.subspan(row_offset());

    const auto nr = rows();
    const auto* vp = v_d.data();
    const auto vs = static_cast<integer>(v_d.extent(0));
    const auto* xp = x.data();
    auto* bp = b.data();

    if (st == 1) {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, nr),
            [=](int i) {
                auto dot = std::inner_product(vp, vp + vs, xp + i, 0.0);
                op(bp[i], dot);
            });
    } else {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, nr),
            [=](int i) {
                real dot = 0.0;
                for (integer j = 0; j < vs; j++)
                    dot += vp[j] * xp[(i + j) * st];
                op(bp[i * st], dot);
            });
    }
}

template void
circulant::operator()<eq_t>(std::span<const real>, std::span<real>, eq_t) const;
template void
circulant::operator()<plus_eq_t>(std::span<const real>, std::span<real>, plus_eq_t) const;

} // namespace ccs::matrix
