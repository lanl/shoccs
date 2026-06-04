#pragma once

#include "fields/field_registry.hpp"

#include <cassert>

namespace ccs
{

// Zero all allocated buffers in a slot.
inline void slot_zero(sim_registry& reg, field_ref ref)
{
    assert(ref.n_vectors == 0 && "slot_ops: vector support not yet implemented");
    for (int s = 0; s < ref.n_scalars; ++s) {
        scalar_handle sh{s * sim_registry::layout_type::scalar_stride};
        for (auto bh : sh.all())
            Kokkos::deep_copy(reg.view(ref, bh), 0.0);
    }
    Kokkos::fence();
}

// dst[i] = src[i] + coeff * rhs[i]  for all allocated buffers.
inline void slot_assign_lc(sim_registry& reg, field_ref dst,
                            field_ref src, real coeff, field_ref rhs)
{
    assert(dst.n_vectors == 0 && "slot_ops: vector support not yet implemented");
    for (int s = 0; s < dst.n_scalars; ++s) {
        scalar_handle sh{s * sim_registry::layout_type::scalar_stride};
        for (auto bh : sh.all()) {
            int n = reg.size(dst, bh);
            real* d = reg.data(dst, bh);
            const real* s0 = reg.data(src, bh);
            const real* r = reg.data(rhs, bh);
            Kokkos::parallel_for(
                Kokkos::RangePolicy<execution_space>(0, n),
                KOKKOS_LAMBDA(int i) { d[i] = s0[i] + coeff * r[i]; });
        }
    }
    Kokkos::fence();
}

// dst[i] += coeff * src[i]  for all allocated buffers.
inline void slot_accumulate(sim_registry& reg, field_ref dst,
                             real coeff, field_ref src)
{
    assert(dst.n_vectors == 0 && "slot_ops: vector support not yet implemented");
    for (int s = 0; s < dst.n_scalars; ++s) {
        scalar_handle sh{s * sim_registry::layout_type::scalar_stride};
        for (auto bh : sh.all()) {
            int n = reg.size(dst, bh);
            real* d = reg.data(dst, bh);
            const real* r = reg.data(src, bh);
            Kokkos::parallel_for(
                Kokkos::RangePolicy<execution_space>(0, n),
                KOKKOS_LAMBDA(int i) { d[i] += coeff * r[i]; });
        }
    }
    Kokkos::fence();
}

} // namespace ccs
