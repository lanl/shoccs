#pragma once

#include "kokkos_types.hpp"
#include "types.hpp"

namespace ccs
{

// ---------------------------------------------------------------------------
// scalar_span / scalar_view — lightweight structs for 4-component field access.
// ---------------------------------------------------------------------------

struct scalar_span {
    std::span<real> D{}, Rx{}, Ry{}, Rz{};

    scalar_span() = default;
    scalar_span(std::span<real> d, std::span<real> rx,
                std::span<real> ry, std::span<real> rz)
        : D(d), Rx(rx), Ry(ry), Rz(rz) {}

    // Broadcast fill: du = 0.
    template <typename T>
        requires std::is_arithmetic_v<T>
    scalar_span& operator=(T val)
    {
        const real v = static_cast<real>(val);
        real* d_ptr = D.data();
        real* rx_ptr = Rx.data();
        real* ry_ptr = Ry.data();
        real* rz_ptr = Rz.data();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, static_cast<int>(D.size())),
            KOKKOS_LAMBDA(int i) { d_ptr[i] = v; });
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, static_cast<int>(Rx.size())),
            KOKKOS_LAMBDA(int i) { rx_ptr[i] = v; });
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, static_cast<int>(Ry.size())),
            KOKKOS_LAMBDA(int i) { ry_ptr[i] = v; });
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, static_cast<int>(Rz.size())),
            KOKKOS_LAMBDA(int i) { rz_ptr[i] = v; });
        Kokkos::fence();
        return *this;
    }

    // Functional assignment: u_rhs = lap(u, nu).
    template <std::invocable<scalar_span&> Fn>
    scalar_span& operator=(Fn&& fn)
    {
        FWD(fn)(*this);
        return *this;
    }
};

struct scalar_view {
    std::span<const real> D{}, Rx{}, Ry{}, Rz{};

    scalar_view() = default;
    scalar_view(std::span<const real> d, std::span<const real> rx,
                std::span<const real> ry, std::span<const real> rz)
        : D(d), Rx(rx), Ry(ry), Rz(rz) {}

    // Converting constructor from scalar_span.
    scalar_view(const scalar_span& s)  // NOLINT(google-explicit-constructor)
        : D(s.D), Rx(s.Rx), Ry(s.Ry), Rz(s.Rz) {}
};

} // namespace ccs
