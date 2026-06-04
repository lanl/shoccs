#pragma once

#include "derivative.hpp"
#include "fields/scalar.hpp"
#include "operator_visitor.hpp"

#include <Kokkos_Graph.hpp>

namespace ccs
{

class gradient
{
    derivative dx;
    derivative dy;
    derivative dz;
    index_extents ex;

public:
    gradient() = default;

    gradient(const mesh&,
             const stencil&,
             const bcs::Grid&,
             const bcs::Object&,
             const logs& = {});

    std::function<void(scalar_span, scalar_span, scalar_span)> operator()(scalar_view) const;

    void visit(operator_visitor& v) const { return v.visit(dx); }

    // Add gradient nodes to an existing graph. Zeros du_x/du_y/du_z, then
    // chains dx/dy/dz in parallel (independent outputs), returns when_all.
    template <typename NodeT>
    auto add_graph_nodes(NodeT parent, scalar_view u,
                         scalar_span du_x, scalar_span du_y, scalar_span du_z) const
    {
        using rp_t = Kokkos::RangePolicy<execution_space>;

        // Extract pointers and sizes for all 12 buffers
        real* dux_d = du_x.D.data();
        real* dux_rx = du_x.Rx.data();
        real* dux_ry = du_x.Ry.data();
        real* dux_rz = du_x.Rz.data();
        const int n_dux_d = static_cast<int>(du_x.D.size());
        const int n_dux_rx = static_cast<int>(du_x.Rx.size());
        const int n_dux_ry = static_cast<int>(du_x.Ry.size());
        const int n_dux_rz = static_cast<int>(du_x.Rz.size());

        real* duy_d = du_y.D.data();
        real* duy_rx = du_y.Rx.data();
        real* duy_ry = du_y.Ry.data();
        real* duy_rz = du_y.Rz.data();
        const int n_duy_d = static_cast<int>(du_y.D.size());
        const int n_duy_rx = static_cast<int>(du_y.Rx.size());
        const int n_duy_ry = static_cast<int>(du_y.Ry.size());
        const int n_duy_rz = static_cast<int>(du_y.Rz.size());

        real* duz_d = du_z.D.data();
        real* duz_rx = du_z.Rx.data();
        real* duz_ry = du_z.Ry.data();
        real* duz_rz = du_z.Rz.data();
        const int n_duz_d = static_cast<int>(du_z.D.size());
        const int n_duz_rx = static_cast<int>(du_z.Rx.size());
        const int n_duz_ry = static_cast<int>(du_z.Ry.size());
        const int n_duz_rz = static_cast<int>(du_z.Rz.size());

        // Zero du_x (4 buffers fan out from parent)
        auto zx_d = parent.then_parallel_for(
            "grad_zero_dux_D", rp_t(0, n_dux_d),
            KOKKOS_LAMBDA(int i) { dux_d[i] = 0; });
        auto zx_rx = parent.then_parallel_for(
            "grad_zero_dux_Rx", rp_t(0, n_dux_rx),
            KOKKOS_LAMBDA(int i) { dux_rx[i] = 0; });
        auto zx_ry = parent.then_parallel_for(
            "grad_zero_dux_Ry", rp_t(0, n_dux_ry),
            KOKKOS_LAMBDA(int i) { dux_ry[i] = 0; });
        auto zx_rz = parent.then_parallel_for(
            "grad_zero_dux_Rz", rp_t(0, n_dux_rz),
            KOKKOS_LAMBDA(int i) { dux_rz[i] = 0; });
        auto dux_zeroed =
            Kokkos::Experimental::when_all(zx_d, zx_rx, zx_ry, zx_rz);

        // Zero du_y
        auto zy_d = parent.then_parallel_for(
            "grad_zero_duy_D", rp_t(0, n_duy_d),
            KOKKOS_LAMBDA(int i) { duy_d[i] = 0; });
        auto zy_rx = parent.then_parallel_for(
            "grad_zero_duy_Rx", rp_t(0, n_duy_rx),
            KOKKOS_LAMBDA(int i) { duy_rx[i] = 0; });
        auto zy_ry = parent.then_parallel_for(
            "grad_zero_duy_Ry", rp_t(0, n_duy_ry),
            KOKKOS_LAMBDA(int i) { duy_ry[i] = 0; });
        auto zy_rz = parent.then_parallel_for(
            "grad_zero_duy_Rz", rp_t(0, n_duy_rz),
            KOKKOS_LAMBDA(int i) { duy_rz[i] = 0; });
        auto duy_zeroed =
            Kokkos::Experimental::when_all(zy_d, zy_rx, zy_ry, zy_rz);

        // Zero du_z
        auto zz_d = parent.then_parallel_for(
            "grad_zero_duz_D", rp_t(0, n_duz_d),
            KOKKOS_LAMBDA(int i) { duz_d[i] = 0; });
        auto zz_rx = parent.then_parallel_for(
            "grad_zero_duz_Rx", rp_t(0, n_duz_rx),
            KOKKOS_LAMBDA(int i) { duz_rx[i] = 0; });
        auto zz_ry = parent.then_parallel_for(
            "grad_zero_duz_Ry", rp_t(0, n_duz_ry),
            KOKKOS_LAMBDA(int i) { duz_ry[i] = 0; });
        auto zz_rz = parent.then_parallel_for(
            "grad_zero_duz_Rz", rp_t(0, n_duz_rz),
            KOKKOS_LAMBDA(int i) { duz_rz[i] = 0; });
        auto duz_zeroed =
            Kokkos::Experimental::when_all(zz_d, zz_rx, zz_ry, zz_rz);

        // Chain derivatives (independent since they write to different outputs)
        auto dx_done = dx.add_graph_nodes(dux_zeroed, u, du_x);
        auto dy_done = dy.add_graph_nodes(duy_zeroed, u, du_y);
        auto dz_done = dz.add_graph_nodes(duz_zeroed, u, du_z);

        return Kokkos::Experimental::when_all(dx_done, dy_done, dz_done);
    }
};
} // namespace ccs
