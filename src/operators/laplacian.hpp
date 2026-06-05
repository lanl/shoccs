#pragma once

#include "derivative.hpp"

#include <Kokkos_Graph.hpp>
#include <optional>

namespace ccs
{
class laplacian
{
    derivative dx;
    derivative dy;
    derivative dz;
    index_extents ex;

    // Pre-built graph for submit_graph().
    std::optional<Kokkos::Experimental::Graph<execution_space>> graph_;

public:
    laplacian() = default;

    laplacian(const mesh&,
              const stencil&,
              const bcs::Grid&,
              const bcs::Object&,
              const logs& logger = {});

    // when there are no neumann conditions in the problem
    std::function<void(scalar_span)> operator()(scalar_view) const;

    std::function<void(scalar_span)> operator()(scalar_view field_values,
                                                scalar_view derivative_values) const;

    // Build a pre-instantiated graph for the non-Neumann overload.
    void build_graph(scalar_view u, scalar_span du);

    // Build a pre-instantiated graph for the Neumann overload.
    void build_graph(scalar_view u, scalar_view nu, scalar_span du);

    // Submit the pre-built graph.
    void submit_graph();

    // Add laplacian nodes to an existing graph, chaining from parent.
    // Zeros du, then chains dx → dy → dz (all accumulate with plus_eq).
    // Returns the final node so the caller can chain further.
    template <typename NodeT>
    auto add_graph_nodes(NodeT parent, scalar_view u, scalar_span du) const
    {
        using rp_t = Kokkos::RangePolicy<execution_space>;

        real* d_ptr = du.D.data();
        real* rx_ptr = du.Rx.data();
        real* ry_ptr = du.Ry.data();
        real* rz_ptr = du.Rz.data();
        const int n_d = static_cast<int>(du.D.size());
        const int n_rx = static_cast<int>(du.Rx.size());
        const int n_ry = static_cast<int>(du.Ry.size());
        const int n_rz = static_cast<int>(du.Rz.size());

        // Zero-fill all 4 components of du
        auto z_d = parent.then_parallel_for(
            "lap_zero_D", rp_t(0, n_d),
            KOKKOS_LAMBDA(int i) { d_ptr[i] = 0; });
        auto z_rx = parent.then_parallel_for(
            "lap_zero_Rx", rp_t(0, n_rx),
            KOKKOS_LAMBDA(int i) { rx_ptr[i] = 0; });
        auto z_ry = parent.then_parallel_for(
            "lap_zero_Ry", rp_t(0, n_ry),
            KOKKOS_LAMBDA(int i) { ry_ptr[i] = 0; });
        auto z_rz = parent.then_parallel_for(
            "lap_zero_Rz", rp_t(0, n_rz),
            KOKKOS_LAMBDA(int i) { rz_ptr[i] = 0; });

        auto zeroed = Kokkos::Experimental::when_all(z_d, z_rx, z_ry, z_rz);

        // Chain derivatives sequentially (all accumulate into du)
        auto d0 = dx.add_graph_nodes(zeroed, u, du, plus_eq);
        auto d1 = dy.add_graph_nodes(d0, u, du, plus_eq);
        return dz.add_graph_nodes(d1, u, du, plus_eq);
    }

    // Neumann overload: adds Neumann nodes at end of each derivative's D-space chain.
    template <typename NodeT>
    auto add_graph_nodes(NodeT parent, scalar_view u, scalar_view nu,
                         scalar_span du) const
    {
        using rp_t = Kokkos::RangePolicy<execution_space>;

        real* d_ptr = du.D.data();
        real* rx_ptr = du.Rx.data();
        real* ry_ptr = du.Ry.data();
        real* rz_ptr = du.Rz.data();
        const int n_d = static_cast<int>(du.D.size());
        const int n_rx = static_cast<int>(du.Rx.size());
        const int n_ry = static_cast<int>(du.Ry.size());
        const int n_rz = static_cast<int>(du.Rz.size());

        auto z_d = parent.then_parallel_for(
            "lap_zero_D", rp_t(0, n_d),
            KOKKOS_LAMBDA(int i) { d_ptr[i] = 0; });
        auto z_rx = parent.then_parallel_for(
            "lap_zero_Rx", rp_t(0, n_rx),
            KOKKOS_LAMBDA(int i) { rx_ptr[i] = 0; });
        auto z_ry = parent.then_parallel_for(
            "lap_zero_Ry", rp_t(0, n_ry),
            KOKKOS_LAMBDA(int i) { ry_ptr[i] = 0; });
        auto z_rz = parent.then_parallel_for(
            "lap_zero_Rz", rp_t(0, n_rz),
            KOKKOS_LAMBDA(int i) { rz_ptr[i] = 0; });

        auto zeroed = Kokkos::Experimental::when_all(z_d, z_rx, z_ry, z_rz);

        // Chain derivatives sequentially with Neumann
        auto d0 = dx.add_graph_nodes(zeroed, u, nu, du, plus_eq);
        auto d1 = dy.add_graph_nodes(d0, u, nu, du, plus_eq);
        return dz.add_graph_nodes(d1, u, nu, du, plus_eq);
    }
};
} // namespace ccs
