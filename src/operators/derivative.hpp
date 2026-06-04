#pragma once

#include "types.hpp"

#include "boundaries.hpp"
#include "fields/scalar.hpp"
#include "matrices/block.hpp"
#include "matrices/csr.hpp"
#include "matrices/matrix_visitor.hpp"
#include "mesh/mesh.hpp"
#include "stencils/stencil.hpp"

#include "io/logging.hpp"

#include <Kokkos_Graph.hpp>
#include <optional>

namespace ccs
{
class derivative
{
    int dir;
    // Operators for updating field data
    matrix::block O;
    matrix::csr B;
    matrix::csr N;
    // operators for updating boundary data on Rx/y/z
    matrix::csr Bfx, Brx;
    matrix::csr Bfy, Bry;
    matrix::csr Bfz, Brz;
    // Pre-built graph for submit_graph().
    std::optional<Kokkos::Experimental::Graph<execution_space>> graph_;

    // Submit all kernels (R-space + D-space) without fencing.
    template <typename Op = eq_t>
        requires std::invocable<Op, real&, real>
    void apply_kernels(scalar_view u, scalar_span du, Op op = {}) const;

public:
    derivative() = default;

    derivative(int dir,
               const mesh& m,
               const stencil& st,
               const bcs::Grid& grid_bcs,
               const bcs::Object& object_bcs,
               const logs& = {});

    void visit(matrix::visitor& v) const
    {
        // Assumes 1d
        O.visit(v);
        B.visit(v);
        Bfx.visit(v);
        Brx.visit(v);
    }

    // operator for when neumann conditions are not needed
    template <typename Op = eq_t>
        requires std::invocable<Op, real&, real>
    void operator()(scalar_view, scalar_span, Op op = {}) const;

    // operator for when neumann conditions may be applied
    template <typename Op = eq_t>
        requires std::invocable<Op, real&, real>
    void operator()(scalar_view field_values,
                    scalar_view derivative_values,
                    scalar_span,
                    Op op = {}) const;

    // Build a pre-instantiated graph for the non-Neumann overload.
    // Buffer pointers are baked in at creation time.
    template <typename Op = eq_t>
        requires std::invocable<Op, real&, real>
    void build_graph(scalar_view u, scalar_span du, Op op = {});

    // Build a pre-instantiated graph for the Neumann overload.
    template <typename Op = eq_t>
        requires std::invocable<Op, real&, real>
    void build_graph(scalar_view u, scalar_view nu, scalar_span du, Op op = {});

    // Submit the pre-built graph.
    void submit_graph();

    // Add derivative nodes to an existing graph, chaining from parent.
    // Returns a when_all of all leaf nodes so the caller can chain further.
    template <typename Op = eq_t, typename NodeT>
        requires std::invocable<Op, real&, real>
    auto add_graph_nodes(NodeT parent, scalar_view u, scalar_span du, Op op = {}) const
    {
        const real* u_D = u.D.data();
        const real* u_Rx = u.Rx.data();
        const real* u_Ry = u.Ry.data();
        const real* u_Rz = u.Rz.data();
        real* du_D = du.D.data();
        real* du_Rx = du.Rx.data();
        real* du_Ry = du.Ry.data();
        real* du_Rz = du.Rz.data();
        const real* b_src = (dir == 0) ? u_Rx : (dir == 1) ? u_Ry : u_Rz;

        // R-space chains (3 independent pairs)
        auto bfx = Bfx.graph_node(parent, u_D, du_Rx);
        auto brx = Brx.graph_node(bfx, u_Rx, du_Rx);

        auto bfy = Bfy.graph_node(parent, u_D, du_Ry);
        auto bry = Bry.graph_node(bfy, u_Ry, du_Ry);

        auto bfz = Bfz.graph_node(parent, u_D, du_Rz);
        auto brz = Brz.graph_node(bfz, u_Rz, du_Rz);

        // D-space chain
        auto o = O.graph_node(parent, u_D, du_D, op);
        auto b = B.graph_node(o, b_src, du_D);

        return Kokkos::Experimental::when_all(brx, bry, brz, b);
    }

    // Neumann overload: adds N node at end of D-space chain.
    template <typename Op = eq_t, typename NodeT>
        requires std::invocable<Op, real&, real>
    auto add_graph_nodes(NodeT parent, scalar_view u, scalar_view nu,
                         scalar_span du, Op op = {}) const
    {
        const real* u_D = u.D.data();
        const real* u_Rx = u.Rx.data();
        const real* u_Ry = u.Ry.data();
        const real* u_Rz = u.Rz.data();
        const real* nu_D = nu.D.data();
        real* du_D = du.D.data();
        real* du_Rx = du.Rx.data();
        real* du_Ry = du.Ry.data();
        real* du_Rz = du.Rz.data();
        const real* b_src = (dir == 0) ? u_Rx : (dir == 1) ? u_Ry : u_Rz;

        // R-space chains (3 independent pairs)
        auto bfx = Bfx.graph_node(parent, u_D, du_Rx);
        auto brx = Brx.graph_node(bfx, u_Rx, du_Rx);

        auto bfy = Bfy.graph_node(parent, u_D, du_Ry);
        auto bry = Bry.graph_node(bfy, u_Ry, du_Ry);

        auto bfz = Bfz.graph_node(parent, u_D, du_Rz);
        auto brz = Brz.graph_node(bfz, u_Rz, du_Rz);

        // D-space chain with Neumann
        auto o = O.graph_node(parent, u_D, du_D, op);
        auto b = B.graph_node(o, b_src, du_D);
        auto n = N.graph_node(b, nu_D, du_D);

        return Kokkos::Experimental::when_all(brx, bry, brz, n);
    }
};
} // namespace ccs
