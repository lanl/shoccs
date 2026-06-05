#pragma once

#include "fields/field_registry.hpp"
#include "io/field_io.hpp"
#include "mesh/mesh.hpp"
#include "mms/manufactured_solutions.hpp"
#include "operators/laplacian.hpp"
#include "temporal/step_controller.hpp"
#include <Kokkos_Graph.hpp>
#include <optional>
#include <sol/forward.hpp>

namespace ccs::systems
{
//
// solve dT/dt = k lap T
//
class heat
{

    mesh m;
    bcs::Grid grid_bcs;
    bcs::Object object_bcs;
    manufactured_solution m_sol;

    laplacian lap;
    real diffusivity;

    std::vector<real> neumann_d, neumann_rx, neumann_ry, neumann_rz;
    std::vector<real> src_d, src_rx, src_ry, src_rz;
    std::vector<real> error_d, error_rx, error_ry, error_rz;

    logs logger;

    std::vector<std::string> io_names = {"U", "Error"};

    // Pre-built graph for submit_rhs_graph().
    std::optional<Kokkos::Experimental::Graph<execution_space>> rhs_graph_;

public:
    heat() = default;

    heat(mesh&& m,
         bcs::Grid&& grid_bcs,
         bcs::Object&& object_bcs,
         manufactured_solution&& m_sol,
         stencil st,
         real diffusivity,
         const logs& = {});

    static std::optional<heat> from_lua(const sol::table&, const logs& = {});

    bool valid(const system_stats&) const;

    void log(const system_stats&, const step_controller&);

    real3 summary(const system_stats&) const;

    system_size size() const;

    void fill_source(real time);

    void rhs(const sim_registry& reg, field_ref input,
             sim_registry& out_reg, field_ref output, real time);
    void build_rhs_graph(scalar_view u, scalar_span du);
    void submit_rhs_graph();
    void update_boundary(sim_registry& reg, field_ref ref, real time);
    real timestep_size(const sim_registry& reg, field_ref ref,
                       const step_controller&) const;
    system_stats stats(const sim_registry& reg, field_ref u0,
                       field_ref u1, const step_controller&) const;
    void initialize(sim_registry& reg, field_ref ref, const step_controller&);
    bool write(field_io& io, const sim_registry& reg, field_ref ref,
               const step_controller& c, real dt);
};
} // namespace ccs::systems
