#pragma once

#include "fields/field_registry.hpp"
#include "io/field_io.hpp"
#include "operators/gradient.hpp"
#include "temporal/step_controller.hpp"
#include "types.hpp"

#include <Kokkos_Graph.hpp>
#include <optional>
#include <sol/forward.hpp>

namespace ccs::systems
{

// the system of pdes to solve is in this class
class scalar_wave
{
    mesh m;
    bcs::Grid grid_bcs;
    bcs::Object object_bcs;

    gradient grad;

    real3 center; // center of the circular wave
    real radius;

    // Wave speed coefficients (3 spatial components x {D, Rx, Ry, Rz})
    std::vector<real> gG_xd, gG_xrx, gG_xry, gG_xrz;
    std::vector<real> gG_yd, gG_yrx, gG_yry, gG_yrz;
    std::vector<real> gG_zd, gG_zrx, gG_zry, gG_zrz;
    // Gradient scratch (3 spatial components x {D, Rx, Ry, Rz})
    std::vector<real> du_xd, du_xrx, du_xry, du_xrz;
    std::vector<real> du_yd, du_yrx, du_yry, du_yrz;
    std::vector<real> du_zd, du_zrx, du_zry, du_zrz;

    std::vector<real> error_d, error_rx, error_ry, error_rz;

    real max_error;

    logs logger;
    std::vector<std::string> io_names = {"U", "Error"};

    // Pre-built graph for submit_rhs_graph().
    std::optional<Kokkos::Experimental::Graph<execution_space>> rhs_graph_;

public:
    scalar_wave() = default;

    scalar_wave(mesh&&,
                bcs::Grid&&,
                bcs::Object&&,
                stencil,
                real3 center,
                real radius,
                real max_error = 100.0,
                const logs& = {});

    bool valid(const system_stats&) const;

    real3 summary(const system_stats&) const;

    void log(const system_stats&, const step_controller&);

    system_size size() const;

    static std::optional<scalar_wave> from_lua(const sol::table&, const logs& = {});

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
