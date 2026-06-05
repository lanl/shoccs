#include "simulation_cycle.hpp"

#include "fields/field_registry.hpp"
#include "io/logging.hpp"
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <Kokkos_Timer.hpp>
#include <sol/sol.hpp>

#include <cassert>
#include <iostream>
#include <string>

using namespace std::string_literals;

namespace ccs
{

simulation_cycle::simulation_cycle(system&& sys,
                                   step_controller&& controller,
                                   integrator&& integrate,
                                   field_io&& io,
                                   bool enable_logging)
    : sys{MOVE(sys)},
      controller{MOVE(controller)},
      integrate{MOVE(integrate)},
      io{MOVE(io)},
      logger{enable_logging, "cycle"}
{
}

real3 simulation_cycle::run()
{
    Kokkos::Profiling::ScopedRegion run_region("simulation_cycle::run");
    logger(spdlog::level::info, "begin time stepping");

    // Registry-based field allocation (9.5a)
    sim_registry reg;
    auto sz = sys.size();
    int d_sz  = sz.d_size;
    int rx_sz = sz.rx_size;
    int ry_sz = sz.ry_size;
    int rz_sz = sz.rz_size;

    field_ref u0_ref{0}, u1_ref{1}, rk_ref{2}, srhs_ref{3};
    for (int s = 0; s < sz.nscalars; ++s) {
        u0_ref   = reg.allocate_scalar(0, s, d_sz, rx_sz, ry_sz, rz_sz);
        u1_ref   = reg.allocate_scalar(1, s, d_sz, rx_sz, ry_sz, rz_sz);
        rk_ref   = reg.allocate_scalar(2, s, d_sz, rx_sz, ry_sz, rz_sz);
        srhs_ref = reg.allocate_scalar(3, s, d_sz, rx_sz, ry_sz, rz_sz);
    }
    for (int v = 0; v < sz.nvectors; ++v) {
        u0_ref   = reg.allocate_vector(0, v, d_sz, rx_sz, ry_sz, rz_sz);
        u1_ref   = reg.allocate_vector(1, v, d_sz, rx_sz, ry_sz, rz_sz);
        rk_ref   = reg.allocate_vector(2, v, d_sz, rx_sz, ry_sz, rz_sz);
        srhs_ref = reg.allocate_vector(3, v, d_sz, rx_sz, ry_sz, rz_sz);
    }
    // For zero-field systems (nscalars==0, nvectors==0), refs retain their
    // initial {slot, 0, 0} state — slot_ops correctly no-op.
    assert(u0_ref.n_scalars == sz.nscalars && u0_ref.n_vectors == sz.nvectors);
    sys.initialize(reg, u0_ref, controller);
    reg.deep_copy_slot(u1_ref.slot, u0_ref.slot);

    sys.update_boundary(reg, u0_ref, controller);

    system_stats stats = sys.stats(reg, u0_ref, u1_ref, controller);

    sys.log(stats, controller);

    // initial write
    sys.write(io, reg, u0_ref, controller, .0);

    // Build RHS graph once for graph-capable systems (heat, scalar_wave).
    // Uses u1_ref as the RHS input slot and srhs_ref as the RHS output slot,
    // matching the rk4 integrator's convention.
    sys.build_rhs_graph(reg, u1_ref, reg, srhs_ref);

    Kokkos::Timer cumulative_timer;

    while (controller && sys.valid(stats)) {

        const std::optional<real> dt = sys.timestep_size(reg, u0_ref, controller);
        if (!dt) {
            logger(spdlog::level::info, "required timestep too small");
            return {null_v<real>}; //{huge<double>, time};
        }

        Kokkos::Timer step_timer;

        {
            Kokkos::Profiling::ScopedRegion integrate_region(
                "simulation_cycle::integrate");
            integrate(
                sys, reg, u0_ref, u1_ref, rk_ref, srhs_ref, controller, *dt);
        }

        // update time and step to reflect u1 data
        controller.advance(*dt);

        // compute statistics and handle io
        {
            Kokkos::Profiling::ScopedRegion stats_region(
                "simulation_cycle::stats");
            stats = sys.stats(reg, u0_ref, u1_ref, controller);
        }
        {
            Kokkos::Profiling::ScopedRegion write_region(
                "simulation_cycle::write");
            sys.write(io, reg, u1_ref, controller, *dt);
        }
        stats.wall_time_s = step_timer.seconds();
        sys.log(stats, controller);

        const double step_wall_ms = stats.wall_time_s * 1000.0;

        logger(spdlog::level::info,
               "time= {}  step={}, dt={}, s0={}, wall={:.3f}ms",
               (real)controller,
               (int)controller,
               *dt,
               stats.stats[0],
               step_wall_ms);
        // Copy latest solution to u0 for next iteration.
        // Uses deep_copy (not swap_slots) to preserve stable data pointers
        // required by the pre-built RHS graph.
        reg.deep_copy_slot(u0_ref.slot, u1_ref.slot);
    }

    logger(spdlog::level::info,
           "cumulative wall time: {:.3f}s",
           cumulative_timer.seconds());

    // only return Linf if system ends in a valid state
    if (controller) {
        logger(spdlog::level::info,
               "simulation ended prematurely at time/step  {} / {}",
               (real)controller,
               (int)controller);
        return {(real)controller, null_v<real>, null_v<real>};
    } else {
        auto&& [e, umin, umax] = sys.summary(stats);
        return {(real)controller, e, e};
    }
}

std::optional<simulation_cycle> simulation_cycle::from_lua(const sol::table& tbl)
{
    bool enable_logging = tbl["logging"].get_or(true);
    std::string logging_dir = enable_logging ? tbl["logging_dir"].get_or("logs"s) : ""s;
    logs l{logging_dir, enable_logging, "builder"};

    auto sys_opt = system::from_lua(tbl, l);
    auto it_opt = integrator::from_lua(tbl, l);
    auto st_opt = step_controller::from_lua(tbl, l);
    auto io_opt = field_io::from_lua(tbl, l);

    if (sys_opt && it_opt && st_opt && io_opt) {
        return simulation_cycle{
            MOVE(*sys_opt), MOVE(*st_opt), MOVE(*it_opt), MOVE(*io_opt), l};
    } else {
        return std::nullopt;
    }
}
} // namespace ccs
