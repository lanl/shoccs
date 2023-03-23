#include "simulation_cycle.hpp"

#include "io/logging.hpp"
#include <sol/sol.hpp>

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
    logger(spdlog::level::info, "begin time stepping");
    // a non-zero time would typically correspond to some kind of restart
    // functionality
    field u0{sys(controller)};
    field u1{u0};

    sys.update_boundary(u0, controller);

    system_stats stats = sys.stats(u0, u1, controller);

    sys.log(stats, controller);

    // initial write
    sys.write(io, u0, controller, .0);

    while (controller && sys.valid(stats)) {

        const std::optional<real> dt = sys.timestep_size(u0, controller);
        if (!dt) {
            logger(spdlog::level::info, "required timestep too small");
            return {null_v<real>}; //{huge<double>, time};
        }
        u1 = integrate(sys, u0, controller, *dt);

        // update time and step to reflect u1 data
        controller.advance(*dt);

        // compute statistics and handle io
        stats = sys.stats(u0, u1, controller);
        sys.write(io, u1, controller, *dt);
        sys.log(stats, controller);

        logger(spdlog::level::info,
               "time= {}  step={}, dt={}, s0={}",
               (real)controller,
               (int)controller,
               *dt,
               stats.stats[0]);
        // prepare for next iteration to overwrite u0
        using std::swap;
        swap(u0, u1);
    }

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
