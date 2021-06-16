#include "simulation_cycle.hpp"

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

#include <iostream>

namespace ccs
{

simulation_cycle::simulation_cycle(system&& sys,
                                   step_controller&& controller,
                                   integrator&& integrate)
    : sys{MOVE(sys)}, controller{MOVE(controller)}, integrate{MOVE(integrate)}, io{}
{
}

real3 simulation_cycle::run()
{
    // a non-zero time would typically correspond to some kind of restart
    // functionality
    field u0{sys(controller)};
    field u1{u0};

    system_stats stats = sys.stats(u0, u1, controller);

    sys.log(stats, controller);

    // initial write
    io.write(u0, controller, 0);

    while (!controller.done() && sys.valid(stats)) {

        const std::optional<real> dt = sys.timestep_size(u0, controller);
        if (!dt) {
            return {null_v<real>}; //{huge<double>, time};
        }

        u1 = integrate(sys, u0, controller, *dt);

        // update time and step to reflect u1 data
        controller.advance(*dt);

        // compute statistics and handle io
        stats = sys.stats(u0, u1, controller);
        io.write(u1, controller, *dt);
        sys.log(stats, controller);

        // prepare for next iteration to overwrite u0
        using std::swap;
        swap(u0, u1);
    }

    // only return Linf if system ends in a valid state
    if (!controller.done()) {
        return {null_v<real>}; //, time};
    } else {
        return sys.summary(stats);
    }
}

std::optional<simulation_cycle> simulation_cycle::from_lua(const sol::table& tbl)
{
    auto sys_opt = system::from_lua(tbl);
    auto it_opt = integrator::from_lua(tbl);
    auto st_opt = step_controller::from_lua(tbl);

    if (sys_opt && it_opt && st_opt) {
        return simulation_cycle{MOVE(*sys_opt), MOVE(*st_opt), MOVE(*it_opt)};
    } else {
        return std::nullopt;
    }
}
} // namespace ccs
