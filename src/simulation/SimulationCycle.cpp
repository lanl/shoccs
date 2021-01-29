#include "SimulationCycle.hpp"

#include <iostream>

namespace ccs
{

real3 SimulationCycle::run()
{
    // a non-zero time would typically correspond to some kind of restart
    // functionality
    SystemField u0{system(controller)};
    SystemField u1{u0};

    SystemStats stats = system.stats(u0, u1, controller);

    system.log(stats, controller);

    // initial write
    io.write(u0, controller, 0);

    while (!controller.done() && system.valid(stats)) {

        const std::optional<real> dt = system.timestep_size(u0, controller);
        if (!dt) {
            return {null_v<real>}; //{huge<double>, time};
        }
        
        u1 = integrator(system, u0, controller, *dt);

        // update time and step to reflect u1 data
        controller.advance(*dt);

        // compute statistics and handle io
        stats = system.stats(u0, u1, controller);
        io.write(u1, controller, *dt);
        system.log(stats, controller);

        // prepare for next iteration to overwrite u0
        using std::swap;
        swap(u0, u1);
    }

    // only return Linf if system ends in a valid state
    if (!controller.done()) {
        return {null_v<real>}; //, time};
    } else {
        return {}; // stats.Linf_acc, std::sqrt(stats.L2_acc / stats.n_acc)};
    }
}
} // namespace ccs
