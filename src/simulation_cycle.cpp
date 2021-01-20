#include "simulation_cycle.hpp"

namespace ccs
{

result_t simulation_cycle::run()
{

        real time = 0.0;
        int step = 0;

        // a non-zero time would typically correspond to some kind of restart
        // functionality
        system_field u0 {sys(time)};
        system_field u1 {sys(time)};

        system_stats stats = sys.stats(u0, u1, time);

        sys.log(logs["system"], stats, step, time);

        // initial write
        io.write(u0, step, time, 0);

        const auto cfl = tc.cfl();

        while (!tc.done(time, step) && sys.valid(stats)) {

                std::optional<real> dt = tc.check_reduce_step_size(sys.timestep_size(cfl));
                if (!dt) {
#ifndef NDEBUG
                        std::cout << "simulation failed at time: " << time << "\n";
#endif
                        return {huge<double>, time};
                }
                // how to avoid making this a redundant copy operation?
                // have integrator.step return a system_field of ranges which are
                // then copied into u1. Requires that the step function have a different
                // return value for each system.  Easy to do with a variant/visit call
                // right here but harder to do if I want to hide the variant in the 
                // `integrator` type. 
                // u1 = integrator.step(u0, time, dt);
                //integrator->step(*sys, time, dt);
                // Actually the step routine requires u0 and u1 so it should just be:
                u1 = integrator.step(sys, u0, time, *dt)

                // update time and step to reflect u1 data
                time += *dt;
                step += 1;

                // compute statistics and handle io
                stats = sys.stats(u0, u1, time);
                io.write(u1, step, time, *dt);
                sys.log(logs["system"], stats, step, time);

                // prepare for next iteration to overwrite u0
                swap(u0, u1);
        }

        // only return Linf if system ends in a valid state
        if (tc.done(time, step)) {
#ifndef NDEBUG
                std::cout << "simulation failed at time/step: " << time << " / " << step
                          << "\n";
#endif
                return {huge<double>, time};
        } else {
#ifndef NDEBUG
                std::cout << "simulation succeeded with Linf/L2: " << stats.Linf_acc
                          << " / " << std::sqrt(stats.L2_acc / stats.n_acc) << "\n";
#endif
                return {stats.Linf_acc, std::sqrt(stats.L2_acc / stats.n_acc)};
        }
}

std::vector<double> simulation_cycle::solution() const
{
        return sys.current_solution();
}

std::vector<double> simulation_cycle::error() const
{
        return sys.current_error();
}
} // namespace pdg
