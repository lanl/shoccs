#pragma once

#include "types.hpp"

#include "io/FieldIO.hpp"
#include "systems/system.hpp"
#include "temporal/integrator.hpp"
#include "temporal/step_controller.hpp"

namespace ccs
{
class simulation_cycle
{
    system sys;
    step_controller controller;
    integrator integrate;
    FieldIO io;

public:
    simulation_cycle() = default;

    // static ::

#if 0
    simulation_cycle()
    simulation_cycle(std::unique_ptr<integrate>&& integrator,
                             std::unique_ptr<system>&& sys,
                             field_io&& io,
                             timestep_controls&& tc,
                             loggers&& logs)
        : integrator{std::move(integrator)},
          sys{std::move(sys)},
          io{std::move(io)},
          tc{std::move(tc)},
          logs{std::move(logs)}
    {
    }
#endif

    real3 run();
};
} // namespace ccs
