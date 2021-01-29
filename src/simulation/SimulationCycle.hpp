#pragma once

#include "types.hpp"

#include "systems/System.hpp"
#include "StepController.hpp"
#include "temporal/Integrator.hpp"
#include "io/FieldIO.hpp"


namespace ccs
{
class SimulationCycle
{
    System system;
    StepController controller;
    Integrator integrator;
    FieldIO io;

public:
    SimulationCycle() = default;

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
