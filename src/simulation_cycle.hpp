#pragma once

#include "io/field_io.hpp"
#include "systems/system.hpp"
#include "temporal/integrate.hpp"
#include "timestep_controls.hpp"

namespace ccs
{
class simulation_cycle
{
    system sys;
    integrator integrate;
    field_io io;
    timestep_controls tc;
    loggers logs;

public:
    simulation_cycle() = default;

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

    result_t run();

    std::vector<double> solution() const;
    std::vector<double> error() const;
};
} // namespace pdg
