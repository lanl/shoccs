#pragma once

#include "types.hpp"

#include "io/FieldIO.hpp"
#include "systems/system.hpp"
#include "temporal/integrator.hpp"
#include "temporal/step_controller.hpp"

#include <sol/forward.hpp>

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

    simulation_cycle(system&&, step_controller&&, integrator&&);

    static std::optional<simulation_cycle> from_lua(const sol::table&);

    real3 run();
};
} // namespace ccs
