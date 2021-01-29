#pragma once

#include <optional>
#include <sol/forward.hpp>

#include "SimulationCycle.hpp"
#include "types.hpp"

namespace ccs
{

class SimulationBuilder
{

public:
    SimulationBuilder() = default;

    std::optional<SimulationCycle> build(const sol::table& lua) &&;
};
} // namespace ccs