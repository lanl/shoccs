#pragma once

#include <optional>
#include <sol/forward.hpp>

#include "simulation_cycle.hpp"
#include "types.hpp"

namespace ccs
{

class simulation_builder
{

public:
    simulation_builder() = default;

    std::optional<simulation_cycle> build(const sol::table& lua) &&;
};
} // namespace ccs