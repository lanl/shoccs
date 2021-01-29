#pragma once

#include <optional>
#include <sol/forward.hpp>

#include "types.hpp"

namespace ccs
{
std::optional<real3> simulation_run(const sol::table& lua);
} // namespace ccs