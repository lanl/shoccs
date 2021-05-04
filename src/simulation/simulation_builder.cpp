#include "simulation_builder.hpp"
#include <sol/sol.hpp>

namespace ccs
{

std::optional<simulation_cycle>
simulation_builder::build([[maybe_unused]] const sol::table& lua) &&
{
    return {simulation_cycle{}};
}
} // namespace ccs