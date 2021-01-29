#include "SimulationBuilder.hpp"
#include <sol/sol.hpp>

namespace ccs
{

std::optional<SimulationCycle>
SimulationBuilder::build([[maybe_unused]] const sol::table& lua) &&
{
    return {SimulationCycle{}};
}
} // namespace ccs