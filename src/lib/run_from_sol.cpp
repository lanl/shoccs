
#include "run_from_sol.hpp"

#include <sol/sol.hpp>

#include "simulation/SimulationBuilder.hpp"

namespace ccs
{
std::optional<real3> simulation_run(const sol::table& lua)
{
    if (auto sim = SimulationBuilder().build(lua); sim) {
        return {sim->run()};
    } else {
        return {std::nullopt};
    }
}
} // namespace ccs