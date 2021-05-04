
#include "run_from_sol.hpp"

#include <sol/sol.hpp>

#include "simulation/simulation_builder.hpp"

namespace ccs
{
std::optional<real3> simulation_run(const sol::table& lua)
{
    if (auto sim = simulation_builder().build(lua); sim) {
        return {sim->run()};
    } else {
        return {std::nullopt};
    }
}
} // namespace ccs