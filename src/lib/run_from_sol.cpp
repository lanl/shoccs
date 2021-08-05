
#include "shoccs.hpp"

#include <sol/sol.hpp>

#include "simulation/simulation_cycle.hpp"

namespace ccs
{
std::optional<real3> simulation_run(const sol::table& lua)
{
    if (auto cycle = simulation_cycle::from_lua(lua); cycle) {
        return {cycle->run()};
    } else {
        return {std::nullopt};
    }
}
} // namespace ccs
