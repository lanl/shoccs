#include "types.hpp"
#include <sol/forward.hpp>

namespace ccs
{
auto simulation_run(const sol::table& lua)
{
    auto builder = simulation_builder();

    if (auto sim = MOVE(builder).build(lua); sim) {
        return sim->run();
    } else {
        return errors;
    }
}
} // namespace ccs