#include "integrator.hpp"
#include "systems/system.hpp"

#include <sol/sol.hpp>

namespace ccs
{

void integrator::operator()(system& sys, sim_registry& reg,
                            field_ref u0, field_ref output,
                            field_ref scratch1, field_ref scratch2,
                            const step_controller& ctrl, real dt)
{
    std::visit(
        [&](auto&& integ) {
            using T = std::decay_t<decltype(integ)>;
            if constexpr (std::is_same_v<T, integrators::rk4>) {
                integ(sys, reg, u0, output, scratch1, scratch2, ctrl, dt);
            } else if constexpr (std::is_same_v<T, integrators::euler>) {
                integ(sys, reg, u0, output, scratch2, ctrl, dt);
            }
            // integrators::empty: no-op
        },
        v);
}

std::optional<integrator> integrator::from_lua(const sol::table& tbl, const logs& logger)
{

    auto m = tbl["integrator"];
    if (!m.valid()) {
        logger(spdlog::level::warn,
               "simulation.integrator not specified.  Defaulting to empty");
        return integrator{integrators::empty{}};
    }

    auto type = m["type"].get_or(std::string{});

    if (type == "rk4") {
        logger(spdlog::level::info, "building rk4 integrator");
        return integrator{integrators::rk4{}};
    } else if (type == "euler") {
        logger(spdlog::level::info, "building euler integrator");
        return integrator{integrators::euler{}};
    } else {
        logger(spdlog::level::err, "integrator.type must be one of: [rk4, euler]");
        return std::nullopt;
    }
}
} // namespace ccs
