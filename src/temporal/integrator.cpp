#include "integrator.hpp"
#include "systems/system.hpp"

#include <sol/sol.hpp>

namespace ccs
{

std::function<void(field_span)> integrator::operator()(system& s,
                                                       const field& f,
                                                       const step_controller& controller,
                                                       real dt)
{
    return std::visit(
        [&s, &f, &controller, dt](auto&& integrator_v) {
            return std::function<void(field_span)>{
                [&s, &f, &controller, dt, &integrator_v](field_span fs) {
                    integrator_v.ensure_size(s.size());
                    integrator_v(s, f, fs, controller, dt);
                }};
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
