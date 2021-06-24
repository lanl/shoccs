#include "integrator.hpp"
#include "systems/system.hpp"
#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

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

std::optional<integrator> integrator::from_lua(const sol::table& tbl)
{

    auto m = tbl["integrator"];
    if (!m.valid()) {
        spdlog::error("simulation.integrator must be specified");
        return std::nullopt;
    }

    auto type = m["type"].get_or(std::string{});

    if (type == "rk4") {
        spdlog::info("building rk4 integrator");
        return integrator{integrators::rk4{}};
    } else if (type == "euler") {
        spdlog::info("building euler integrator");
        return integrator{integrators::euler{}};
    } else {
        spdlog::error("integrator.type must be one of: rk4");
        return std::nullopt;
    }
}
} // namespace ccs
