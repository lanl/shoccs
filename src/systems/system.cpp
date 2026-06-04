#include "system.hpp"

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

namespace ccs
{

bool system::valid(const system_stats& stats) const
{
    return std::visit([&stats](auto&& sys) { return sys.valid(stats); }, v);
}

void system::log(const system_stats& stats, const step_controller& controller)
{
    return std::visit(
        [&stats, &controller](auto&& current_system) {
            current_system.log(stats, controller);
        },
        v);
}

real3 system::summary(const system_stats& stats) const
{
    return std::visit([&stats](auto&& s) { return s.summary(stats); }, v);
}

system_size system::size() const
{
    return std::visit([](auto&& current_system) { return current_system.size(); }, v);
}

// Registry-based dispatch methods

void system::rhs(const sim_registry& creg, field_ref input,
                 sim_registry& reg, field_ref output, real time)
{
    std::visit([&](auto&& s) { s.rhs(creg, input, reg, output, time); }, v);
}

void system::build_rhs_graph(const sim_registry& creg, field_ref input,
                             sim_registry& reg, field_ref output)
{
    std::visit([&](auto&& s) {
        if constexpr (requires {
            s.build_rhs_graph(std::declval<scalar_view>(),
                              std::declval<scalar_span>());
        }) {
            constexpr auto sh = scalar_handle{0};
            auto u = extract_scalar_view(creg, input, sh);
            auto du = extract_scalar_span(reg, output, sh);
            s.build_rhs_graph(u, du);
        }
    }, v);
}

void system::submit_rhs_graph(const sim_registry& creg, field_ref input,
                              sim_registry& reg, field_ref output, real time)
{
    std::visit([&](auto&& s) {
        if constexpr (requires { s.submit_rhs_graph(); }) {
            if constexpr (requires { s.fill_source(time); })
                s.fill_source(time);
            s.submit_rhs_graph();
        } else {
            s.rhs(creg, input, reg, output, time);
        }
    }, v);
}

void system::update_boundary(sim_registry& reg, field_ref ref, real time)
{
    std::visit([&](auto&& s) { s.update_boundary(reg, ref, time); }, v);
}

system_stats system::stats(const sim_registry& reg, field_ref u0,
                           field_ref u1, const step_controller& ctrl) const
{
    return std::visit(
        [&](auto&& s) { return s.stats(reg, u0, u1, ctrl); }, v);
}

void system::initialize(sim_registry& reg, field_ref ref, const step_controller& ctrl)
{
    std::visit([&](auto&& s) { s.initialize(reg, ref, ctrl); }, v);
}

bool system::write(field_io& io, const sim_registry& reg, field_ref ref,
                   const step_controller& c, real dt)
{
    return std::visit(
        [&](auto&& s) { return s.write(io, reg, ref, c, dt); }, v);
}

std::optional<real> system::timestep_size(const sim_registry& reg, field_ref u,
                                          const step_controller& ctrl) const
{
    const auto predicted_dt = std::visit(
        [&](auto&& s) { return s.timestep_size(reg, u, ctrl); }, v);
    return ctrl.check_timestep_size(predicted_dt);
}

std::optional<system> system::from_lua(const sol::table& tbl, const logs& logger)
{
    auto m = tbl["system"];
    if (!m.valid()) {
        logger(spdlog::level::err, "simulation.system must be specified");
        return std::nullopt;
    }

    auto type = m["type"].get_or(std::string{});

    if (type == "heat") {
        logger(spdlog::level::info, "building heat system");
        if (auto opt = systems::heat::from_lua(tbl, logger); opt)
            return system(MOVE(*opt));
    } else if (type == "scalar wave") {
        logger(spdlog::level::info, "building scalar_wave system");
        if (auto opt = systems::scalar_wave::from_lua(tbl, logger); opt)
            return system(MOVE(*opt));
    } else if (type == "inviscid vortex") {
        logger(spdlog::level::info, "building inviscid_vortex system");
        return system(systems::inviscid_vortex{});
    } else if (type == "eigenvalues") {
        logger(spdlog::level::info, "building hyperbolic_eigenvalues system");
        if (auto opt = systems::hyperbolic_eigenvalues::from_lua(tbl, logger); opt)
            return system(MOVE(*opt));
    } else {
        logger(spdlog::level::err, "unrecognized system.type");
    }
    return std::nullopt;
}
} // namespace ccs
