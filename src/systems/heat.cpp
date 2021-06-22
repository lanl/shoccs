#include "heat.hpp"
#include "fields/algorithms.hpp"
#include "fields/selector.hpp"
#include "real3_operators.hpp"
#include <cmath>
#include <numbers>

#include <sol/sol.hpp>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include "operators/discrete_operator.hpp"

#include <range/v3/algorithm/max.hpp>
#include <range/v3/algorithm/min.hpp>
#include <range/v3/algorithm/minmax.hpp>

namespace ccs::systems
{

constexpr auto abs = lift([](auto&& x) { return std::abs(x); });
enum class scalars : int { u };

heat::heat(mesh&& m,
           bcs::Grid&& grid_bcs,
           bcs::Object&& object_bcs,
           manufactured_solution&& m_sol,
           stencil st,
           real diffusivity)
    : m{MOVE(m)},
      grid_bcs{MOVE(grid_bcs)},
      object_bcs{MOVE(object_bcs)},
      m_sol{MOVE(m_sol)},
      lap{laplacian(this->m, st, this->grid_bcs, this->object_bcs)},
      diffusivity{diffusivity},
      neumann_u{m.ss()}
{
    assert(!!(this->m_sol));
    logger = spdlog::basic_logger_st("system", "logs/system.csv", true);
    logger->set_pattern("%v");
    logger->info("Date,Time,Step,Linf,Min,Max");
    logger->set_pattern("%Y-%m-%d %H:%M:%S.%f,%v");
}

//
// sets the field f to the solution
//
void heat::operator()(field& f, const step_controller& c)
{
    if (!m_sol) return;

    auto&& u = f.scalars(scalars::u);
    auto sol = m.xyz | m_sol(c.simulation_time());

    u | sel::D = 0;
    u | m.fluid = sol;
    u | sel::R = sol;
}

//
// Compute the linf error as well as the min/max of the field
//
system_stats heat::stats(const field&, const field& f, const step_controller& step) const
{
    auto&& u = f.scalars(scalars::u);
    auto sol = m.xyz | m_sol(step.simulation_time());
    auto [min, max] = rs::minmax(u | m.fluid);
    real error = rs::max(abs(u - sol) | m.fluid);
    return system_stats{.stats = {error, min, max}};
}

//
// Determine if the computed field is valid by checking the linf error
//
bool heat::valid(const system_stats& stats) const
{
    const auto& v = stats.stats[0];
    return std::isfinite(v) && std::abs(v) <= 1e6;
}

//
// parabolic timestep constraint
//
real heat::timestep_size(const field&, const step_controller& step) const
{
    const auto h_min = rs::min(m.h());
    return step.parabolic_cfl() * h_min * h_min / (4 * diffusivity);
};

//
// rhs = diffusivity * lap(f) + (dQ/dt - diffusivity * lap(Q))
//
// Q is the manufactured solution
//
void heat::rhs(field_view f, real time, field_span rhs) const
{
    auto&& u_rhs = rhs.scalars(scalars::u);
    auto&& u = f.scalars(scalars::u);

    // rhs = diffusivity * lap(u) + (dS/dt - diffusivity * lap(S))
    u_rhs = lap(u, neumann_u);
    u_rhs *= diffusivity;

    if (m_sol) {
        const auto& l = m.xyz;
        // this does not currently account for non-dirichlet bcs on R
        u_rhs | m.fluid +=
            (l | m_sol.ddt(time)) - (diffusivity * (l | m_sol.laplacian(time)));
        // need to zero the mms contribution on dirichlet boundaries
        u_rhs | m.dirichlet(grid_bcs) = 0;
    }
}

//
// Sets the dirichlet boundary values on f at given time.
// Also updates the internal neumann_u to apply neumann boundary conditions.
// This routine MUST be called before evaluating the rhs of the system
//
void heat::update_boundary(field_span f, real time)
{
    auto&& u = f.scalars(scalars::u);
    auto l = m.xyz;

    u | m.dirichlet(grid_bcs) = l | m_sol(time);
    // assumes dirichlet right how
    u | sel::R = l | m_sol(time);

    // set possible neumann bcs;
    neumann_u | m.neumann<0>(grid_bcs) = l | m_sol.gradient(0, time);
    neumann_u | m.neumann<1>(grid_bcs) = l | m_sol.gradient(1, time);
    neumann_u | m.neumann<2>(grid_bcs) = l | m_sol.gradient(2, time);
}

void heat::log(const system_stats& stats, const step_controller& step)
{
    if (!logger) return;

    logger->info(
        fmt::format("{},{},{}", (real)step, (int)step, fmt::join(stats.stats, ",")));
}

std::span<const std::string> heat::names() const { return io_names; }

//
// Convert the system statistics into a real3 summary
//
real3 heat::summary(const system_stats& stats) const
{
    return {stats.stats[0], stats.stats[1], stats.stats[2]};
}

std::optional<heat> heat::from_lua(const sol::table& tbl)
{
    // assume we can only get here if simulation.system.type == "heat" so check
    // for the rest
    real diff = tbl["system"]["diffusivity"].get_or(1.0);

    auto mesh_opt = mesh::from_lua(tbl);
    if (!mesh_opt) return std::nullopt;

    auto bc_opt = bcs::from_lua(tbl, mesh_opt->extents());
    auto st_opt = stencil::from_lua(tbl);

    if (bc_opt && st_opt) {
        auto ms_opt = manufactured_solution::from_lua(tbl, mesh_opt->dims());
        auto t = ms_opt ? MOVE(*ms_opt) : manufactured_solution{};

        return heat{MOVE(*mesh_opt),
                    MOVE(bc_opt->first),
                    MOVE(bc_opt->second),
                    MOVE(t),
                    *st_opt,
                    diff};
    }

    return std::nullopt;
}

system_size heat::size() const { return {1, 0, m.ss()}; }

} // namespace ccs::systems
