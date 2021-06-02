#include "heat.hpp"
#include "fields/algorithms.hpp"
#include "fields/selector.hpp"
#include "real3_operators.hpp"
#include <cmath>
#include <numbers>

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

#include "operators/discrete_operator.hpp"

#include <range/v3/algorithm/max.hpp>

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
}

void heat::operator()(field& f, const step_controller& c)
{
    if (f.nscalars() == 0) { f = field{size()}; }
    if (!m_sol) return;

    auto&& u = f.scalars(scalars::u);
    auto sol = m.location | m_sol(c.simulation_time());

    u | sel::D = 0;
    u | m.fluid = sol;
    u | sel::R = sol;
}

system_stats heat::stats(const field&, const field& f, const step_controller& step) const
{
    auto&& u = f.scalars(scalars::u);
    auto sol = m.location | m_sol(step.simulation_time());
    real error = rs::max(abs(u - sol) | m.fluid);
    return system_stats{.stats = {error}};
}

bool heat::valid(const system_stats& stats) const
{
    const auto& v = stats.stats[0];
    return std::isfinite(v) && std::abs(v) <= 1e6;
}

real heat::timestep_size(const field&, const step_controller&) const { return {}; };

void heat::rhs(field_view f, real time, field_span rhs) const
{
    auto&& u_rhs = rhs.scalars(scalars::u);
    auto&& u = f.scalars(scalars::u);

    // rhs = diffusivity * lap(u) + (dS/dt - diffusivity * lap(S))
    u_rhs = lap(u, neumann_u);
    u_rhs *= diffusivity;

    if (m_sol) {
        const auto& l = m.location;
        // this does not currently account for non-dirichlet bcs on R
        u_rhs | m.fluid +=
            (l | m_sol.ddt(time)) - (diffusivity * (l | m_sol.laplacian(time)));
        // need to zero the mms contribution on dirichlet boundaries
        u_rhs | m.dirichlet(grid_bcs) = 0;
    }
}

void heat::update_boundary(field_span f, real time)
{
    auto&& u = f.scalars(scalars::u);
    auto l = m.location;

    u | m.dirichlet(grid_bcs) = l | m_sol(time);
    // assumes dirichlet right how
    u | sel::R = l | m_sol(time);

    // set possible neumann bcs;
    neumann_u | m.neumann<0>(grid_bcs) = l | m_sol.gradient(0, time);
    neumann_u | m.neumann<1>(grid_bcs) = l | m_sol.gradient(1, time);
    neumann_u | m.neumann<2>(grid_bcs) = l | m_sol.gradient(2, time);
}

void heat::log(const system_stats&, const step_controller&)
{
    if (auto logger = spdlog::get("system"); logger) { logger->info("Heat"); }
}

std::optional<heat> heat::from_lua(const sol::table& tbl)
{
    // assume we can only get here if simulation.system.type == "heat" so check
    // for the rest
    real diff = tbl["system"]["diffusivity"].get_or(1.0);

    auto mesh_opt = mesh::from_lua(tbl);
    auto bc_opt = bcs::from_lua(tbl);
    auto st_opt = stencil::from_lua(tbl);

    if (mesh_opt && bc_opt && st_opt) {
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
