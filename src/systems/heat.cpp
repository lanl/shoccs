#include "heat.hpp"
#include "fields/algorithms.hpp"
#include "fields/selector.hpp"
#include "real3_operators.hpp"
#include <cmath>
#include <numbers>

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

#include "operators/discrete_operator.hpp"

namespace ccs::systems
{
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
      diffusivity{diffusivity}
{
}

void heat::operator()(field&, const step_controller&) {}

system_stats heat::stats(const field&, const field&, const step_controller&) const
{
    return {};
}

bool heat::valid(const system_stats&) const { return {}; }

real heat::timestep_size(const field&, const step_controller&) const { return {}; };

void heat::rhs(field_view, real, field_span) {}

void heat::update_boundary(field_span, real)
{
    // auto&& [u] = system.scalars(scalars::u);
}

void heat::log(const system_stats&, const step_controller&)
{
    if (auto logger = spdlog::get("system"); logger) { logger->info("Heat"); }
}

std::optional<heat> heat::from_lua(const sol::table& tbl)
{
    // assume we can only get here if simulation.system.type == "heat" so check
    // for the rest
    real diff = tbl["simulation"]["system"]["diffusivity"].get_or(1.0);

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

system_size heat::size() const {
    return {};
}

} // namespace ccs::systems
