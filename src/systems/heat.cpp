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

void heat::operator()(field& f, const step_controller& c)
{
    if (f.nscalars() == 0) { f = field{size()}; }
    if (!m_sol) return;

    auto&& u = f.scalars(scalars::u);
    auto t = vs::transform([this, time = c.simulation_time()](auto&& loc) {
        return m_sol(time, real3{get<0>(loc), get<1>(loc), get<2>(loc)});
    });
    u | m.fluid = m.location | t;
    u | sel::R = m.location | t;
}

system_stats heat::stats(const field&, const field&, const step_controller&) const
{
    return {};
}

bool heat::valid(const system_stats&) const { return {}; }

real heat::timestep_size(const field&, const step_controller&) const { return {}; };

void heat::rhs(field_view, real, field_span) {}

void heat::update_boundary(field_span f, real time)
{
    auto&& u = f.scalars(scalars::u);
    auto bvals = m.location | vs::transform([this, time](auto&& loc) {
                     return m_sol(time, real3{get<0>(loc), get<1>(loc), get<2>(loc)});
                 });
    if (grid_bcs[0].left == bcs::Dirichlet) u | m.xmin = bvals;
    if (grid_bcs[0].right == bcs::Dirichlet) u | m.xmax = bvals;
    if (grid_bcs[1].left == bcs::Dirichlet) u | m.ymin = bvals;
    if (grid_bcs[1].right == bcs::Dirichlet) u | m.ymax = bvals;
    if (grid_bcs[2].left == bcs::Dirichlet) u | m.zmin = bvals;
    if (grid_bcs[2].right == bcs::Dirichlet) u | m.zmax = bvals;

    // what if we could write as:
    // u | m.dirichlet(grid_bcs) = bvals
    // u | m.dirichelt(object_bcs) = bvals
    // or
    // u | m.dirichlet(grid_bcs, object_bcs) = bvals ?
    // u | m.neumann(grid_bcs, object_bcs) = nvals ?
    // would this work if this became something like tuple{u | m.xmin, u | m.ymin } =
    // bvals...? no. because the size of the tuple is compile time choice but the
    // predicate is a run-time choice. we could add a "boolean component selector":
    // tuple{u | m.xmin, ..., u | m.zmax} | tuple{selector(true), selector(false))} such
    // that the assignment to a selector(false) is a noop while assignment to a
    // selector(true) is passed through...
    // would need something similar for selecting objects within R that match object id's
    // or bc's

    // only works for dirichlet right how
    u | sel::R = bvals;
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

system_size heat::size() const { return {1, 0, m.ss()}; }

} // namespace ccs::systems
