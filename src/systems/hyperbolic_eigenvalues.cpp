#include "hyperbolic_eigenvalues.hpp"

#include <sol/sol.hpp>

#include <spdlog/sinks/basic_file_sink.h>

#include "operators/discrete_operator.hpp"
#include "operators/eigenvalue_visitor.hpp"

#include <range/v3/algorithm/max.hpp>

namespace ccs::systems
{

hyperbolic_eigenvalues::hyperbolic_eigenvalues(mesh&& m,
                                               bcs::Grid&& grid_bcs,
                                               bcs::Object&& object_bcs,
                                               stencil st,
                                               const logs& build_logger)
    : m{MOVE(m)},
      grid_bcs{MOVE(grid_bcs)},
      object_bcs{MOVE(object_bcs)},
      grad{gradient(this->m, st, this->grid_bcs, this->object_bcs, build_logger)},
      logger{build_logger, "system", "system.csv"}
{

    logger.set_pattern("%v");
    logger(spdlog::level::info, "Timestamp,MaxEigen");
    logger.set_pattern("%Y-%m-%d %H:%M:%S.%f,%v");
}

//
// Compute the max eigenvalues
//
system_stats
hyperbolic_eigenvalues::stats(const field&, const field&, const step_controller&) const
{

    auto p = m.Rx() | vs::transform([this](auto&& info) {
                 return object_bcs[info.shape_id] == bcs::Dirichlet;
             });
    auto v = eigenvalue_visitor{m.extents(), p, std::vector<bool>{}, std::vector<bool>{}};
    grad.visit(v);

    return system_stats{.stats = {-m.h(0) * rs::min(v.eigenvalues_real())}};
}

real3 hyperbolic_eigenvalues::summary(const system_stats& stats) const
{
    return {stats.stats[0], 0.0, 0.0};
}

void hyperbolic_eigenvalues::log(const system_stats& stats, const step_controller&)
{
    logger(spdlog::level::info, "{}", stats.stats[0]);
}

system_size hyperbolic_eigenvalues::size() const { return {0, 0, m.ss()}; }

std::optional<hyperbolic_eigenvalues>
hyperbolic_eigenvalues::from_lua(const sol::table& tbl, const logs& logger)
{
    auto mesh_opt = mesh::from_lua(tbl, logger);
    if (!mesh_opt) return std::nullopt;

    auto bc_opt = bcs::from_lua(tbl, mesh_opt->extents(), logger);
    auto st_opt = stencil::from_lua(tbl, logger);

    if (bc_opt && st_opt)
        return hyperbolic_eigenvalues{
            MOVE(*mesh_opt), MOVE(bc_opt->first), MOVE(bc_opt->second), *st_opt, logger};
    else
        return std::nullopt;
}

//
// Dummy routines
//
real hyperbolic_eigenvalues::timestep_size(const field&, const step_controller&) const
{
    return 1.0;
}

void hyperbolic_eigenvalues::operator()(field&, const step_controller&) {}

bool hyperbolic_eigenvalues::valid(const system_stats&) const { return true; }

void hyperbolic_eigenvalues::rhs(field_view, real, field_span) {}

void hyperbolic_eigenvalues::update_boundary(field_span, real) {}

bool hyperbolic_eigenvalues::write(field_io&, field_view, const step_controller&, real)
{
    return true;
}

} // namespace ccs::systems
