#include "heat.hpp"
#include "detail/scalar_system_utils.hpp"
#include "fields/expr.hpp"
#include "fields/selection_desc.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include <fmt/ranges.h>
#include <sol/sol.hpp>

#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <iterator>

namespace ccs::systems
{

using detail::eval_at_locations;

heat::heat(mesh&& m,
           bcs::Grid&& grid_bcs,
           bcs::Object&& object_bcs,
           manufactured_solution&& m_sol,
           stencil st,
           real diffusivity,
           const logs& build_logger)
    : m{MOVE(m)},
      grid_bcs{MOVE(grid_bcs)},
      object_bcs{MOVE(object_bcs)},
      m_sol{MOVE(m_sol)},
      lap{this->m, st, this->grid_bcs, this->object_bcs, build_logger},
      diffusivity{diffusivity},
      neumann_d(this->m.size()), neumann_rx(this->m.Rx().size()),
      neumann_ry(this->m.Ry().size()), neumann_rz(this->m.Rz().size()),
      src_d(this->m.size()), src_rx(this->m.Rx().size()),
      src_ry(this->m.Ry().size()), src_rz(this->m.Rz().size()),
      error_d(this->m.size()), error_rx(this->m.Rx().size()),
      error_ry(this->m.Ry().size()), error_rz(this->m.Rz().size()),
      logger{build_logger, "system", "system.csv"}
{
    assert(!!(this->m_sol));

    logger.set_pattern("%v");
    logger(spdlog::level::info,
           "Timestamp,Time,Step,Linf,Min,Max,Domain_Linf,Domain_ic,Rx_Linf,Rx_ic,Ry_"
           "Linf,Ry_ic,Rz_Linf,Rz_ic,Wall_ms");

    logger.set_pattern("%Y-%m-%d %H:%M:%S.%f,%v");
}


bool heat::valid(const system_stats& stats) const
{
    const auto& v = stats.stats[0];
    return std::isfinite(v) && std::abs(v) <= 1e6;
}

void heat::log(const system_stats& stats, const step_controller& step)
{
    logger(spdlog::level::info,
           "{},{},{},{:.3f}",
           (real)step,
           (int)step,
           fmt::join(stats.stats, ","),
           stats.wall_time_s * 1000.0);
}

//
// Convert the system statistics into a real3 summary
//
real3 heat::summary(const system_stats& stats) const
{
    return {stats.stats[0], stats.stats[1], stats.stats[2]};
}

std::optional<heat> heat::from_lua(const sol::table& tbl, const logs& logger)
{
    // assume we can only get here if simulation.system.type == "heat" so check
    // for the rest
    real diff = tbl["system"]["diffusivity"].get_or(1.0);

    auto mesh_opt = mesh::from_lua(tbl, logger);
    if (!mesh_opt) return std::nullopt;

    auto bc_opt = bcs::from_lua(tbl, mesh_opt->extents(), logger);
    auto st_opt = stencil::from_lua(tbl, logger);

    if (bc_opt && st_opt) {
        auto ms_opt = manufactured_solution::from_lua(tbl, mesh_opt->dims(), logger);
        auto t = ms_opt ? MOVE(*ms_opt) : manufactured_solution{};

        return heat{MOVE(*mesh_opt),
                    MOVE(bc_opt->first),
                    MOVE(bc_opt->second),
                    MOVE(t),
                    *st_opt,
                    diff,
                    logger};
    }

    return std::nullopt;
}

system_size heat::size() const
{
    return {1, 0, m.size(), (integer)m.Rx().size(), (integer)m.Ry().size(), (integer)m.Rz().size()};
}

void heat::fill_source(real time)
{
    scalar_span src{src_d, src_rx, src_ry, src_rz};
    eval_at_locations(m, [&](const real3& loc) {
        return m_sol.ddt(time, loc) - diffusivity * m_sol.laplacian(time, loc);
    }, src, m_sol.is_thread_safe());
}

void heat::rhs(const sim_registry& reg, field_ref input,
               sim_registry& out_reg, field_ref output, real time)
{
    Kokkos::Profiling::ScopedRegion region("heat::rhs");
    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_view(reg, input, sh);
    auto u_rhs = extract_scalar_span(out_reg, output, sh);

    // rhs = diffusivity * lap(u) + (dS/dt - diffusivity * lap(S))
    u_rhs = lap(u, scalar_view{neumann_d, neumann_rx, neumann_ry, neumann_rz});
    times_assign_scalar(out_reg, output, sh, diffusivity);

    if (m_sol) {
        // Evaluate source expression into member buffers
        fill_source(time);
        scalar_span src{src_d, src_rx, src_ry, src_rz};

        // Shared destination pointers
        real* rhs_D = out_reg.data(output, sh.D());
        auto R = sh.R();

        // Fluid on D buffer: plus_assign from gather_selection of fluid indices
        plus_assign_selected(rhs_D, m.fluid_desc(), handle_expr{src.D.data()});

        // Non-dirichlet objects on Rx/Ry/Rz buffers
        real* src_R[] = {src.Rx.data(), src.Ry.data(), src.Rz.data()};
        for (int dir = 0; dir < 3; ++dir) {
            auto gd = m.non_dirichlet_object_desc(dir, object_bcs);
            plus_assign_selected(out_reg.data(output, R[dir]), gd,
                                 handle_expr{src_R[dir]});
        }

        // Grid Dirichlet: fill plane subsets of D buffer with zero
        for_each_grid_bc_desc<bcs::Dirichlet>(grid_bcs, m.extents(), [&](auto desc) {
            fill_selected(rhs_D, desc, 0.0);
        });

        // Object Dirichlet: fill predicate subsets of Rx/Ry/Rz buffers
        for (int dir = 0; dir < 3; ++dir) {
            auto gd = m.dirichlet_object_desc(dir, object_bcs);
            fill_selected(out_reg.data(output, R[dir]), gd, 0.0);
        }
    }
}

void heat::build_rhs_graph(scalar_view u, scalar_span du)
{
    scalar_view nu{neumann_d, neumann_rx, neumann_ry, neumann_rz};
    const real k = diffusivity;

    // Extract du pointers and sizes for graph node lambdas
    real* d_ptr = du.D.data();
    real* rx_ptr = du.Rx.data();
    real* ry_ptr = du.Ry.data();
    real* rz_ptr = du.Rz.data();
    const int n_d = static_cast<int>(du.D.size());
    const int n_rx = static_cast<int>(du.Rx.size());
    const int n_ry = static_cast<int>(du.Ry.size());
    const int n_rz = static_cast<int>(du.Rz.size());

    // Pre-compute source pointers (stable member data)
    real* src_d_ptr = src_d.data();
    real* src_rx_ptr = src_rx.data();
    real* src_ry_ptr = src_ry.data();
    real* src_rz_ptr = src_rz.data();

    // Pre-compute descriptors for source scatter and BC fill
    gather_selection fluid = m.fluid_desc();
    gather_selection nd_rx = m.non_dirichlet_object_desc(0, object_bcs);
    gather_selection nd_ry = m.non_dirichlet_object_desc(1, object_bcs);
    gather_selection nd_rz = m.non_dirichlet_object_desc(2, object_bcs);

    // Flatten all Dirichlet grid face indices into a single gather_selection
    gather_selection dir_d;
    {
        std::vector<int> indices;
        for_each_grid_bc_desc<bcs::Dirichlet>(grid_bcs, m.extents(), [&](auto desc) {
            for (int i = 0; i < desc.count(); ++i)
                indices.push_back(desc.element(i));
        });
        Kokkos::View<int*, memory_space> idx("dir_d_idx", indices.size());
        auto h = Kokkos::create_mirror_view(idx);
        for (size_t i = 0; i < indices.size(); ++i)
            h(i) = indices[i];
        Kokkos::deep_copy(idx, h);
        dir_d = gather_selection{idx};
    }

    gather_selection dir_rx = m.dirichlet_object_desc(0, object_bcs);
    gather_selection dir_ry = m.dirichlet_object_desc(1, object_bcs);
    gather_selection dir_rz = m.dirichlet_object_desc(2, object_bcs);

    bool has_sol = !!m_sol;

    rhs_graph_ = Kokkos::Experimental::create_graph<execution_space>(
        [&](auto root) {
            using rp_t = Kokkos::RangePolicy<execution_space>;

            // 1. Laplacian: zeros du, then accumulates dx + dy + dz with Neumann
            auto lap_done = lap.add_graph_nodes(root, u, nu, du);

            // 2. Scale all 4 buffers by diffusivity
            auto s_d = lap_done.then_parallel_for(
                "heat_scale_D", rp_t(0, n_d),
                KOKKOS_LAMBDA(int i) { d_ptr[i] *= k; });
            auto s_rx = lap_done.then_parallel_for(
                "heat_scale_Rx", rp_t(0, n_rx),
                KOKKOS_LAMBDA(int i) { rx_ptr[i] *= k; });
            auto s_ry = lap_done.then_parallel_for(
                "heat_scale_Ry", rp_t(0, n_ry),
                KOKKOS_LAMBDA(int i) { ry_ptr[i] *= k; });
            auto s_rz = lap_done.then_parallel_for(
                "heat_scale_Rz", rp_t(0, n_rz),
                KOKKOS_LAMBDA(int i) { rz_ptr[i] *= k; });

            if (!has_sol) return;

            // 3. Source scatter: plus_assign at selected indices
            auto src_d_node = s_d.then_parallel_for(
                "heat_src_D", rp_t(0, fluid.count()),
                KOKKOS_LAMBDA(int i) {
                    int idx = fluid.element(i);
                    d_ptr[idx] += src_d_ptr[idx];
                });
            auto src_rx_node = s_rx.then_parallel_for(
                "heat_src_Rx", rp_t(0, nd_rx.count()),
                KOKKOS_LAMBDA(int i) {
                    int idx = nd_rx.element(i);
                    rx_ptr[idx] += src_rx_ptr[idx];
                });
            auto src_ry_node = s_ry.then_parallel_for(
                "heat_src_Ry", rp_t(0, nd_ry.count()),
                KOKKOS_LAMBDA(int i) {
                    int idx = nd_ry.element(i);
                    ry_ptr[idx] += src_ry_ptr[idx];
                });
            auto src_rz_node = s_rz.then_parallel_for(
                "heat_src_Rz", rp_t(0, nd_rz.count()),
                KOKKOS_LAMBDA(int i) {
                    int idx = nd_rz.element(i);
                    rz_ptr[idx] += src_rz_ptr[idx];
                });

            // 4. BC fill: zero Dirichlet indices
            // D: grid Dirichlet faces
            if (dir_d.count() > 0) {
                src_d_node.then_parallel_for(
                    "heat_fill_dir_D", rp_t(0, dir_d.count()),
                    KOKKOS_LAMBDA(int i) { d_ptr[dir_d.element(i)] = 0; });
            }
            // Rx/Ry/Rz: object Dirichlet
            if (dir_rx.count() > 0) {
                src_rx_node.then_parallel_for(
                    "heat_fill_dir_Rx", rp_t(0, dir_rx.count()),
                    KOKKOS_LAMBDA(int i) { rx_ptr[dir_rx.element(i)] = 0; });
            }
            if (dir_ry.count() > 0) {
                src_ry_node.then_parallel_for(
                    "heat_fill_dir_Ry", rp_t(0, dir_ry.count()),
                    KOKKOS_LAMBDA(int i) { ry_ptr[dir_ry.element(i)] = 0; });
            }
            if (dir_rz.count() > 0) {
                src_rz_node.then_parallel_for(
                    "heat_fill_dir_Rz", rp_t(0, dir_rz.count()),
                    KOKKOS_LAMBDA(int i) { rz_ptr[dir_rz.element(i)] = 0; });
            }
        });

    rhs_graph_->instantiate();
}

void heat::submit_rhs_graph()
{
    rhs_graph_->submit();
    Kokkos::fence("heat::submit_rhs_graph() complete");
}

void heat::update_boundary(sim_registry& reg, field_ref ref, real time)
{
    Kokkos::Profiling::ScopedRegion region("heat::update_boundary");
    constexpr auto sh = scalar_handle{0};
    // Evaluate manufactured solution at all mesh locations
    std::vector<real> sol_d(m.size());
    std::vector<real> sol_rx(m.Rx().size());
    std::vector<real> sol_ry(m.Ry().size());
    std::vector<real> sol_rz(m.Rz().size());
    scalar_span sol{sol_d, sol_rx, sol_ry, sol_rz};
    eval_at_locations(m, [&](const real3& loc) {
        return m_sol(time, loc);
    }, sol, m_sol.is_thread_safe());

    // Grid Dirichlet: assign plane subsets of D buffer
    real* u_D = reg.data(ref, sh.D());
    for_each_grid_bc_desc<bcs::Dirichlet>(grid_bcs, m.extents(), [&](auto desc) {
        assign_selected(u_D, desc, handle_expr{sol.D.data()});
    });

    // Object Dirichlet: assign predicate subsets of Rx/Ry/Rz buffers
    auto R = sh.R();
    real* sol_R[] = {sol.Rx.data(), sol.Ry.data(), sol.Rz.data()};
    for (int dir = 0; dir < 3; ++dir) {
        auto gd = m.dirichlet_object_desc(dir, object_bcs);
        assign_selected(reg.data(ref, R[dir]), gd, handle_expr{sol_R[dir]});
    }

    // Set Neumann BCs: evaluate gradient component at domain locations, assign at faces
    scalar_span neu{neumann_d, neumann_rx, neumann_ry, neumann_rz};
    auto ext = m.extents();
    for (int dir = 0; dir < 3; ++dir) {
        bool need_left = grid_bcs[dir].left == bcs::Neumann;
        bool need_right = grid_bcs[dir].right == bcs::Neumann;
        if (!need_left && !need_right) continue;

        std::vector<real> grad_d(m.size());
        std::vector<real> grad_rx(m.Rx().size());
        std::vector<real> grad_ry(m.Ry().size());
        std::vector<real> grad_rz(m.Rz().size());
        scalar_span grad{grad_d, grad_rx, grad_ry, grad_rz};
        eval_at_locations(m, [&](const real3& loc) {
            return m_sol.gradient(time, loc)[dir];
        }, grad, m_sol.is_thread_safe());
        auto src = handle_expr{grad_d.data()};

        auto assign_face = [&](int face_idx) {
            if (dir == 0)
                assign_selected(neu.D.data(), make_x_plane_desc(ext, face_idx), src);
            else if (dir == 1)
                assign_selected(neu.D.data(), make_y_plane_desc(ext, face_idx), src);
            else
                assign_selected(neu.D.data(), make_z_plane_desc(ext, face_idx), src);
        };

        if (need_left) assign_face(0);
        if (need_right) assign_face(ext[dir] - 1);
    }
}

real heat::timestep_size(const sim_registry&, field_ref,
                         const step_controller& step) const
{
    const auto h_min = std::ranges::min(m.h());
    return step.parabolic_cfl() * h_min * h_min / (4 * diffusivity);
}

system_stats heat::stats(const sim_registry& reg, field_ref /*u0*/,
                          field_ref u1, const step_controller& step) const
{
    Kokkos::Profiling::ScopedRegion region("heat::stats");
    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_view(reg, u1, sh);

    // Evaluate manufactured solution at all mesh locations
    std::vector<real> sol_d(m.size());
    std::vector<real> sol_rx(m.Rx().size());
    std::vector<real> sol_ry(m.Ry().size());
    std::vector<real> sol_rz(m.Rz().size());
    scalar_span sol{sol_d, sol_rx, sol_ry, sol_rz};
    eval_at_locations(m, [&](const real3& loc) {
        return m_sol(step.simulation_time(), loc);
    }, sol, m_sol.is_thread_safe());

    return detail::compute_scalar_stats(m, object_bcs, u,
        scalar_view{sol_d, sol_rx, sol_ry, sol_rz});
}

void heat::initialize(sim_registry& reg, field_ref ref, const step_controller& c)
{
    if (!m_sol) return;

    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_span(reg, ref, sh);

    // Evaluate manufactured solution at all mesh locations
    std::vector<real> sol_d(m.size());
    std::vector<real> sol_rx(m.Rx().size());
    std::vector<real> sol_ry(m.Ry().size());
    std::vector<real> sol_rz(m.Rz().size());
    scalar_span sol{sol_d, sol_rx, sol_ry, sol_rz};
    eval_at_locations(m, [&](const real3& loc) {
        return m_sol(c.simulation_time(), loc);
    }, sol, m_sol.is_thread_safe());

    detail::initialize_scalar_field(m, u, sol);
}

bool heat::write(field_io& io, const sim_registry& reg, field_ref ref,
                 const step_controller& c, real dt)
{
    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_view(reg, ref, sh);

    // Evaluate manufactured solution at all mesh locations
    std::vector<real> sol_d(m.size());
    std::vector<real> sol_rx(m.Rx().size());
    std::vector<real> sol_ry(m.Ry().size());
    std::vector<real> sol_rz(m.Rz().size());
    scalar_span sol{sol_d, sol_rx, sol_ry, sol_rz};
    eval_at_locations(m, [&](const real3& loc) {
        return m_sol(c.simulation_time(), loc);
    }, sol, m_sol.is_thread_safe());

    scalar_span error{error_d, error_rx, error_ry, error_rz};
    return detail::write_scalar_error(m, object_bcs, grid_bcs, u,
        scalar_view{sol_d, sol_rx, sol_ry, sol_rz}, error,
        io, io_names, c, dt);
}

} // namespace ccs::systems
