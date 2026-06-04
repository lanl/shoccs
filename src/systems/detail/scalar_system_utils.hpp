#pragma once

#include "fields/expr.hpp"
#include "fields/scalar.hpp"
#include "fields/selection_desc.hpp"
#include "io/field_io.hpp"
#include "mesh/mesh.hpp"
#include "temporal/step_controller.hpp"

namespace ccs::systems::detail
{

// Evaluate func(loc) at every mesh location, storing results in out.
// When parallel=true, uses Kokkos::parallel_for for D and R buffers.
// When parallel=false (e.g. Lua MMS which is not thread-safe), uses serial loops.
inline void eval_at_locations(const mesh& m, auto&& func, scalar_span out,
                              bool parallel = true)
{
    const auto* xv = m.x().data();
    const auto* yv = m.y().data();
    const auto* zv = m.z().data();
    int nx = (int)m.x().size(), ny = (int)m.y().size(), nz = (int)m.z().size();

    if (parallel) {
        // D-buffer: flat parallel_for over cartesian product of x, y, z
        auto* d = out.D.data();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, nx * ny * nz),
            [=, &func](int idx) {
                int i = idx / (ny * nz);
                int j = (idx / nz) % ny;
                int k = idx % nz;
                d[idx] = func(real3{xv[i], yv[j], zv[k]});
            });

        // Rx buffer
        const auto* rx_data = m.Rx().data();
        auto* rx_out = out.Rx.data();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, (int)m.Rx().size()),
            [=, &func](int i) { rx_out[i] = func(rx_data[i].position); });

        // Ry buffer
        const auto* ry_data = m.Ry().data();
        auto* ry_out = out.Ry.data();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, (int)m.Ry().size()),
            [=, &func](int i) { ry_out[i] = func(ry_data[i].position); });

        // Rz buffer
        const auto* rz_data = m.Rz().data();
        auto* rz_out = out.Rz.data();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, (int)m.Rz().size()),
            [=, &func](int i) { rz_out[i] = func(rz_data[i].position); });

        Kokkos::fence();
    } else {
        // Serial fallback for non-thread-safe callables (e.g. Lua MMS)
        for (int idx = 0; idx < nx * ny * nz; ++idx) {
            int i = idx / (ny * nz);
            int j = (idx / nz) % ny;
            int k = idx % nz;
            out.D[idx] = func(real3{xv[i], yv[j], zv[k]});
        }
        for (size_t i = 0; i < m.Rx().size(); ++i)
            out.Rx[i] = func(m.Rx()[i].position);
        for (size_t i = 0; i < m.Ry().size(); ++i)
            out.Ry[i] = func(m.Ry()[i].position);
        for (size_t i = 0; i < m.Rz().size(); ++i)
            out.Rz[i] = func(m.Rz()[i].position);
    }
}

// Compute Linf error, min/max, and per-component stats for a scalar field
// against an exact solution. Used by both heat::stats() and scalar_wave::stats().
inline system_stats compute_scalar_stats(const mesh& m,
                                         const bcs::Object& object_bcs,
                                         scalar_view u,
                                         scalar_view sol)
{
    // Compute min/max and per-component error over fluid D indices
    const auto fd = m.fluid_desc();
    const real* u_D = u.D.data();
    const real* sol_D = sol.D.data();

    // MinMax reduction for u_min/u_max
    Kokkos::MinMaxScalar<real> minmax_result;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, fd.count()),
        KOKKOS_LAMBDA(int k, Kokkos::MinMaxScalar<real>& update) {
            int i = fd.element(k);
            if (u_D[i] < update.min_val) update.min_val = u_D[i];
            if (u_D[i] > update.max_val) update.max_val = u_D[i];
        },
        Kokkos::MinMax<real>(minmax_result));

    real u_min = fd.count() > 0 ? minmax_result.min_val : 0.0;
    real u_max = fd.count() > 0 ? minmax_result.max_val : 0.0;

    // MaxLoc reduction for err_d/err_d_idx
    Kokkos::ValLocScalar<real, int> maxloc_result;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, fd.count()),
        KOKKOS_LAMBDA(int k, Kokkos::ValLocScalar<real, int>& update) {
            int i = fd.element(k);
            real e = Kokkos::abs(u_D[i] - sol_D[i]);
            if (e > update.val) {
                update.val = e;
                update.loc = i;
            }
        },
        Kokkos::MaxLoc<real, int>(maxloc_result));

    real err_d = fd.count() > 0 ? maxloc_result.val : 0.0;
    real err_d_idx = fd.count() > 0 ? (real)maxloc_result.loc : 0.0;
    Kokkos::fence();

    // Per-component stats for Rx/Ry/Rz over non-dirichlet object indices
    std::span<const real> u_Rs[] = {u.Rx, u.Ry, u.Rz};
    std::span<const real> sol_Rs[] = {sol.Rx, sol.Ry, sol.Rz};
    real comp_errs[3] = {0.0, 0.0, 0.0};
    real comp_idxs[3] = {0.0, 0.0, 0.0};

    for (int dir = 0; dir < 3; ++dir) {
        auto nd = m.non_dirichlet_object_desc(dir, object_bcs);
        if (nd.count() == 0) continue;

        const real* u_R_ptr = u_Rs[dir].data();
        const real* sol_R_ptr = sol_Rs[dir].data();

        // MinMax reduction for this R component
        Kokkos::MinMaxScalar<real> r_minmax;
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<execution_space>(0, nd.count()),
            KOKKOS_LAMBDA(int k, Kokkos::MinMaxScalar<real>& update) {
                int i = nd.element(k);
                if (u_R_ptr[i] < update.min_val) update.min_val = u_R_ptr[i];
                if (u_R_ptr[i] > update.max_val) update.max_val = u_R_ptr[i];
            },
            Kokkos::MinMax<real>(r_minmax));

        u_min = std::min(u_min, r_minmax.min_val);
        u_max = std::max(u_max, r_minmax.max_val);

        // MaxLoc reduction for error
        Kokkos::ValLocScalar<real, int> r_maxloc;
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<execution_space>(0, nd.count()),
            KOKKOS_LAMBDA(int k, Kokkos::ValLocScalar<real, int>& update) {
                int i = nd.element(k);
                real e = Kokkos::abs(u_R_ptr[i] - sol_R_ptr[i]);
                if (e > update.val) {
                    update.val = e;
                    update.loc = i;
                }
            },
            Kokkos::MaxLoc<real, int>(r_maxloc));

        comp_errs[dir] = r_maxloc.val;
        comp_idxs[dir] = (real)r_maxloc.loc;
    }
    Kokkos::fence();

    real err_rx = comp_errs[0], idx_rx = comp_idxs[0];
    real err_ry = comp_errs[1], idx_ry = comp_idxs[1];
    real err_rz = comp_errs[2], idx_rz = comp_idxs[2];

    real err = std::max({err_d, err_rx, err_ry, err_rz});
    return system_stats{.stats = {err,
                                  u_min,
                                  u_max,
                                  err_d,
                                  err_d_idx,
                                  err_rx,
                                  idx_rx,
                                  err_ry,
                                  idx_ry,
                                  err_rz,
                                  idx_rz}};
}

// Initialize a scalar field from an evaluated solution: zero D, assign fluid
// indices from sol, copy R buffers. Used by heat::initialize() and
// scalar_wave::initialize().
inline void initialize_scalar_field(const mesh& m, scalar_span u, scalar_span sol)
{
    // Fill D with zeros via parallel_for
    real* u_D = u.D.data();
    int u_D_size = (int)u.D.size();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, u_D_size),
        KOKKOS_LAMBDA(int i) { u_D[i] = 0.0; });

    // Copy sol at fluid indices
    const auto fd = m.fluid_desc();
    assign_selected(u_D, fd, handle_expr{sol.D.data()});

    // Copy sol's R components to u's R components via parallel_for
    real* u_Rx = u.Rx.data();
    const real* sol_Rx_ptr = sol.Rx.data();
    int rx_size = (int)u.Rx.size();
    real* u_Ry = u.Ry.data();
    const real* sol_Ry_ptr = sol.Ry.data();
    int ry_size = (int)u.Ry.size();
    real* u_Rz = u.Rz.data();
    const real* sol_Rz_ptr = sol.Rz.data();
    int rz_size = (int)u.Rz.size();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, rx_size),
        KOKKOS_LAMBDA(int i) { u_Rx[i] = sol_Rx_ptr[i]; });
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, ry_size),
        KOKKOS_LAMBDA(int i) { u_Ry[i] = sol_Ry_ptr[i]; });
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, rz_size),
        KOKKOS_LAMBDA(int i) { u_Rz[i] = sol_Rz_ptr[i]; });
    Kokkos::fence();
}

// Compute |u - sol| error at fluid/non-dirichlet indices, zero Dirichlet
// entries, and write scalar fields to IO. Used by both heat::write() and
// scalar_wave::write().
inline bool write_scalar_error(const mesh& m,
                               const bcs::Object& object_bcs,
                               const bcs::Grid& grid_bcs,
                               scalar_view u,
                               scalar_view sol,
                               scalar_span error,
                               field_io& io,
                               std::span<const std::string> io_names,
                               const step_controller& c,
                               real dt)
{
    // Zero all error buffers
    real* err_d_ptr = error.D.data();
    real* err_rx_ptr = error.Rx.data();
    real* err_ry_ptr = error.Ry.data();
    real* err_rz_ptr = error.Rz.data();
    int n_d = (int)error.D.size();
    int n_rx = (int)error.Rx.size();
    int n_ry = (int)error.Ry.size();
    int n_rz = (int)error.Rz.size();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, n_d),
        KOKKOS_LAMBDA(int i) { err_d_ptr[i] = 0.0; });
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, n_rx),
        KOKKOS_LAMBDA(int i) { err_rx_ptr[i] = 0.0; });
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, n_ry),
        KOKKOS_LAMBDA(int i) { err_ry_ptr[i] = 0.0; });
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, n_rz),
        KOKKOS_LAMBDA(int i) { err_rz_ptr[i] = 0.0; });

    // Compute |u - sol| at fluid D indices
    const auto fd = m.fluid_desc();
    const real* u_D = u.D.data();
    const real* sol_D = sol.D.data();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, fd.count()),
        KOKKOS_LAMBDA(int k) {
            int i = fd.element(k);
            err_d_ptr[i] = Kokkos::abs(u_D[i] - sol_D[i]);
        });

    // Compute |u - sol| at non-dirichlet R indices
    std::span<const real> u_R[] = {u.Rx, u.Ry, u.Rz};
    std::span<const real> sol_R[] = {sol.Rx, sol.Ry, sol.Rz};
    real* err_R_ptrs[] = {err_rx_ptr, err_ry_ptr, err_rz_ptr};
    for (int dir = 0; dir < 3; ++dir) {
        auto nd = m.non_dirichlet_object_desc(dir, object_bcs);
        if (nd.count() == 0) continue;
        const real* u_R_ptr = u_R[dir].data();
        const real* sol_R_ptr = sol_R[dir].data();
        real* err_R_ptr = err_R_ptrs[dir];
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, nd.count()),
            KOKKOS_LAMBDA(int k) {
                int i = nd.element(k);
                err_R_ptr[i] = Kokkos::abs(u_R_ptr[i] - sol_R_ptr[i]);
            });
    }
    Kokkos::fence();

    // Zero Dirichlet grid faces on D buffer
    for_each_grid_bc_desc<bcs::Dirichlet>(grid_bcs, m.extents(), [&](auto desc) {
        fill_selected(err_d_ptr, desc, 0.0);
    });

    // Zero Dirichlet object entries on Rx/Ry/Rz buffers
    for (int dir = 0; dir < 3; ++dir) {
        auto gd = m.dirichlet_object_desc(dir, object_bcs);
        fill_selected(err_R_ptrs[dir], gd, 0.0);
    }

    scalar_view err_view{error.D, error.Rx, error.Ry, error.Rz};
    std::vector<scalar_view> io_scalars{u, err_view};

    return io.write(io_names, io_scalars, c, dt, m.R());
}

} // namespace ccs::systems::detail
