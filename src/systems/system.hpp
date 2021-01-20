#pragma once

#include "../cart_mesh.hpp"
#include "../io/field_io.hpp"
#include "../mesh.hpp"
#include "../mms/manufactured_solutions.hpp"
#include "../operators/discrete_operator.hpp"
#include <array>
#include <vector>

namespace ccs
{

template <auto N>
struct system_stats {
        std::array<real, N> u_min;
        std::array<real, N> u_max;
        std::array<real, N> Linf;
        std::array<real, N> L2;
        // accumulated error norms
        std::array<real, N> Linf_acc;
        std::array<real, N> L2_acc;
        std::array<real, N> n_acc;
};

// Public API of a `system'
//
// timestep -> computes an acceptable timestep based on current solution state
//
// update -> given a rhs and timestep, update the solution state
//
// prestep -> call before the start of a timestep to precompute things
//
// valid -> returns true if the solution is in a valid state (may be overridden by
// children)
//
// call operator -> evaluates the rhs of the system at the given time
//
// stats -> computes statistics of the system at its present state
//
// log -> given a logger, write system specific information
//
// current_solution/error
//

// May be overridden by child class
//
// valid
// log

// Must be define by child
// private:
//      system_timestep_size
// public:
//      call operator
//      stats -> the corresponding routine in child class should make use of stats_ in
//              parent

// the system of pdes to solve is in this class
class system
{
    protected:
        cart_mesh cart;
        mesh cut_mesh;
        std::vector<double> u0; // Initial value of u at the start of a timestep
        std::vector<double> u;  // Currently evolving value of u
        std::vector<double> error;

        // need to compute u_min/max and linf of the error
        template <typename F>
        void stats_(system_stats& stats, bool accumulate, double time, F&& sol)
        {
                double u_min = std::numeric_limits<double>::max();
                double u_max = std::numeric_limits<double>::min();

                const auto fast_dim = cart.ndims() - 1;
                for (int i = 0; i < cart.ndims(); ++i)
                        cut_mesh.on_unpartition_dirichlet_obj(
                            i,
                            u,
                            [this, dim = i, bounds = cart.size<0, 1, 2>(), &sol](
                                double time,
                                const auto& loc,
                                std::ptrdiff_t index) -> void {
                                    auto i = rp2ru<3>(dim, index, bounds);
                                    u[i] = sol(time, loc);
                            },
                            time);

                cut_mesh.on_unpartition(fast_dim, u, [&u_min, &u_max](double u) {
                        u_min = std::min(u_min, u);
                        u_max = std::max(u_max, u);
                });

                for (const auto& [i, it] :
                     ranges::views::zip(cut_mesh.mask(), cart.location_view<0, 1, 2>()) |
                         ranges::views::enumerate) {
                        auto [mask, loc] = it;
                        if (mask & (cell_type::fluid | cell_type::domain_n |
                                    cell_type::domain_d)) {
                                error[i] = std::abs(u[i] - sol(time, loc));
                        }
                }
                // evaluate error on neumann boundaries
                const auto& mask = cut_mesh.mask();
                for (int i = 0; i < cart.ndims(); ++i)
                        cut_mesh.on_unpartition_neumann_obj(
                            i,
                            error,
                            [this, dim = i, bounds = cart.size<0, 1, 2>(), &sol, &mask](
                                double time,
                                const std::array<double, 3>& loc,
                                std::ptrdiff_t index) -> void {
                                    auto i = rp2ru<3>(dim, index, bounds);
                                    if (mask[i] & cell_type::object_n) {
                                            error[i] = std::abs(u[i] - sol(time, loc));
                                    }
                            },
                            time);

                cut_mesh.on_unpartition_dirichlet_obj(
                    fast_dim, error, [](auto&&...) { return 0.0; });

                double linf = std::numeric_limits<double>::min();
                double l2 = 0.0;
                double n = 0;

                cut_mesh.fill_void(fast_dim, error, 0.0);
                cut_mesh.on_unpartition(fast_dim, error, [&linf, &l2, &n](double err) {
                        ++n;
                        l2 = l2 + err * err;
                        linf = std::max(linf, err);
                });

                stats.u_min = u_min;
                stats.u_max = u_max;
                stats.Linf = linf;
                stats.L2 = std::sqrt(l2 / n);
                if (accumulate) {
                        ++stats.n_acc;
                        stats.L2_acc += stats.L2 * stats.L2;
                        stats.Linf_acc = std::max(stats.Linf_acc, stats.Linf);
                }
        }

    public:
        system(cart_mesh&& cart, mesh&& cut_mesh);

        // each system may have it's own way of limiting the timestep
        double timestep_size(double cfl, double max_step_size) const;

        void prestep();

        void update(std::vector<double>& rhs, double dtf);

        virtual bool valid(const system_stats& stats);

        virtual int_t rhs_size() const = 0;

        // evaluates the rhs of the system at a particular time
        std::vector<double> operator()(double time);

        virtual void operator()(std::vector<double>& rhs, double time) = 0;

        virtual system_stats stats(double time) = 0;

        virtual void log(std::optional<logger>& lg,
                         const system_stats& sstats,
                         int step,
                         double time) const;

        virtual ~system() = default;

        // return a copy of the current solution
        std::vector<double> current_solution() const;

        std::vector<double> current_error() const;

    private:
        virtual double system_timestep_size(double cfl) const = 0;
};

// initialization for vc_scalar_wave
std::unique_ptr<system> build_system(cart_mesh&& cart,
                                     mesh&& cut_mesh,
                                     discrete_operator&& grad,
                                     field_io& io,
                                     std::array<double, 3> center,
                                     double radius,
                                     double stats_begin_accumulate);

// cc_heat_equation
std::unique_ptr<system> build_system(cart_mesh&& cart,
                                     mesh&& cut_mesh,
                                     std::unique_ptr<manufactured_solution>&& ms,
                                     discrete_operator&& lap,
                                     field_io& io,
                                     double diffusivity,
                                     double stats_begin_accumulate);

// cc_elliptic
std::unique_ptr<system> build_system(cart_mesh&& cart,
                                     mesh&& cut_mesh,
                                     std::unique_ptr<manufactured_solution>&& ms,
                                     coupled_discrete_operator&& lap,
                                     field_io& io);

// euler_vortex
std::unique_ptr<system> build_system(cart_mesh&& cart,
                                     mesh&& cut_mesh,
                                     discrete_operator&& div,
                                     field_io& io,
                                     std::array<real_t, 2> center,
                                     real_t eps,
                                     real_t M0,
                                     double stats_begin_accumulate);

} // namespace pdg
