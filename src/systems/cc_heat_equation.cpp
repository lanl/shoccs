#include "cc_heat_equation.hpp"

// get pi
#define _USE_MATH_DEFINES
#include <cassert>
#include <cmath>

#include <iostream>

namespace pdg
{

namespace views = ranges::views;

cc_heat_equation::cc_heat_equation(cart_mesh&& cart_,
                                   mesh&& cut_mesh_,
                                   std::unique_ptr<manufactured_solution>&& ms_,
                                   discrete_operator&& lap_,
                                   field_io& io,
                                   double diffusivity_,
                                   double stats_begin_accumulate_)
    : system{std::move(cart_), std::move(cut_mesh_)},
      ms{std::move(ms_)},
      lap{std::move(lap_)},
      diffusivity{diffusivity_},
      stats0{},
      stats_begin_accumulate{stats_begin_accumulate_}
{
        assert(ms);
        // allocate mesh wide data
        u0 = cart.allocate();
        error = cart.allocate();
        lap_u.resize(cut_mesh.workspace_all());

        auto finit = [this](const auto& loc) {
                return ms->eval(0.0,
                                {std::get<0>(loc), std::get<1>(loc), std::get<2>(loc)});
        };

        // initialize all data using ms and zero inside
        u = cart.location_view<0, 1, 2>() | views::transform(finit) |
            ranges::to<std::vector>();

        auto fast_dim = cart.ndims() - 1;

        cut_mesh.fill_void(fast_dim, u, 0.0);
        for (int i = 0; i < fast_dim; ++i) {
                cut_mesh.on_unpartition_obj(
                    i,
                    u,
                    [this, dim = i, bounds = cart.size<0, 1, 2>(), &finit](
                        const auto& loc, std::ptrdiff_t idx) -> void {
                            u[rp2ru<3>(dim, idx, bounds)] = finit(loc);
                    });
        }
        cut_mesh.on_unpartition_obj(fast_dim, u, finit);

        // register 'u' and 'error' with io.  The current setup of capturing this pointer
        // means we can't invalidate the vector via moves at any point.  Ranges likes to
        // do this so we use iterators instead (for now)
        io.add("U", &u[0]);
        io.add("Error", &error[0]);
}

double cc_heat_equation::system_timestep_size(double cfl) const
{
        auto dx = std::min({cart.delta(0), cart.delta(1), cart.delta(2)});
        return cfl * dx * dx / (4.0 * diffusivity);
}

// Evaluate the rhs of the system using the current u
void cc_heat_equation::operator()(std::vector<double>& rhs, double time)
{
        cut_mesh.apply_operator(u,
                                lap_u,
                                cart,
                                lap,
                                &manufactured_solution::eval,      // dirichlet function
                                &manufactured_solution::gradienti, // neumann function
                                ms.get(),
                                time);

        dot(cart.ndims(), cart.total_size(), diffusivity, lap_u, rhs);
        // add mms source term
        for (const auto& [r, mask, loc] :
             ranges::view::zip(rhs, cut_mesh.mask(), cart.location_view<0, 1, 2>())) {
                if (mask & (cell_type::fluid | cell_type::domain_n)) {
                        auto [x, y, z] = loc;
                        r += (ms->ddt(time, {x, y, z}) -
                              diffusivity * ms->laplacian(time, {x, y, z}));
                }
        }

        const auto& mask = cut_mesh.mask();

        for (int i = 0; i < cart.ndims(); ++i) {
                cut_mesh.on_unpartition_neumann_obj(
                    i,
                    rhs,
                    [this, dim = i, bounds = cart.size<0, 1, 2>(), t = time, &mask, &rhs](
                        const std::array<double, 3>& loc, std::ptrdiff_t index) -> void {
                            auto i = rp2ru<3>(dim, index, bounds);
                            if (neumann_set(dim, mask[i])) {
                                    rhs[i] += (ms->ddt(t, loc) -
                                               diffusivity * ms->laplacian(t, loc));
                            }
                    });
        }

        cut_mesh.fill_void(cart.ndims() - 1, rhs, 0.0);
}

system_stats cc_heat_equation::stats(double time)
{
        this->stats_(stats0,
                     time >= stats_begin_accumulate,
                     time,
                     [this](double time, const auto& loc) {
                             return ms->eval(
                                 time,
                                 {std::get<0>(loc), std::get<1>(loc), std::get<2>(loc)});
                     });
        return stats0;
}

int_t cc_heat_equation::rhs_size() const { return cart.total_size(); }

std::unique_ptr<system> build_system(cart_mesh&& cart,
                                     mesh&& cut_mesh,
                                     std::unique_ptr<manufactured_solution>&& ms,
                                     discrete_operator&& lap,
                                     field_io& io,
                                     double diffusivity,
                                     double stats_begin_accumulate)
{
        return std::make_unique<cc_heat_equation>(std::move(cart),
                                                  std::move(cut_mesh),
                                                  std::move(ms),
                                                  std::move(lap),
                                                  io,
                                                  diffusivity,
                                                  stats_begin_accumulate);
}

} // namespace pdg
