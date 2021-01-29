#include "cc_elliptic.hpp"

#include "../pudding_limits.hpp"
#include <cassert>

namespace pdg
{

namespace views = ranges::views;

cc_elliptic::cc_elliptic(cart_mesh&& cart_,
                         mesh&& cut_mesh_,
                         std::unique_ptr<manufactured_solution>&& ms_,
                         coupled_discrete_operator&& lap_,
                         field_io& io)
    : system{std::move(cart_), std::move(cut_mesh_)},
      ms{std::move(ms_)},
      lap{std::move(lap_)},
      stats0{}
{
        assert(ms);
        // allocate mesh wide data
        u0 = cart.allocate();
        error = cart.allocate();
        u = cart.allocate();

        io.add("U", &u[0]);
        io.add("Error", &error[0]);
}

double cc_elliptic::system_timestep_size(double) const { return huge<double>; }

// Evaluate the rhs of the system using the current u
void cc_elliptic::operator()(std::vector<double>& rhs, double time)
{
        lap.on_boundaries(rhs,
                          &manufactured_solution::eval,      // dirichlet function
                          &manufactured_solution::gradienti, // neumann function
                          ms.get(),
                          time);
        lap(rhs, u0);

        // expand u0 to full space... temporary hack
        for (auto it = u0.begin();
             auto&& [flag, loc, v] :
             views::zip(lap.coordinates(), cart.location_view<0, 1, 2>(), u)) {
                if (flag == -1) {
                        v = ms->eval(
                            time, {std::get<0>(loc), std::get<1>(loc), std::get<2>(loc)});
                } else {
                        v = *it++;
                }
        }
}

SystemStats cc_elliptic::stats(double time)
{
        auto f = [this, time](const auto& loc) {
                return ms->laplacian(
                    time, {std::get<0>(loc), std::get<1>(loc), std::get<2>(loc)});
        };
        // baseline source terms
        auto src = cart.location_view<0, 1, 2>() | views::transform(f) |
                   ranges::to<std::vector>();

        // account for ms source terms at cut neumann boundaries
        const auto& mask = cut_mesh.mask();
        for (int i = 0; i < cart.ndims(); ++i) {
                cut_mesh.on_unpartition_neumann_obj(
                    i,
                    src,
                    [this, dim = i, bounds = cart.size<0, 1, 2>(), &mask, &src](
                        double time,
                        const std::array<double, 3>& loc,
                        std::ptrdiff_t index) -> void {
                            auto i = rp2ru<3>(dim, index, bounds);
                            if (neumann_set(dim, mask[i])) {
                                    src[i] = ms->laplacian(time, loc);
                            }
                    },
                    time);
        }

        const auto& coords = lap.coordinates();
        std::remove_if(
            src.begin(), src.end(), [start = &src[0], &coords](const double& item) {
                    std::ptrdiff_t idx = &item - start;
                    return coords[idx] == -1;
            });

        (*this)(src, time);

        this->stats_(stats0, true, time, [this](double time, const auto& loc) {
                return ms->eval(time,
                                {std::get<0>(loc), std::get<1>(loc), std::get<2>(loc)});
        });
        return stats0;
}

int_t cc_elliptic::rhs_size() const { return cart.total_size(); }

std::unique_ptr<system> build_system(cart_mesh&& cart,
                                     mesh&& cut_mesh,
                                     std::unique_ptr<manufactured_solution>&& ms,
                                     coupled_discrete_operator&& lap,
                                     field_io& io)
{
        return std::make_unique<cc_elliptic>(
            std::move(cart), std::move(cut_mesh), std::move(ms), std::move(lap), io);
}

} // namespace pdg
