#include "euler_vortex.hpp"

// get pi
#define _USE_MATH_DEFINES
#include <cassert>
#include <cmath>

namespace pdg
{

namespace views = ranges::views;

constexpr real_t g = 1.4;
constexpr real_t g1 = 0.4;
constexpr real_t tpi = 2.0 * M_PI;


template<int N>
static auto const_var_spans(const std::vector<real_t>& v, int_t sz)
{
        return [&]<auto... I>(std::integer_sequence<int, I...>)
        {
                return std::tuple{absl::MakeConstSpan(&v[I * sz], &v[(I + 1) * sz])...};
        }
        (std::make_integer_sequence<int, N>{});
}

template<int N>
static auto var_spans(std::vector<real_t>& v, int_t sz)
{
        return [&]<auto... I>(std::integer_sequence<int, I...>)
        {
                return std::tuple{absl::MakeSpan(&v[I * sz], &v[(I + 1) * sz])...};
        }
        (std::make_integer_sequence<int, N>{});
}

namespace solution
{

struct rho {
        real_t x0;
        real_t y0;
        real_t eps;
        real_t M0;

        template <typename T>
        real_t operator()(double time, const T& location) const
        {
                auto [x, y, _] = location;

                real_t xh = x - x0 - time;
                real_t yh = y - y0;
                real_t f = 1.0 - (xh * xh + yh * yh);

                return std::pow(
                    (1.0 - 0.5 * g1 * (eps * M0 / tpi) * (eps * M0 / tpi) * std::exp(f)),
                    1.0 / g1);
        }
};

struct rhoU {
        real_t x0;
        real_t y0;
        real_t eps;
        real_t M0;

        template <typename T>
        real_t operator()(double time, const T& location) const
        {
                auto [x, y, _] = location;

                real_t xh = x - x0 - time;
                real_t yh = y - y0;
                real_t f2 = (1.0 - (xh * xh + yh * yh)) / 2;
                real_t rho_ = rho{x0, y0, eps, M0}(time, location);

                return rho_ * (1.0 - eps * yh * std::exp(f2) / tpi);
        }
};

struct rhoV {
        real_t x0;
        real_t y0;
        real_t eps;
        real_t M0;

        template <typename T>
        real_t operator()(double time, const T& location) const
        {
                auto [x, y, _] = location;

                real_t xh = x - x0 - time;
                real_t yh = y - y0;
                real_t f = 1.0 - (xh * xh + yh * yh);
                real_t rho_ = rho{x0, y0, eps, M0}(time, location);

                return rho_ * eps * xh / tpi * std::exp(0.5 * f);
        }
};

struct P {
        real_t x0;
        real_t y0;
        real_t eps;
        real_t M0;

        template <typename T>
        real_t operator()(double time, const T& location) const
        {
                real_t rho_ = rho{x0, y0, eps, M0}(time, location);

                return std::pow(rho_, g) / (g * M0 * M0);
        }
};

struct rhoE {
        real_t x0;
        real_t y0;
        real_t eps;
        real_t M0;

        template <typename T>
        real_t operator()(double time, const T& location) const
        {
                real_t rho_ = rho{x0, y0, eps, M0}(time, location);
		real_t rhoU_ = rhoU{x0, y0, eps, M0}(time, location);
		real_t rhoV_ = rhoV{x0, y0, eps, M0}(time, location);
		real_t P_ = P{x0, y0, eps, M0}(time, location);

                return (P_ / g1) + 0.5 * (rhoU_ * rhoU_ + rhoV_ * rhoV_) / rho_;
        }
};


} // namespace solution

euler_vortex::euler_vortex(cart_mesh&& cart_,
                           mesh&& cut_mesh_,
                           discrete_operator&& div_,
                           field_io& io,
                           std::array<real_t, 2> center_,
                           real_t eps_,
                           real_t M0_,
                           double stats_begin_accumulate_)
    : system{std::move(cart_), std::move(cut_mesh_)},
      div{std::move(div_)},
      center{center_},
      eps{eps_},
      M0{M0_},
      stats0{},
      stats_begin_accumulate{stats_begin_accumulate_}
{
        // allocate mesh wide data
        u0 = cart.allocate(4);
	u = cart.allocate(4);
	P = cart.allocate();
	work = cart.allocate(3);
        error = cart.allocate();
        div_u = cart.allocate(3);

	const auto bounds = cart.size<0, 1, 2>();
	const auto sz = cart.total_size();

        auto [rho, rhoU, rhoV, rhoE] = var_spans<4>(u, sz);
	auto rho_f = solution::rho{center[0], center[1], eps, M0};
        auto rhoU_f = solution::rhoU{center[0], center[1], eps, M0};
	auto rhoV_f = solution::rhoV{center[0], center[1], eps, M0};
        auto rhoE_f = solution::rhoE{center[0], center[1], eps, M0};
	auto P_f = solution::P{center[0], center[1], eps, M0};


        // initialize all data
        auto xyz = cart.location_view<0, 1, 2>();
        for (int_t i = 0; i < sz; ++i) {
                rho[i] = rho_f(0.0, xyz[i]);
                rhoU[i] = rhoU_f(0.0, xyz[i]);
		rhoV[i] = rhoV_f(0.0, xyz[i]);
		rhoE[i] = rhoE_f(0.0, xyz[i]);
		P[i] = P_f(0.0, xyz[i]);
        }


        auto fast_dim = cart.ndims() - 1;
        cut_mesh.fill_void(fast_dim, u, 0.0);
	for (int i = 0; i < fast_dim; ++i) {
                cut_mesh.on_unpartition_obj(
                    i,
                    u,
                    [&, this, dim = i](
                        real_t time, const auto& loc, std::ptrdiff_t idx) -> void {
                            auto j = rp2ru<3>(dim, idx, bounds);
                            rho[j] = rho_f(time, loc);
                            rhoU[j] = rhoU_f(time, loc);
                            rhoV[j] = rhoV_f(time, loc);
                            rhoE[j] = rhoE_f(time, loc);
                            P[j] = P_f(time, loc);
                    },
                    0.0);
        }

        // register 'u' and 'error' with io.  The current setup of capturing this pointer
        // means we can't invalidate the vector via moves at any point.  Ranges likes to
        // do this so we use iterators instead (for now)
        io.add("Rho", &rho[0]);
	io.add("RhoU", &rhoU[0]);
        io.add("RhoV", &rhoV[0]);
        io.add("RhoE", &rhoE[0]);
        io.add("P", &P[0]);
        io.add("Error", &error[0]);
}

double euler_vortex::system_timestep_size(real_t cfl) const
{
        int_t sz = cart.total_size();
	real_t d = 0;
        auto [rho, rhoU, rhoV, rhoE] = const_var_spans<4>(u, sz);
	for (int_t i = 0; i < sz; ++i) {
                d = std::max({d,
                              std::abs(rhoU[i] / rho[i]) + std::sqrt(g * P[i] / rho[i]),
                              std::abs(rhoV[i] / rho[i]) + std::sqrt(g * P[i] / rho[i])});
        }
        return cfl * std::min({cart.delta(0), cart.delta(1), cart.delta(2)}) / d;
}

// Evaluate the rhs of the system using the current u
void euler_vortex::operator()(std::vector<real_t>& rhs, double time)
{

        int_t sz = cart.total_size();
	auto bounds = cart.size<0, 1, 2>();
        auto [rho, rhoU, rhoV, rhoE] = var_spans<4>(u, sz);
	auto [wkx, wky] = var_spans<2>(work, sz);

	auto rho_f = solution::rho{center[0], center[1], eps, M0};
        auto rhoU_f = solution::rhoU{center[0], center[1], eps, M0};
	auto rhoV_f = solution::rhoV{center[0], center[1], eps, M0};
	auto rhoE_f = solution::rhoE{center[0], center[1], eps, M0};
	auto P_f = solution::P{center[0], center[1], eps, M0};

	// apply inflow bcs here
        cut_mesh.on_unpartition_dirichlet_obj(
            0,
            u,
            [&, this, dim = 0](real_t time, const auto& loc, std::ptrdiff_t idx) -> void {
                    auto j = rp2ru<3>(dim, idx, bounds);
                    rho[j] = rho_f(time, loc);
                    rhoU[j] = rhoU_f(time, loc);
                    rhoV[j] = rhoV_f(time, loc);
                    rhoE[j] = rhoE_f(time, loc);
            },
            time);
	// eos
        for (int_t i = 0; i < sz; ++i) {
                auto ke = 0.5 * (rhoU[i] * rhoU[i] + rhoV[i] * rhoV[i]) / rho[i];
                P[i] = g1 * (rhoE[i] - ke);
        }

        auto rs = absl::Span<real_t>{rhs};
        auto div_span = absl::Span<const real_t>{div_u};

        // rhs(rho) = - div(rho * vec(u))
        for (int_t i = 0; i < sz; ++i) {
                wkx[i] = rhoU[i];
                wky[i] = rhoV[i];
        }

        cut_mesh.apply_vector_operator(work,
                                       div_u,
                                       cart,
                                       div,
                                       std::tuple{rhoU_f, rhoV_f},
                                       null_boundary_tuple{},
                                       time);
        dot(cart.ndims(), sz, -1.0, div_span, rs);


        // rhs(rhoU) = - dp/dx - div (rhoU (vec(u)))
        rs = rs.subspan(sz);
	for (int_t i = 0; i < sz; ++i) {
	    wkx[i] = P[i] + rhoU[i]*rhoU[i]/rho[i];
	    wky[i] = rhoU[i]*rhoV[i]/rho[i];
	}

        cut_mesh.apply_vector_operator(
            work,
            div_u,
            cart,
            div,
            std::tuple{[&](auto&&... args) {
                               return P_f(args...) +
                                      rhoU_f(args...) * rhoU_f(args...) / rho_f(args...);
                       },
                       [&](auto&&... args) {
                               return rhoU_f(args...) * rhoV_f(args...) / rho_f(args...);
                       }},
            null_boundary_tuple{},
            time);

        dot(cart.ndims(), sz, -1.0, div_span, rs);

	// rhs(rhoV) = - dp/dy - div (rhoV (vec(u)))
        rs = rs.subspan(sz);
	for (int_t i = 0; i < sz; ++i) {
	    wkx[i] = rhoV[i]*rhoU[i]/rho[i];
	    wky[i] = P[i]+rhoV[i]*rhoV[i]/rho[i];
	}

        cut_mesh.apply_vector_operator(
            work,
            div_u,
            cart,
            div,
            std::tuple{[&](auto&&... args) {
                               return rhoV_f(args...) * rhoU_f(args...) / rho_f(args...);
                       },
                       [&](auto&&... args) {
                               return P_f(args...) +
                                      rhoV_f(args...) * rhoV_f(args...) / rho_f(args...);
                       }},
            null_boundary_tuple{},
            time);

        dot(cart.ndims(), sz, -1.0, div_span, rs);

        // rhs(rhoE) = - div ((vec(u)(rhoE + P)))
        rs = rs.subspan(sz);
	for (int_t i = 0; i < sz; ++i) {
	    wkx[i] = rhoU[i]*(rhoE[i]+P[i])/rho[i];
	    wky[i] = rhoV[i]*(rhoE[i]+P[i])/rho[i];
	}

        cut_mesh.apply_vector_operator(
            work,
            div_u,
            cart,
            div,
            std::tuple{[&](auto&&... args) {
                               return rhoU_f(args...) * (rhoE_f(args...) + P_f(args...)) /
                                      rho_f(args...);
                       },
                       [&](auto&&... args) {
                               return rhoV_f(args...) * (rhoE_f(args...) + P_f(args...)) /
                                      rho_f(args...);
                       }},
            null_boundary_tuple{},
            time);

        dot(cart.ndims(), sz, -1.0, div_span, rs);
}

system_stats euler_vortex::stats(double time)
{
        this->stats_(stats0,
                     time >= stats_begin_accumulate,
                     time,
                     solution::rho{center[0], center[1], eps, M0});
        return stats0;
}

int_t euler_vortex::rhs_size() const { return 4 * cart.total_size(); }

std::unique_ptr<system> build_system(cart_mesh&& cart,
                                     mesh&& cut_mesh,
                                     discrete_operator&& grad,
                                     field_io& io,
                                     std::array<real_t, 2> center,
                                     real_t eps,
                                     real_t M0,
                                     double stats_begin_accumulate)
{
        return std::make_unique<euler_vortex>(std::move(cart),
                                              std::move(cut_mesh),
                                              std::move(grad),
                                              io,
                                              center,
                                              eps,
					      M0,
                                              stats_begin_accumulate);
}

} // namespace pdg
