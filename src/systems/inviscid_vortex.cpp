#include "inviscid_vortex.hpp"
#include "fields/selector.hpp"

#include "real3_operators.hpp"
#include <cmath>
#include <numbers>
#include <spdlog/spdlog.h>

#include <range/v3/algorithm/max.hpp>
#include <range/v3/view/transform.hpp>

namespace ccs::systems
{

constexpr real g = 1.4;
constexpr real g1 = 0.4;
constexpr real twoPi = 2 * std::numbers::pi_v<real>;

constexpr auto max_abs =
    lift([](auto&& a, auto&& b) { return std::max(std::abs(a), std::abs(b)); });
constexpr auto sqrt = lift([](auto&& x) { return std::sqrt(x); });

enum class vars { rho, rhoU, rhoV, rhoE, P };

#if 0
template <int N>
static auto const_var_spans(const std::vector<real>& v, int_t sz)
{
    return [&]<auto... I>(std::integer_sequence<int, I...>)
    {
        return std::tuple{absl::MakeConstSpan(&v[I * sz], &v[(I + 1) * sz])...};
    }
    (std::make_integer_sequence<int, N>{});
}

template <int N>
static auto var_spans(std::vector<real>& v, int_t sz)
{
    return [&]<auto... I>(std::integer_sequence<int, I...>)
    {
        return std::tuple{absl::MakeSpan(&v[I * sz], &v[(I + 1) * sz])...};
    }
    (std::make_integer_sequence<int, N>{});
}
#endif

namespace solution
{

struct rho {
    real x0;
    real y0;
    real eps;
    real M0;

    template <typename T>
    real operator()(real time, const T& location) const
    {
        auto [x, y, _] = location;

        real xh = x - x0 - time;
        real yh = y - y0;
        real f = 1.0 - (xh * xh + yh * yh);

        return std::pow(
            (1.0 - 0.5 * g1 * (eps * M0 / twoPi) * (eps * M0 / twoPi) * std::exp(f)),
            1.0 / g1);
    }
};

struct rhoU {
    real x0;
    real y0;
    real eps;
    real M0;

    template <typename T>
    real operator()(real time, const T& location) const
    {
        auto [x, y, _] = location;

        real xh = x - x0 - time;
        real yh = y - y0;
        real f2 = (1.0 - (xh * xh + yh * yh)) / 2;
        real rho_ = rho{x0, y0, eps, M0}(time, location);

        return rho_ * (1.0 - eps * yh * std::exp(f2) / twoPi);
    }
};

struct rhoV {
    real x0;
    real y0;
    real eps;
    real M0;

    template <typename T>
    real operator()(real time, const T& location) const
    {
        auto [x, y, _] = location;

        real xh = x - x0 - time;
        real yh = y - y0;
        real f = 1.0 - (xh * xh + yh * yh);
        real rho_ = rho{x0, y0, eps, M0}(time, location);

        return rho_ * eps * xh / twoPi * std::exp(0.5 * f);
    }
};

struct P {
    real x0;
    real y0;
    real eps;
    real M0;

    template <typename T>
    real operator()(real time, const T& location) const
    {
        real rho_ = rho{x0, y0, eps, M0}(time, location);

        return std::pow(rho_, g) / (g * M0 * M0);
    }
};

struct rhoE {
    real x0;
    real y0;
    real eps;
    real M0;

    template <typename T>
    real operator()(real time, const T& location) const
    {
        real rho_ = rho{x0, y0, eps, M0}(time, location);
        real rhoU_ = rhoU{x0, y0, eps, M0}(time, location);
        real rhoV_ = rhoV{x0, y0, eps, M0}(time, location);
        real P_ = P{x0, y0, eps, M0}(time, location);

        return (P_ / g1) + 0.5 * (rhoU_ * rhoU_ + rhoV_ * rhoV_) / rho_;
    }
};

} // namespace solution

#if 0
euler_vortex::euler_vortex(cart_mesh&& cart_,
                           mesh&& cut_mesh_,
                           discrete_operator&& div_,
                           field_io& io,
                           std::array<real, 2> center_,
                           real eps_,
                           real M0_,
                           real stats_begin_accumulate_)
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
            [&, this, dim = i](real time, const auto& loc, std::ptrdiff_t idx) -> void {
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
#endif

void inviscid_vortex::operator()(field&, const step_controller&) {}

system_stats
inviscid_vortex::stats(const field&, const field&, const step_controller&) const
{
    return {};
}

bool inviscid_vortex::valid(const system_stats&) const { return false; }

real inviscid_vortex::timestep_size(const field&, const step_controller&) const
{

    auto&& [rho, rhoU, rhoV, rhoE, P] =
        U.scalars(vars::rho, vars::rhoU, vars::rhoV, vars::rhoE, vars::P);

    const auto d =
        rs::max((max_abs(rhoU / rho, rhoV / rho) + sqrt(g * P / rho)) | sel::D);

    // return cfl *
    // std::min({cart.delta(0), cart.delta(1), cart.delta(2)}) / d;
    return 1 / d;
    //    return null_v<>;
};

void inviscid_vortex::rhs(field_view field, real, field_span field_rhs)
{
    auto&& [rho, rhoU, rhoV, rhoE, P] =
        field.scalars(vars::rho, vars::rhoU, vars::rhoV, vars::rhoE, vars::P);

    // rhs(rho) == - div(rho * vec(u))
    {
        auto&& rhs = field_rhs.scalars(vars::rho);
        // rhs = divergence(field::Vector{rhoU, rhoV});
    }

    // rhs(rhoU) == - dp/dx - div(rhoU * vec(U))
    {
        auto&& rhs = field_rhs.scalars(vars::rhoU);
        // rhs = divergence(field::Vector{P + rhoU * rhoU / rho, rhoU * rhoV / rho});
    }

    // rhs(rhoV) == -dp/dy - div(rhoV * vec(U))
    {
        auto&& rhs = field_rhs.scalars(vars::rhoU);
        // rhs = divergence(field::Vector{rhoV * rhoU / rho, P + rhoV * rhoV / rho});
    }

    // rhs(rhoE) == - div(vec(u) (rhoE + P))
    {
        auto&& rhs = field_rhs.scalars(vars::rhoE);
        // rhs = divergence(field::Vector{rhoU * (rhoE + P) / rho, rhoV * (rhoE + P) /
        // rho});
    }

    // field_rhs *= -1;
}

void inviscid_vortex::update_boundary(field_span, real) {}

real3 inviscid_vortex::summary(const system_stats&) const { return {}; }

void inviscid_vortex::log(const system_stats&, const step_controller&)
{
    if (auto logger = spdlog::get("system"); logger) { logger->info("InvisicdVortex"); }
}

system_size inviscid_vortex::size() const { return {}; }

#if 0
real euler_vortex::system_timestep_size(real cfl) const
{
    int_t sz = cart.total_size();
    real d = 0;
    auto [rho, rhoU, rhoV, rhoE] = const_var_spans<4>(u, sz);
    for (int_t i = 0; i < sz; ++i) {
        d = std::max({d,
                      std::abs(rhoU[i] / rho[i]) + std::sqrt(g * P[i] / rho[i]),
                      std::abs(rhoV[i] / rho[i]) + std::sqrt(g * P[i] / rho[i])});
    }
    return cfl * std::min({cart.delta(0), cart.delta(1), cart.delta(2)}) / d;
}

// Evaluate the rhs of the system using the current u
void euler_vortex::operator()(std::vector<real>& rhs, real time)
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
        [&, this, dim = 0](real time, const auto& loc, std::ptrdiff_t idx) -> void {
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

    auto rs = absl::Span<real>{rhs};
    auto div_span = absl::Span<const real>{div_u};

    // rhs(rho) = - div(rho * vec(u))
    for (int_t i = 0; i < sz; ++i) {
        wkx[i] = rhoU[i];
        wky[i] = rhoV[i];
    }

    cut_mesh.apply_vector_operator(
        work, div_u, cart, div, std::tuple{rhoU_f, rhoV_f}, null_boundary_tuple{}, time);
    dot(cart.ndims(), sz, -1.0, div_span, rs);

    // rhs(rhoU) = - dp/dx - div (rhoU (vec(u)))
    rs = rs.subspan(sz);
    for (int_t i = 0; i < sz; ++i) {
        wkx[i] = P[i] + rhoU[i] * rhoU[i] / rho[i];
        wky[i] = rhoU[i] * rhoV[i] / rho[i];
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
        wkx[i] = rhoV[i] * rhoU[i] / rho[i];
        wky[i] = P[i] + rhoV[i] * rhoV[i] / rho[i];
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
        wkx[i] = rhoU[i] * (rhoE[i] + P[i]) / rho[i];
        wky[i] = rhoV[i] * (rhoE[i] + P[i]) / rho[i];
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

system_stats euler_vortex::stats(real time)
{
    this->stats_(stats0,
                 time >= stats_begin_accumulate,
                 time,
                 solution::rho{center[0], center[1], eps, M0});
    return stats0;
}

int_t euler_vortex::rhs_size() const { return 4 * cart.total_size(); }
#endif

} // namespace ccs::systems
