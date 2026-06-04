#include "inviscid_vortex.hpp"

#include "real3_operators.hpp"
#include <cmath>
#include <numbers>
#include <spdlog/spdlog.h>

#include <algorithm>

namespace ccs::systems
{

constexpr real g = 1.4;
constexpr real g1 = 0.4;
constexpr real twoPi = 2 * std::numbers::pi_v<real>;

enum class vars { rho, rhoU, rhoV, rhoE, P };

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

bool inviscid_vortex::valid(const system_stats&) const { return false; }

real3 inviscid_vortex::summary(const system_stats&) const { return {}; }

void inviscid_vortex::log(const system_stats&, const step_controller&)
{
    if (auto logger = spdlog::get("system"); logger) { logger->info("InvisicdVortex"); }
}

system_size inviscid_vortex::size() const { return {}; }

void inviscid_vortex::rhs(const sim_registry&, field_ref,
                           sim_registry&, field_ref, real)
{
}

void inviscid_vortex::update_boundary(sim_registry&, field_ref, real) {}

real inviscid_vortex::timestep_size(const sim_registry&, field_ref,
                                     const step_controller&) const
{
    return 1.0;
}

system_stats inviscid_vortex::stats(const sim_registry&, field_ref,
                                     field_ref, const step_controller&) const
{
    return {};
}

void inviscid_vortex::initialize(sim_registry&, field_ref, const step_controller&) {}

bool inviscid_vortex::write(field_io&, const sim_registry&, field_ref,
                             const step_controller&, real)
{
    return false;
}

} // namespace ccs::systems
