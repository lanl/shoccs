#include "scalar_wave.hpp"
#include "fields/algorithms.hpp"
#include "fields/selector.hpp"
#include "real3_operators.hpp"
#include <cmath>
#include <numbers>

#include <sol/sol.hpp>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include "operators/discrete_operator.hpp"

#include <range/v3/algorithm/max.hpp>
#include <range/v3/algorithm/min.hpp>
#include <range/v3/algorithm/minmax.hpp>
#include <range/v3/view/transform.hpp>

namespace ccs::systems
{
namespace
{
constexpr auto abs = lift([](auto&& x) { return std::abs(x); });
// system variables to be used in this system
enum class scalars : int { u };

constexpr real twoPI = 2 * std::numbers::pi_v<real>;

// negative gradient - coefficients of gradient
template <int I>
constexpr auto neg_G(const real3& center, real radius)
{
    return vs::transform([=](auto&& location) {
        return -(get<I>(location) - get<I>(center)) / length(location - center);
    });
}

constexpr auto solution(const real3& center, real radius, real time)
{
    return vs::transform([=](auto&& location) {
        return std::sin(twoPI * (length(location - center) - radius - time));
    });
}

} // namespace

scalar_wave::scalar_wave(mesh&& m_,
                         bcs::Grid&& grid_bcs,
                         bcs::Object&& object_bcs,
                         stencil st,
                         real3 center,
                         real radius)
    : m{MOVE(m_)},
      grid_bcs{MOVE(grid_bcs)},
      object_bcs{MOVE(object_bcs)},
      grad{gradient(this->m, st, this->grid_bcs, this->object_bcs)},
      center{center},
      radius{radius}
{

    // Initialize wave speeds
    grad_G | m.fluid = m.vxyz | tuple{neg_G<0>(center, radius),
                                      neg_G<1>(center, radius),
                                      neg_G<2>(center, radius)};
    // need to change this for outflow
    grad_G | sel::R = 0;

    logger = spdlog::basic_logger_st("system", "logs/system.csv", true);
    logger->set_pattern("%v");
    logger->info("Date,Time,Step,Linf,Min,Max");
    logger->set_pattern("%Y-%m-%d %H:%M:%S.%f,%v");
}

real scalar_wave::timestep_size(const field&, const step_controller& step) const
{
    const auto h_min = rs::min(m.h());
    return step.hyperbolic_cfl() * h_min;
}

//
// sets the field f to the solution
//
void scalar_wave::operator()(field& f, const step_controller& c)
{

    // extract the field components to initialize
    auto&& u = f.scalars(scalars::u);
    auto sol = m.xyz | solution(center, radius, c);

    u | sel::D = 0;
    u | m.fluid = sol;
    u | sel::R = sol;
}

//
// Compute the linf error as well as the min/max of the field
//
system_stats
scalar_wave::stats(const field&, const field& f, const step_controller& c) const
{
    auto&& u = f.scalars(scalars::u);
    auto sol = m.xyz | solution(center, radius, c);
    auto [min, max] = rs::minmax(u | m.fluid);
    real error = rs::max(abs(u - sol) | m.fluid);
    return system_stats{.stats = {error, min, max}};
}

//
// Determine if the computed field is valid by checking the linf error
//
bool scalar_wave::valid(const system_stats& stats) const
{
    const auto& v = stats.stats[0];
    return std::isfinite(v) && std::abs(v) <= 1e6;
}

//
// rhs = - grad(G) . grad(u) -> dot(neg_G, du)
//
void scalar_wave::rhs(field_view f, real, field_span rhs)
{
    auto&& u = f.scalars(scalars::u);
    auto&& u_rhs = rhs.scalars(scalars::u);

    du = grad(u);
    u_rhs = dot(grad_G, du);
}

real3 scalar_wave::summary(const system_stats& stats) const
{
    return {stats.stats[0], stats.stats[1], stats.stats[2]};
}

void scalar_wave::update_boundary(field_span f, real time)
{
    auto&& u = f.scalars(scalars::u);
    auto sol = m.xyz | solution(center, radius, time);

    u | m.dirichlet(grid_bcs) = sol;
    // assumes dirichlet right now
    u | sel::R = sol;
}

std::span<const std::string> scalar_wave::names() const { return io_names; }

void scalar_wave::log(const system_stats& stats, const step_controller& step)
{
    if (!logger) return;
    logger->info(
        fmt::format("{},{},{}", (real)step, (int)step, fmt::join(stats.stats, ",")));
}

system_size scalar_wave::size() const { return {1, 0, m.ss()}; }

std::optional<scalar_wave> scalar_wave::from_lua(const sol::table&)
{
    return std::nullopt;
}

} // namespace ccs::systems
