#include "scalar_wave.hpp"
#include "fields/algorithms.hpp"
#include "fields/selector.hpp"
#include "real3_operators.hpp"
#include <cmath>
#include <numbers>

#include <sol/sol.hpp>

#include "operators/discrete_operator.hpp"

#include <range/v3/algorithm/max_element.hpp>
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
                         real radius,
                         real max_error,
                         const logs& build_logger)
    : m{MOVE(m_)},
      grid_bcs{MOVE(grid_bcs)},
      object_bcs{MOVE(object_bcs)},
      grad{gradient(this->m, st, this->grid_bcs, this->object_bcs, build_logger)},
      center{center},
      radius{radius},
      grad_G{m.vs()},
      du{m.vs()},
      error{m.ss()},
      max_error{max_error},
      logger{build_logger, "system", "system.csv"}
{

    // Initialize wave speeds
    grad_G | m.fluid = m.vxyz | tuple{neg_G<0>(center, radius),
                                      neg_G<1>(center, radius),
                                      neg_G<2>(center, radius)};
    grad_G | sel::xR = m.vxyz | neg_G<0>(center, radius);
    grad_G | sel::yR = m.vxyz | neg_G<1>(center, radius);
    grad_G | sel::zR = m.vxyz | neg_G<2>(center, radius);
    grad_G | m.dirichlet(this->grid_bcs, this->object_bcs) = 0;

    spdlog::debug("-grad_G {}\n", get<vi::xRx>(grad_G)[0]);

    logger.set_pattern("%v");
    logger(spdlog::level::info,
           "Timestamp,Time,Step,Linf,Min,Max,Domain_Linf,Domain_ic,Rx_Linf,Rx_ic,Ry_"
           "Linf,Ry_ic,Rz_Linf,Rz_ic");
    logger.set_pattern("%Y-%m-%d %H:%M:%S.%f,%v");
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
    auto [u_min, u_max] = minmax(u | m.fluid_all(object_bcs));

    real err = max(abs(u - sol) | m.fluid_all(object_bcs));
    // Extra info for debugging:
    auto linf = abs(u - sol);
    auto fluid_error = linf | m.fluid_all(object_bcs);
    auto max_el = transform(rs::max_element, fluid_error);
    auto err_pairs = transform(
        [](auto&& rng, auto&& max_el) {
            if (rs::end(rng) != max_el)
                return std::pair{
                    *max_el, (real)rs::distance(rs::begin(rng.base()), max_el.base())};
            else
                return std::pair{0.0, (real)0};
        },
        fluid_error,
        max_el);

    auto&& [d, rx, ry, rz] = err_pairs;
    return system_stats{.stats = {err,
                                  u_min,
                                  u_max,
                                  d.first,
                                  d.second,
                                  rx.first,
                                  rx.second,
                                  ry.first,
                                  ry.second,
                                  rz.first,
                                  rz.second}};
}

//
// Determine if the computed field is valid by checking the linf error
//
bool scalar_wave::valid(const system_stats& stats) const
{
    const auto& v = stats.stats[0];
    return std::isfinite(v) && std::abs(v) <= max_error;
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

//
// Must be called before computing the rhs
//
void scalar_wave::update_boundary(field_span f, real time)
{
    auto&& u = f.scalars(scalars::u);
    auto sol = m.xyz | solution(center, radius, time);

    u | m.dirichlet(grid_bcs, object_bcs) = sol;
}

bool scalar_wave::write(field_io& io, field_view f, const step_controller& c, real dt)
{
    auto&& u = f.scalars(scalars::u);
    auto sol = m.xyz | solution(center, radius, (real)c);

    error = 0;
    error | m.fluid_all(object_bcs) = abs(u - sol);
    error | m.dirichlet(grid_bcs, object_bcs) = 0;

    field_view io_view{std::vector<scalar_view>{u, error}, std::vector<vector_view>{}};

    return io.write(io_names, io_view, c, dt, m.R());
}

void scalar_wave::log(const system_stats& stats, const step_controller& step)
{
    logger(spdlog::level::info,
           "{},{},{}",
           (real)step,
           (int)step,
           fmt::join(stats.stats, ","));
}

system_size scalar_wave::size() const { return {1, 0, m.ss()}; }

std::optional<scalar_wave> scalar_wave::from_lua(const sol::table& tbl,
                                                 const logs& logger)
{
    real max_error = tbl["system"]["max_error"].get_or(100.0);
    // assume we can only get here if simulation.system.type == "scalar_wave" so check
    // for the rest
    real3 center;
    real radius;
    // if the center/radius was specified in the system table, use it.
    if (tbl["system"]["center"].valid() && tbl["system"]["radius"].valid()) {

        auto c = tbl["system"]["center"];
        center = {c[1].get_or(0.0), c[2].get_or(0.0), c[3].get_or(0.0)};
        radius = tbl["system"]["radius"];

    } else if (tbl["shapes"].valid()) {
        // attempt to extract the center/radius from the first specified shape in the
        // shapes table
        bool found{false};
        auto t = tbl["shapes"];
        for (int i = 1; t[i].valid() && !found; i++) {
            found = (t[i]["type"].get_or(std::string{}) == "sphere");
            if (found) {
                center = {t[i]["center"][1].get_or(0.0),
                          t[i]["center"][2].get_or(0.0),
                          t[i]["center"][3].get_or(0.0)};
                radius = t[i]["radius"].get_or(0.0);
            }
        }
        if (!found) {
            logger(spdlog::level::err,
                   "No valid spheres found in simulation.shapes for scalar_wave");
            return std::nullopt;
        }
    } else {
        logger(spdlog::level::err,
               "a system.center / system.radius must be specified for scalar_wave");
        return std::nullopt;
    }

    auto mesh_opt = mesh::from_lua(tbl, logger);
    if (!mesh_opt) return std::nullopt;

    auto bc_opt = bcs::from_lua(tbl, mesh_opt->extents(), logger);
    auto st_opt = stencil::from_lua(tbl, logger);

    if (bc_opt && st_opt) {

        return scalar_wave{MOVE(*mesh_opt),
                           MOVE(bc_opt->first),
                           MOVE(bc_opt->second),
                           *st_opt,
                           center,
                           radius,
                           max_error,
                           logger};
    }

    return std::nullopt;
}

} // namespace ccs::systems
