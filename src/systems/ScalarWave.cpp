#include "ScalarWave.hpp"
#include "fields/Selector.hpp"
#include "fields/algorithms.hpp"
#include "fields/views.hpp"
#include "real3_operators.hpp"
#include <cmath>
#include <numbers>
#include <spdlog/spdlog.h>

#include "operators/DiscreteOperator.hpp"

#include <range/v3/view/transform.hpp>

namespace ccs::systems
{
// system variables to be used in this system
enum class scalars : int { u };

constexpr real twoPI = 2 * std::numbers::pi_v<real>;

// negative gradient - coefficients of gradient
template <int I>
struct neg_G {
    real3 center;
    real radius;

    template <typename T>
    real operator()(T&& location) const
    {
        return -(std::get<I>(location) - std::get<I>(center)) / length(location - center);
    }
};

struct solution {
    real3 center;
    real radius;
    real time;

    //    real operator()(const mesh_object_info& info) const { return
    //    (*this)(info.position); }

    template <typename T>
    real operator()(T&& location) const
    {
        return std::sin(twoPI * (length(location - center) - radius - time));
    }

    // gradient for neumann boundaries
    real3 grad(const real3& location) const
    {
        const auto d = location - center;
        return twoPI * d * std::cos(twoPI * (length(d) - radius - time)) / length(d);
    }
};

ScalarWave::ScalarWave( // cart_mesh&& cart_,
                        // mesh&& cut_mesh_,
                        // discrete_operator&& grad_,
                        // field_io& io,
    real3 center_,
    real radius_) //,
                  // double stats_begin_accumulate_)
    : gradient{}, grad_G{}, du{}, center{center_}, radius{radius_}
{

    // Initialize the operator
    auto op = operators::DiscreteOperator{};
    gradient =
        op.to<operators::Gradient>(bcs::Grid{}); // domainBoundaries, ObjectBoundaries);
    // Initialize wave speeds
    grad_G | selector::D =
        mesh::location | field::Tuple{vs::transform(neg_G<0>{center, radius}),
                                      vs::transform(neg_G<1>{center, radius}),
                                      vs::transform(neg_G<2>{center, radius})};
    grad_G | selector::Rxyz = 0;
}

real ScalarWave::timestep_size(const SystemField&, const StepController&) const
{
    // will need some notion of mesh size to implement this
    // return cfl * std::min(m.h());
    return null_v<>;
}

// Evaluate the rhs of the system using the current u
// this shouldn't be a std::vector<double>... This should be encapsulated in some
// kind of field, vector/scalar field seems inadequate.  What would a system field look
// like? It could contain vector and scalar fields.  Should it be owning (vectors) or
// non-owning (spans)? If SystemField be a templated entity than the call operator can't
// be virtual (perhaps tag_invoke)?
void ScalarWave::operator()(SystemField& field, const StepController& controller)
{
    // ensure the SystemField is the right size
    if (controller.simulation_step() == 0) {
        // if (field.nfields() == 1 && field.extents() == int3{} && field.solid().size()
        // != 0) {
        // adjust the sizes
        field.nscalars(1).nvectors(0).extents(int3{}).solid(0).object_boundaries(int3{});
    }
    // extract the field components to initialize
    auto&& [u] = field.scalars(scalars::u);

    const auto time = controller.simulation_time();

    auto sol = mesh::location | vs::transform(solution{center, radius, time});
    u | selector::D = sol;
    u | selector::Rxyz = sol;
}

SystemStats
ScalarWave::stats(const SystemField&, const SystemField&, const StepController&) const
{
    // this->stats_(stats0, time >= stats_begin_accumulate, time, solution{center,
    // radius}); return stats0;
    return {};
}

bool ScalarWave::valid(const SystemStats&) const { return false; }

void ScalarWave::rhs(SystemView_Const field, real, SystemView_Mutable field_rhs)
{
    // here we assume that the boundary values have been properly set in u
    // but what to do do about neumann bc's?  Should we store them in du?
    // Or should we just have a separate field for them?  Keep them separate for now

    // for multivariate systems, will need to extract the variables and apply different
    // operators on them
    auto&& [u] = field.scalars(scalars::u);
    // grad should d/dx, d/dy, d/dz on F and Rx, Ry, Rz, respectively
    // du = gradient(u);
    gradient(u);
    // compute grad_G . u_rhs and store in v_rhs
    auto&& [u_rhs] = field_rhs.scalars(scalars::u);
    u_rhs = dot(grad_G, du);
}

void ScalarWave::update_boundary(SystemView_Mutable field, real time)
{
    // update object boundaries
    auto sol = mesh::location | vs::transform(solution{center, radius, time});
    auto&& [u] = field.scalars(scalars::u);

    u | selector::Rxyz = sol;

    // update domain boundaries
}

void ScalarWave::log(const SystemStats&, const StepController&)
{
    if (auto logger = spdlog::get("system"); logger) { logger->info("ScalarWave"); }
}

} // namespace ccs::systems
