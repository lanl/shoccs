#include "Heat.hpp"
#include "fields/Selector.hpp"
#include "fields/algorithms.hpp"
#include "fields/views.hpp"
#include "real3_operators.hpp"
#include <cmath>
#include <numbers>
#include <spdlog/spdlog.h>

#include "operators/DiscreteOperator.hpp"

namespace ccs::systems
{
enum class scalars : int { u };

// Heat::Heat() {}

void Heat::operator()(SystemField&, const StepController&) {}

SystemStats
Heat::stats(const SystemField&, const SystemField&, const StepController&) const
{
    return {};
}

bool Heat::valid(const SystemStats&) const { return {}; }

real Heat::timestep_size(const SystemField&, const StepController&) const { return {}; };

void Heat::rhs(SystemView_Const, real, SystemView_Mutable) {}

void Heat::update_boundary(SystemView_Mutable, real)
{
    // auto&& [u] = system.scalars(scalars::u);
}

void Heat::log(const SystemStats&, const StepController&)
{
    if (auto logger = spdlog::get("system"); logger) { logger->info("Heat"); }
}
} // namespace ccs::systems