#include "heat.hpp"
#include "fields/algorithms.hpp"
#include "fields/selector.hpp"
#include "real3_operators.hpp"
#include <cmath>
#include <numbers>
#include <spdlog/spdlog.h>

#include "operators/discrete_operator.hpp"

namespace ccs::systems
{
enum class scalars : int { u };

// Heat::Heat() {}

void heat::operator()(field&, const step_controller&) {}

system_stats heat::stats(const field&, const field&, const step_controller&) const
{
    return {};
}

bool heat::valid(const system_stats&) const { return {}; }

real heat::timestep_size(const field&, const step_controller&) const { return {}; };

void heat::rhs(field_view, real, field_span) {}

void heat::update_boundary(field_span, real)
{
    // auto&& [u] = system.scalars(scalars::u);
}

void heat::log(const system_stats&, const step_controller&)
{
    if (auto logger = spdlog::get("system"); logger) { logger->info("Heat"); }
}
} // namespace ccs::systems