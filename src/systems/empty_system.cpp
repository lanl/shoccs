#include "empty_system.hpp"

#include <spdlog/spdlog.h>

#include <limits>

namespace ccs::systems
{
void empty::operator()(field&, const step_controller&) {}

system_stats empty::stats(const field&, const field&, const step_controller&) const
{
    return {};
}

bool empty::valid(const system_stats&) const { return false; }

real empty::timestep_size(const field&, const step_controller&) const
{
    return std::numeric_limits<real>::max();
}

void empty::rhs(field_view, real, field_span) {}

void empty::update_boundary(field_span, real) {}

void empty::log(const system_stats&, const step_controller&)
{
    if (auto logger = spdlog::get("system"); logger) { logger->info("EmptySystem"); }
}

real3 empty::summary(const system_stats&) const { return {}; }

bool empty::write(field_io&, field_view, const step_controller&, real) { return false; }

system_size empty::size() const { return {}; }

} // namespace ccs::systems
