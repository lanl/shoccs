#include "empty_system.hpp"

#include <spdlog/spdlog.h>

#include <limits>

namespace ccs::systems
{

bool empty::valid(const system_stats&) const { return false; }

void empty::log(const system_stats&, const step_controller&)
{
    if (auto logger = spdlog::get("system"); logger) { logger->info("EmptySystem"); }
}

real3 empty::summary(const system_stats&) const { return {}; }

system_size empty::size() const { return {}; }

void empty::rhs(const sim_registry&, field_ref,
                sim_registry&, field_ref, real)
{
}

void empty::update_boundary(sim_registry&, field_ref, real) {}

real empty::timestep_size(const sim_registry&, field_ref,
                          const step_controller&) const
{
    return std::numeric_limits<real>::max();
}

system_stats empty::stats(const sim_registry&, field_ref,
                           field_ref, const step_controller&) const
{
    return {};
}

void empty::initialize(sim_registry&, field_ref, const step_controller&) {}

bool empty::write(field_io&, const sim_registry&, field_ref,
                  const step_controller&, real)
{
    return false;
}

} // namespace ccs::systems
