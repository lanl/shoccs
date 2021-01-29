#include "EmptySystem.hpp"

#include <spdlog/spdlog.h>

#include <limits>

namespace ccs::systems
{
void EmptySystem::operator()(SystemField&, const StepController&) {}

SystemStats
EmptySystem::stats(const SystemField&, const SystemField&, const StepController&) const
{
    return {};
}

bool EmptySystem::valid(const SystemStats&) const { return false; }

real EmptySystem::timestep_size(const SystemField&, const StepController&) const
{
    return std::numeric_limits<real>::max();
}

void EmptySystem::rhs(SystemView_Const, real, SystemView_Mutable) {}

void EmptySystem::update_boundary(SystemView_Mutable, real) {}

void EmptySystem::log(const SystemStats&, const StepController&)
{
    if (auto logger = spdlog::get("system"); logger) { logger->info("EmptySystem"); }
}
} // namespace ccs::systems