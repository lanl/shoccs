#include "System.hpp"

namespace ccs
{

std::function<void(SystemField&)> System::operator()(const StepController& controller)
{
    return std::visit(
        [&controller](auto&& current_system) {
            return std::function<void(SystemField&)>{
                [&controller, &current_system](SystemField& field) {
                    current_system(field, controller);
                }};
        },
        system);
}

std::function<void(SystemView_Mutable)> System::rhs(SystemView_Const field, real time)
{
    return std::visit(
        [field, time](auto&& current_system) {
            return std::function<void(SystemView_Mutable)>{
                [&current_system, field, time](SystemView_Mutable view) {
                    current_system.rhs(field, time, view);
                }};
        },
        system);
}

void System::update_boundary(SystemView_Mutable view, real time)
{
    std::visit([view, time](
                   auto&& current_system) { current_system.update_boundary(view, time); },
               system);
}

std::optional<real> System::timestep_size(const SystemField& field,
                                          const StepController& controller) const
{
    const auto predicted_dt = std::visit(
        [&field, &controller](auto&& current_system) {
            return current_system.timestep_size(field, controller);
        },
        system);
    // the controller may adjust or invalidate this timestep size
    return controller.check_timestep_size(predicted_dt);
}

bool System::valid(const SystemStats& stats) const
{
    return std::visit([&stats](auto&& sys) { return sys.valid(stats); }, system);
}

SystemStats System::stats(const SystemField& u0,
                          const SystemField& u1,
                          const StepController& controller) const
{
    return std::visit(
        [&u0, &u1, &controller](auto&& sys) { return sys.stats(u0, u1, controller); },
        system);
}

void System::log(const SystemStats& stats, const StepController& controller)
{
    return std::visit(
        [&stats, &controller](auto&& current_system) {
            current_system.log(stats, controller);
        },
        system);
}

} // namespace ccs
