#include "Integrator.hpp"

namespace ccs
{

std::function<void(SystemView_Mutable)> Integrator::operator()(
    System& system, const SystemField& field, const StepController& controller, real dt)
{
    return std::visit(
        [&system, &field, &controller, dt](auto&& integrator_v) {
            return std::function<void(SystemView_Mutable)>{
                [&system, &field, &controller, dt, &integrator_v](
                    SystemView_Mutable view) {
                    integrator_v(system, field, view, controller, dt);
                }};
        },
        integrator);
}
} // namespace ccs