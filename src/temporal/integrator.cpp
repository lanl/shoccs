#include "integrator.hpp"

namespace ccs
{

std::function<void(field_span)> integrator::operator()(system& s,
                                                       const field& f,
                                                       const step_controller& controller,
                                                       real dt)
{
    return std::visit(
        [&s, &f, &controller, dt](auto&& integrator_v) {
            return std::function<void(field_span)>{
                [&s, &f, &controller, dt, &integrator_v](field_span fs) {
                    integrator_v(s, f, fs, controller, dt);
                }};
        },
        v);
}
} // namespace ccs