#include "euler.hpp"
#include "step_controller.hpp"
#include "systems/system.hpp"

namespace ccs::integrators
{

void euler::ensure_size(system_size sz)
{
    if (ssize(system_rhs) != sz) { system_rhs = field{sz}; }
}

void euler::operator()(
    system& system, const field& u0, field_span u, const step_controller& step, real dt)
{
    const real time = step;
    system_rhs = 0;

    system_rhs = system.rhs(u0, time);
    u = u0 + dt * system_rhs;
    system.update_boundary(u, time + dt);
}

} // namespace ccs::integrators
