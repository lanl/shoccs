#include "rk4.hpp"
#include "step_controller.hpp"
#include "systems/system.hpp"

namespace ccs::integrators
{

constexpr std::array rki{0.0, 0.5, 0.5, 1.0};
constexpr std::array rkf{1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};

void rk4::ensure_size(system_size sz)
{
    if (ssize(rk_rhs) != sz) {
        rk_rhs = field{sz};
        system_rhs = field{sz};
    }
}

void rk4::operator()(system& system,
                     const field& u0,
                     field_span u,
                     const step_controller& controller,
                     real dt)
{
    rk_rhs = 0;
    system_rhs = 0;
    const real time = controller;
    u = u0;

    for (int i = 0; i < 4; ++i) {
        if (i > 0) {
            u = u0 + dt * rki[i] * system_rhs;
            system.update_boundary(u, time + dt * rki[i]);
        }
        system_rhs = system.rhs(u, time + dt * rki[i]);
        rk_rhs += dt * rkf[i] * system_rhs;
    }

    // final update
    u = u0 + rk_rhs;
    system.update_boundary(u, time + dt);
}
} // namespace ccs::integrators
