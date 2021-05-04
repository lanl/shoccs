#include "rk4.hpp"
#include "step_controller.hpp"
#include "systems/system.hpp"

namespace ccs::integrators
{

constexpr std::array rki{0.0, 0.5, 0.5, 1.0};
constexpr std::array rkf{1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};

void rk4::operator()(system& system,
                     const field& u0,
                     field_span u,
                     const step_controller& controller,
                     real dt)
{
    rk_rhs = 0;
    const auto time = controller.simulation_time();

    for (int i = 0; i < 4; ++i) {
        if (i > 0) {
            system_rhs = system.rhs(u, time + dt * rki[i]);
            // sys.rhs(u, dt * rki[i], system_rhs);
            // could we instead do?
            u = u0 + dt * rki[i] * system_rhs;
            // this is far more intuitive but does require that the system
            // data (u, u0) be exposed.  would also require some encapsulation
            // of boundary values.  Should they all be packaged up in a "field?"
            // u.update_boundary(time + dt * rk[i]);

            // or maybe something like
            system.update_boundary(u, time + dt * rki[i]);
            // boundaries are very much related to the system of equations
            // being solved.  The first approach would require embeddeding
            // a lot of information in the field that might better
            // belong to the system class
        }
        //(system_rhs, time + dt * rki[i]);
        // or maybe
        // sys.rhs(u, time + dt * rki[i], system_rhs);
        // or
        system_rhs = system.rhs(u, time + dt * rki[i]);

        // accumulate rk fluxes
        // for (int_t j = 0, end = system_rhs.size(); j < end; ++j)
        //        rk_rhs[j] += dt * rkf[i] * system_rhs[j];
        // or maybe just
        rk_rhs += dt * rkf[i] * system_rhs;
    }

    // final update
    // sys.update(rk_rhs, 1.0);
    // or maybe just
    u = u0 + rk_rhs;
    // or
    // return u0 + rk_rhs;
    // if we do the above then this becomes an `auto` return type and needs to move
    // to a header...
    // unless this returns a std::function which can be invoke
}
} // namespace ccs::integrators
