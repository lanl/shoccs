#include "rk4.hpp"

namespace pdg
{

constexpr std::array rki{0.0, 0.5, 0.5, 1.0};
constexpr std::array rkf{1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};

rk4::rk4(int_t rhs_sz) : system_rhs(rhs_sz), rk_rhs(rhs_sz) {}

void rk4::step(const system& sys, const system_field& u0, real time, real dt, system_field& u)
{
        // could we instead do
        rk_rhs = 0;

        for (int i = 0; i < 4; ++i) {
                if (i > 0) {
                        system_rhs = sys.rhs(u, dt * rki[i]);
                        //sys.rhs(u, dt * rki[i], system_rhs);
                        // could we instead do?
                        u = u0 + dt * rki[i] * system_rhs;
                        // this is far more intuitive but does require that the system
                        // data (u, u0) be exposed.  would also require some encapsulation
                        // of boundary values.  Should they all be packaged up in a "system_field?"
                        u.update_boundary(time + dt * rk[i]);
                        // or maybe something like
                        sys.update_boundary(u, time + dt * rk[i]);
                        // boundaries are very much related to the system of equations
                        // being solved.  The first approach would require embeddeding
                        // a lot of information in the system_field that might better
                        // belong to the system class
                }

                (system_rhs, time + dt * rki[i]);
                // or maybe
                sys.rhs(u, time + dt * rki[i], system_rhs);
                // or
                system_rhs = sys.rhs(u, time + dt * rki[i])

                // accumulate rk fluxes
                //for (int_t j = 0, end = system_rhs.size(); j < end; ++j)
                //        rk_rhs[j] += dt * rkf[i] * system_rhs[j];
                // or maybe just
                rk_rhs += dt * rkf[i] * system_rhs;
        }

        // final update
        // sys.update(rk_rhs, 1.0);
        // or maybe just
        u = u0 + rk_rhs;
        // or
        //return u0 + rk_rhs;
        // if we do the above then this becomes an `auto` return type and needs to move
        // to a header...
        // unless this returns a std::function which can be invoke

std::unique_ptr<integrate> build_rk4(int_t rhs_sz)
{
        return std::make_unique<rk4>(rhs_sz);
}
} // namespace pdg
