#include "euler.hpp"

namespace pdg
{
euler::euler(int_t rhs_sz) : system_rhs(rhs_sz) {}

void euler::step(system& sys, double time, double dt)
{
        sys(system_rhs, time);
        sys.update(system_rhs, dt);
}

std::unique_ptr<integrate> build_euler(int_t rhs_sz)
{
    return std::make_unique<euler>(rhs_sz);
}
} // namespace pdg
