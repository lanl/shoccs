#pragma once
#include "integrate.hpp"

namespace pdg
{
class rk4 : public integrate
{
        std::vector<real_t> system_rhs;
        std::vector<real_t> rk_rhs;

    public:
        rk4() = default;

        rk4(int_t rhs_sz);

        void step(system& sys, double time, double dt) override;
};
} // namespace pdg
