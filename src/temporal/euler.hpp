#pragma once
#include "integrate.hpp"

namespace pdg
{
class euler : public integrate
{
        std::vector<double> system_rhs;

    public:
        euler() = default;

        euler(int_t rhs_sz);

        void step(system& sys, double time, double dt) override;
};
} // namespace pdg
