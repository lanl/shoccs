#pragma once

#include "system.hpp"

namespace pdg
{

// the system of pdes to solve is in this class
class euler_vortex : public system
{
        // only require one operator
        discrete_operator div;
        // required data
        std::vector<real_t> P;
        std::vector<real_t> div_u;
        std::vector<real_t> work;


        std::array<real_t, 2> center; // initial center of the vortex
        real_t eps;                   // vortex strength
        real_t M0;                    // background Mach number

        SystemStats stats0;           // the stats associated with the previous timestep
        real_t stats_begin_accumulate; // time when accumululated errors begin;

    public:
        euler_vortex() = default;

        euler_vortex(cart_mesh&& cart,
                     mesh&& cut_mesh,
                     discrete_operator&& grad,
                     field_io& io,
                     std::array<real_t, 2> center,
                     real_t eps,
                     real_t M0,
                     real_t stats_begin_accumulate);

        // evaluates the rhs of the system at a particular time
        void operator()(std::vector<real_t>& rhs, real_t time) override;

        SystemStats stats(real_t time) override;

        int_t rhs_size() const override;

    private:
        real_t system_timestep_size(real_t cfl) const override;
};

} // namespace pdg
