#pragma once

#include "system.hpp"

namespace pdg
{

// the system of pdes to solve is in this class
class vc_scalar_wave : public system
{
        // required data structure
        discrete_operator grad;
        // required data
        std::vector<double> grad_c;
        std::vector<double> grad_u;

        std::array<double, 3> center; // center of the circular wave
        double radius;

        system_stats stats0;           // the stats associated with the previous timestep
        double stats_begin_accumulate; // time when accumululated errors begin;

    public:
        vc_scalar_wave() = default;

        vc_scalar_wave(cart_mesh&& cart,
                       mesh&& cut_mesh,
                       discrete_operator&& grad,
                       field_io& io,
                       std::array<double, 3> center,
                       double radius,
                       double stats_begin_accumulate);

        // evaluates the rhs of the system at a particular time
        void operator()(std::vector<double>& rhs, double time) override;

        system_stats stats(double time) override;

        int_t rhs_size() const override;

    private:
        double system_timestep_size(double cfl) const override;
};

} // namespace pdg
