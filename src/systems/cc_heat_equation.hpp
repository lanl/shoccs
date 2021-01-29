#pragma once

#include "system.hpp"

namespace pdg
{

// the system of pdes to solve is in this class
class cc_heat_equation : public system
{
        std::unique_ptr<manufactured_solution> ms;
        discrete_operator lap;

        // required data
        std::vector<double> lap_u;

        double diffusivity;

        SystemStats stats0;           // the stats associated with the previous timestep
        double stats_begin_accumulate; // time when accumululated errors begin;

    public:
        cc_heat_equation() = default;

        cc_heat_equation(cart_mesh&& cart,
                         mesh&& cut_mesh,
                         std::unique_ptr<manufactured_solution>&& ms,
                         discrete_operator&& lap,
                         field_io& io,
                         double diffusivity,
                         double stats_begin_accumulate);

        // evaluates the rhs of the system at a particular time
        void operator()(std::vector<double>& rhs, double time) override;

        SystemStats stats(double time) override;

        int_t rhs_size() const override;

    private:
        double system_timestep_size(double cfl) const override;
};

} // namespace pdg
