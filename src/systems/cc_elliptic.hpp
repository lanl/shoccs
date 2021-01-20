#pragma once

#include "system.hpp"

namespace pdg
{

// the system of pdes to solve is in this class
class cc_elliptic : public system
{
        std::unique_ptr<manufactured_solution> ms;
        coupled_discrete_operator lap;

        system_stats stats0; // the stats associated with the previous timestep

    public:
        cc_elliptic() = default;

        cc_elliptic(cart_mesh&& cart,
                    mesh&& cut_mesh,
                    std::unique_ptr<manufactured_solution>&& ms,
                    coupled_discrete_operator&& lap,
                    field_io& io);

        // evaluates the rhs of the system at a particular time
        void operator()(std::vector<double>& rhs, double time) override;

        system_stats stats(double time) override;

        int_t rhs_size() const override;

    private:
        double system_timestep_size(double) const override;
};

} // namespace pdg
