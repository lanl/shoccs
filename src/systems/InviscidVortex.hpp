#pragma once

#include "StepController.hpp"
#include "fields/SystemField.hpp"
#include "operators/Divergence.hpp"
#include "types.hpp"

namespace ccs::systems
{

// the system of pdes to solve is in this class
class InviscidVortex
{
    // only require one operator
    operators::Divergence divergence;

    SystemField U;
#if 0   
    // required data
    std::vector<real_t> P;
    std::vector<real_t> div_u;
    std::vector<real_t> work;

    std::array<real_t, 2> center; // initial center of the vortex
    real_t eps;                   // vortex strength
    real_t M0;                    // background Mach number

    SystemStats stats0;            // the stats associated with the previous timestep
    real_t stats_begin_accumulate; // time when accumululated errors begin;
#endif

public:
    InviscidVortex() = default;

#if 0
    euler_vortex(cart_mesh&& cart,
                 mesh&& cut_mesh,
                 discrete_operator&& grad,
                 field_io& io,
                 std::array<real_t, 2> center,
                 real_t eps,
                 real_t M0,
                 real_t stats_begin_accumulate);
#endif
    void operator()(SystemField& s, const StepController&);

    SystemStats
    stats(const SystemField& u0, const SystemField& u1, const StepController&) const;

    bool valid(const SystemStats&) const;

    real timestep_size(const SystemField&, const StepController&) const;

    void rhs(SystemView_Const, real, SystemView_Mutable);

    void update_boundary(SystemView_Mutable, real time);

    void log(const SystemStats&, const StepController&);
};

} // namespace ccs::systems
