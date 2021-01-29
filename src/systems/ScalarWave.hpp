#pragma once

#include "StepController.hpp"
#include "fields/SystemField.hpp"
#include "types.hpp"

namespace ccs::systems
{

// the system of pdes to solve is in this class
class ScalarWave
{
    // required data structure
    //ccs::op::gradient grad;
    // required data
    //std::vector<double> grad_c;
    //std::vector<double> grad_u;

    //std::array<double, 3> center; // center of the circular wave
    //double radius;

    //SystemStats stats0;           // the stats associated with the previous timestep
    //double stats_begin_accumulate; // time when accumululated errors begin;

public:
    ScalarWave() = default;

#if 0
    vc_scalar_wave(cart_mesh&& cart,
                   mesh&& cut_mesh,
                   discrete_operator&& grad,
                   field_io& io,
                   std::array<double, 3> center,
                   double radius,
                   double stats_begin_accumulate);
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
