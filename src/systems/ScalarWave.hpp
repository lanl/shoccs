#pragma once

#include "StepController.hpp"
#include "fields/SystemField.hpp"
#include "operators/Gradient.hpp"
#include "types.hpp"

namespace ccs::systems
{

// the system of pdes to solve is in this class
class ScalarWave
{
    // required data structure
    operators::Gradient gradient;

    // how should one initialize these?  Do they need the mesh or do they
    // only require some reduced set of information
    SystemField u_rhs;
    SimpleVector<std::vector<real>> grad_G;
    SystemField du;

    // required data
    // std::vector<double> grad_c;
    // std::vector<double> grad_u;

    real3 center; // center of the circular wave
    real radius;

    // SystemStats stats0;           // the stats associated with the previous timestep
    // double stats_begin_accumulate; // time when accumululated errors begin;

public:
    ScalarWave() = default;

    ScalarWave(real3 center, real radius);

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
