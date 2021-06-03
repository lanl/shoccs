#pragma once

#include "fields/field.hpp"
#include "operators/gradient.hpp"
#include "temporal/step_controller.hpp"
#include "types.hpp"

namespace ccs::systems
{

// the system of pdes to solve is in this class
class scalar_wave
{
    // required data structure
    gradient grad;

    // how should one initialize these?  Do they need the mesh or do they
    // only require some reduced set of information
    vector<std::vector<real>> grad_G;
    vector<std::vector<real>> du;

    // required data
    // std::vector<double> grad_c;
    // std::vector<double> grad_u;

    real3 center; // center of the circular wave
    real radius;

    // system_stats stats0;           // the stats associated with the previous timestep
    // double stats_begin_accumulate; // time when accumululated errors begin;

public:
    scalar_wave() = default;

    scalar_wave(real3 center, real radius);

    void operator()(field& s, const step_controller&);

    system_stats stats(const field& u0, const field& u1, const step_controller&) const;

    bool valid(const system_stats&) const;

    real timestep_size(const field&, const step_controller&) const;

    void rhs(field_view, real, field_span);

    void update_boundary(field_span, real time);

    void log(const system_stats&, const step_controller&);

    system_size size() const;
};

} // namespace ccs::systems
