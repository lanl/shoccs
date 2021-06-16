#pragma once

#include "fields/field.hpp"
#include "operators/divergence.hpp"
#include "temporal/step_controller.hpp"
#include "types.hpp"

namespace ccs::systems
{

// the system of pdes to solve is in this class
class inviscid_vortex
{
    // only require one operator
    divergence div;

    field U;
#if 0   
    // required data
    std::vector<real_t> P;
    std::vector<real_t> div_u;
    std::vector<real_t> work;

    std::array<real_t, 2> center; // initial center of the vortex
    real_t eps;                   // vortex strength
    real_t M0;                    // background Mach number

    system_stats stats0;            // the stats associated with the previous timestep
    real_t stats_begin_accumulate; // time when accumululated errors begin;
#endif

public:
    inviscid_vortex() = default;

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
    void operator()(field& s, const step_controller&);

    system_stats stats(const field& u0, const field& u1, const step_controller&) const;

    bool valid(const system_stats&) const;

    real timestep_size(const field&, const step_controller&) const;

    void rhs(field_view, real, field_span);

    void update_boundary(field_span, real time);

    void log(const system_stats&, const step_controller&);

    real3 summary(const system_stats&) const;

    system_size size() const;
};

} // namespace ccs::systems
