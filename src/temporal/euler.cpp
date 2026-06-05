#include "euler.hpp"
#include "slot_ops.hpp"
#include "step_controller.hpp"
#include "systems/system.hpp"

#include <Kokkos_Profiling_ScopedRegion.hpp>

namespace ccs::integrators
{

void euler::operator()(system& sys, sim_registry& reg,
                       field_ref u0, field_ref output,
                       field_ref system_rhs_ref,
                       const step_controller& ctrl, real dt)
{
    Kokkos::Profiling::ScopedRegion step_region("euler::step");
    const real time = ctrl;

    // Copy u0 into output so the pre-built RHS graph (bound to the output
    // slot) reads the current solution, matching RK4's slot convention.
    reg.deep_copy_slot(output.slot, u0.slot);

    slot_zero(reg, system_rhs_ref);
    sys.submit_rhs_graph(reg, output, reg, system_rhs_ref, time);
    slot_assign_lc(reg, output, u0, dt, system_rhs_ref);
    sys.update_boundary(reg, output, time + dt);
}

} // namespace ccs::integrators
