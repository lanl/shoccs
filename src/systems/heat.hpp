#include "fields/system_field.hpp"
#include "mms/manufactured_solutions.hpp"
#include "operators/laplacian.hpp"
#include "step_controller.hpp"

namespace ccs::systems
{
//
// solve dT/dt = k lap T
//
class heat
{
    laplacian lap;

    manufactured_solution m_sol;

    real diffusivity;

public:
    heat() = default;

    // real constructor
    // Heat();

    void operator()(field&, const step_controller&);

    system_stats stats(const field& u0, const field& u1, const step_controller&) const;

    bool valid(const system_stats&) const;

    real timestep_size(const field&, const step_controller&) const;

    void rhs(field_view, real, field_span);

    void update_boundary(field_span, real time);

    void log(const system_stats&, const step_controller&);
};
} // namespace ccs::systems
