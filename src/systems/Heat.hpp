#include "StepController.hpp"
#include "fields/SystemField.hpp"
#include "mms/manufactured_solutions.hpp"
#include "operators/Laplacian.hpp"

namespace ccs::systems
{
//
// solve dT/dt = k lap T
//
class Heat
{
    operators::Laplacian lap;

    manufactured_solution m_sol;

    real diffusivity;

public:
    Heat() = default;

    // real constructor
    // Heat();

    void operator()(SystemField&, const StepController&);

    SystemStats
    stats(const SystemField& u0, const SystemField& u1, const StepController&) const;

    bool valid(const SystemStats&) const;

    real timestep_size(const SystemField&, const StepController&) const;

    void rhs(SystemView_Const, real, SystemView_Mutable);

    void update_boundary(SystemView_Mutable, real time);

    void log(const SystemStats&, const StepController&);
};
} // namespace ccs::systems
