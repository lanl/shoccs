#include "EmptyIntegrator.hpp"

namespace ccs::integrators
{
void EmptyIntegrator::operator()(
    System&, const SystemField&, SystemView_Mutable, const StepController&, real)
{
}
} // namespace ccs::integrators