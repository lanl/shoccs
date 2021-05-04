#include "empty_integrator.hpp"

namespace ccs::integrators
{
void empty::operator()(system&, const field&, field_span, const step_controller&, real) {}
} // namespace ccs::integrators