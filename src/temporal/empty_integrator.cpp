#include "empty_integrator.hpp"

namespace ccs::integrators
{
void empty::operator()(system&, const field&, field_span, const step_controller&, real) {}

void empty::ensure_size(system_size) {}
} // namespace ccs::integrators
