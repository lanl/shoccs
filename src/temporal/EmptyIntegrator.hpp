#pragma once

#include "fields/SystemField.hpp"
#include "types.hpp"

namespace ccs
{
// Forward decls
class System;
class StepController;

namespace integrators
{

struct EmptyIntegrator {
    void operator()(
        System&, const SystemField&, SystemView_Mutable, const StepController&, real);
};
} // namespace integrators

} // namespace ccs