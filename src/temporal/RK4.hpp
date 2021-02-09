#pragma once

#include "fields/SystemField.hpp"

namespace ccs
{
// Forward decls
class System;
class StepController;

namespace integrators
{

class RK4
{
    SystemField rk_rhs;
    SystemField system_rhs;

public:
    RK4() = default;

    void operator()(
        System&, const SystemField&, SystemView_Mutable, const StepController&, real);
};
} // namespace integrators
} // namespace ccs
