#pragma once

#include <functional>
#include <variant>

#include "RK4.hpp"
#include "EmptyIntegrator.hpp"
#include "fields/SystemField.hpp"
#include "types.hpp"

namespace ccs
{
// forward decl
class System;

class Integrator
{
    std::variant<integrators::EmptyIntegrator, integrators::RK4> integrator;

public:
    Integrator() = default;

    std::function<void(SystemView_Mutable)>
    operator()(System&, const SystemField&, const StepController&, real dt);
};

} // namespace ccs
