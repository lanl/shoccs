#pragma once

#include <functional>
#include <variant>

#include "empty_integrator.hpp"
#include "fields/system_field.hpp"
#include "rk4.hpp"
#include "types.hpp"

namespace ccs
{
// forward decl
class system;

class integrator
{
    std::variant<integrators::empty, integrators::rk4> v;

public:
    integrator() = default;

    std::function<void(field_span)>
    operator()(system&, const field&, const step_controller&, real dt);
};

} // namespace ccs
