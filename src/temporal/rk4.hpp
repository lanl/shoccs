#pragma once

#include "fields/field_registry.hpp"

namespace ccs
{
// Forward decls
class system;
class step_controller;

namespace integrators
{

class rk4
{
public:
    rk4() = default;

    void operator()(system& sys, sim_registry& reg,
                    field_ref u0, field_ref output,
                    field_ref rk_rhs_ref, field_ref system_rhs_ref,
                    const step_controller& ctrl, real dt);
};
} // namespace integrators
} // namespace ccs
