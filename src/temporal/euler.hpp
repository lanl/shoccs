#pragma once

#include "fields/field_registry.hpp"

namespace ccs
{
// Forward decls
class system;
class step_controller;

namespace integrators
{

class euler
{
public:
    euler() = default;

    void operator()(system& sys, sim_registry& reg,
                    field_ref u0, field_ref output,
                    field_ref system_rhs_ref,
                    const step_controller& ctrl, real dt);
};
} // namespace integrators
} // namespace ccs
