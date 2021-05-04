#pragma once

#include "fields/system_field.hpp"

namespace ccs
{
// Forward decls
class system;
class step_controller;

namespace integrators
{

class rk4
{
    field rk_rhs;
    field system_rhs;

public:
    rk4() = default;

    void operator()(system&, const field&, field_span, const step_controller&, real);
};
} // namespace integrators
} // namespace ccs
