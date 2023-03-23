#pragma once

#include "fields/field.hpp"

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
    void ensure_size(system_size);

    void operator()(system&, const field&, field_span, const step_controller&, real);
};
} // namespace integrators
} // namespace ccs
