#pragma once

#include "fields/field.hpp"

namespace ccs
{
// Forward decls
class system;
class step_controller;

namespace integrators
{

class euler
{
    field system_rhs;

public:
    euler() = default;

    void ensure_size(system_size);

    void operator()(system&, const field&, field_span, const step_controller&, real);
};
} // namespace integrators
} // namespace ccs
