#pragma once

#include "fields/field.hpp"
#include "types.hpp"

namespace ccs
{
// Forward decls
class system;
class step_controller;

namespace integrators
{

struct empty {
    void operator()(system&, const field&, field_span, const step_controller&, real);

    void ensure_size(system_size);
};
} // namespace integrators

} // namespace ccs
