#pragma once


#include "fields/system_field.hpp"
#include "step_controller.hpp"
#include "types.hpp"

namespace ccs::systems
{

// This class is a template demonstrating the api and ensuring the
// variant in system is default constructible

struct empty {
    void operator()(field& s, const step_controller&);

    system_stats stats(const field& u0, const field& u1, const step_controller&) const;

    bool valid(const system_stats&) const;

    real timestep_size(const field&, const step_controller&) const;

    void rhs(field_view, real, field_span);

    void update_boundary(field_span, real time);

    void log(const system_stats& stats, const step_controller& controller);

    system_size size() const;
};
} // namespace ccs::systems
