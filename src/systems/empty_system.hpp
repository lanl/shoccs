#pragma once

#include "fields/field_registry.hpp"
#include "io/field_io.hpp"
#include "temporal/step_controller.hpp"
#include "types.hpp"

namespace ccs::systems
{

// This class is a template demonstrating the api and ensuring the
// variant in system is default constructible

struct empty {
    bool valid(const system_stats&) const;

    void log(const system_stats& stats, const step_controller& controller);

    real3 summary(const system_stats&) const;

    system_size size() const;

    void rhs(const sim_registry& reg, field_ref input,
             sim_registry& out_reg, field_ref output, real time);
    void update_boundary(sim_registry& reg, field_ref ref, real time);
    real timestep_size(const sim_registry& reg, field_ref ref,
                       const step_controller&) const;
    system_stats stats(const sim_registry& reg, field_ref u0,
                       field_ref u1, const step_controller&) const;
    void initialize(sim_registry& reg, field_ref ref, const step_controller&);
    bool write(field_io& io, const sim_registry& reg, field_ref ref,
               const step_controller& c, real dt);
};
} // namespace ccs::systems
