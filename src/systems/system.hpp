#pragma once

// systems
#include "empty_system.hpp"
#include "heat.hpp"
#include "hyperbolic_eigenvalues.hpp"
#include "inviscid_vortex.hpp"
#include "scalar_wave.hpp"

#include "fields/field_registry.hpp"
#include "io/logging.hpp"
#include "temporal/step_controller.hpp"
#include "types.hpp"
#include <sol/forward.hpp>
#include <variant>

namespace ccs
{

// Variant wrapper over concrete PDE systems.
// All field operations use sim_registry + field_ref (registry-based API).
class system
{
    std::variant<systems::empty,
                 systems::scalar_wave,
                 systems::inviscid_vortex,
                 systems::heat,
                 systems::hyperbolic_eigenvalues>
        v;
    using v_t = decltype(v);

public:
    system() = default;

    template <typename T>
        requires(std::constructible_from<v_t, T>)
    system(T&& t) : v{FWD(t)} {}

    void log(const system_stats&, const step_controller&);

    // returns true if the system stats say so
    bool valid(const system_stats&) const;

    real3 summary(const system_stats&) const;

    static std::optional<system> from_lua(const sol::table&, const logs& = {});

    system_size size() const;

    // Registry-based dispatch methods
    void rhs(const sim_registry& creg, field_ref input,
             sim_registry& reg, field_ref output, real time);
    void build_rhs_graph(const sim_registry& creg, field_ref input,
                         sim_registry& reg, field_ref output);
    void submit_rhs_graph(const sim_registry& creg, field_ref input,
                          sim_registry& reg, field_ref output, real time);
    void update_boundary(sim_registry& reg, field_ref ref, real time);
    system_stats stats(const sim_registry& reg, field_ref u0,
                       field_ref u1, const step_controller&) const;
    void initialize(sim_registry& reg, field_ref ref, const step_controller&);
    bool write(field_io& io, const sim_registry& reg, field_ref ref,
               const step_controller& c, real dt);
    std::optional<real> timestep_size(const sim_registry& reg, field_ref u,
                                      const step_controller&) const;
};

} // namespace ccs
