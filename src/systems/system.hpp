#pragma once

#include "empty_system.hpp"
#include "heat.hpp"
#include "inviscid_vortex.hpp"
#include "scalar_wave.hpp"
#include "temporal/step_controller.hpp"
#include "types.hpp"
#include <sol/forward.hpp>
#include <variant>

namespace ccs
{

// Public API of a `system'
//
// timestep -> computes an acceptable timestep based on current solution state
//
// update -> given a rhs and timestep, update the solution state
//
// prestep -> call before the start of a timestep to precompute things
//
// valid -> returns true if the solution is in a valid state (may be overridden by
// children)
//
// call operator -> evaluates the rhs of the system at the given time
//
// stats -> computes statistics of the system at its present state
//
// log -> given a logger, write system specific information
//
// current_solution/error
//

// May be overridden by child class
//
// valid
// log

// Must be define by child
// private:
//      system_timestep_size
// public:
//      call operator
//      stats -> the corresponding routine in child class should make use of stats_ in
//              parent

// the system of pdes to solve is in this class
class system
{
    std::variant<systems::empty,
                 systems::scalar_wave,
                 systems::inviscid_vortex,
                 systems::heat>
        v;
    using v_t = decltype(v);

public:
    system() = default;

    template <typename T>
        requires(std::constructible_from<v_t, T>)
    system(T&& t) : v{FWD(t)} {}

    // call operator for solution evaluation
    std::function<void(field&)> operator()(const step_controller&);

    system_stats stats(const field& u0, const field& u1, const step_controller&) const;

    void log(const system_stats&, const step_controller&);

    // returns true if the system stats say so
    bool valid(const system_stats&) const;

    // return a valid timestep size based on cfl and system-specific data
    std::optional<real> timestep_size(const field&, const step_controller&) const;

    std::function<void(field_span)> rhs(field_view, real);

    void update_boundary(field_span, real time);

    real3 summary(const system_stats&) const;

    std::span<const std::string> names() const;

    static std::optional<system> from_lua(const sol::table&);

    system_size size() const;
};

} // namespace ccs
