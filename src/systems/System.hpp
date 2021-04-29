#pragma once

#include "EmptySystem.hpp"
#include "Heat.hpp"
#include "InviscidVortex.hpp"
#include "ScalarWave.hpp"
#include "StepController.hpp"
#include "types.hpp"
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
class System
{
    std::variant<systems::EmptySystem,
                 systems::ScalarWave,
                 systems::InviscidVortex,
                 systems::Heat>
        system;

public:
    System() = default;

    // call operator for solution evaluation
    std::function<void(SystemField&)> operator()(const StepController&);

    SystemStats
    stats(const SystemField& u0, const SystemField& u1, const StepController&) const;

    void log(const SystemStats&, const StepController&);

    // returns true if the system stats say so
    bool valid(const SystemStats&) const;

    // return a valid timestep size based on cfl and system-specific data
    std::optional<real> timestep_size(const SystemField&, const StepController&) const;

    std::function<void(SystemView_Mutable)> rhs(SystemView_Const, real);

    void update_boundary(SystemView_Mutable, real time);
};

} // namespace ccs
