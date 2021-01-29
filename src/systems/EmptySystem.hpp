#pragma once

#include "StepController.hpp"
#include "fields/SystemField.hpp"
#include "types.hpp"

namespace ccs::systems
{

// This class is a template demonstrating the api and ensuring the
// variant in system is default constructible

struct EmptySystem {
    void operator()(SystemField& s, const StepController&);

    SystemStats
    stats(const SystemField& u0, const SystemField& u1, const StepController&) const;

    bool valid(const SystemStats&) const;

    real timestep_size(const SystemField&, const StepController&) const;

    void rhs(SystemView_Const, real, SystemView_Mutable);

    void update_boundary(SystemView_Mutable, real time);

    void log(const SystemStats& stats, const StepController& controller);    
};
} // namespace ccs::systems