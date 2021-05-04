#pragma once

#include "types.hpp"
#include <optional>

namespace ccs
{
struct step_controller {

    bool done() const { return true; }

    std::optional<real> check_timestep_size(real) const { return std::nullopt; }

    real simulation_time() const { return {}; }

    int simulation_step() const { return {}; }

    void advance(real)
    {
        // time += *dt;
        // step += 1;
    }
};
} // namespace ccs