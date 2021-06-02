#pragma once

#include "types.hpp"
#include <optional>

namespace ccs
{
struct step_controller {

    bool done() const { return true; }

    std::optional<real> check_timestep_size(real) const { return std::nullopt; }

    operator real() const { return 0; }

    real simulation_time() const { return 0; }

    int simulation_step() const { return 0; }

    void advance(real)
    {
        // time += *dt;
        // step += 1;
    }
};
} // namespace ccs
