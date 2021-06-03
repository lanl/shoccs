#pragma once

#include "types.hpp"
#include "utils/bounded.hpp"
#include <optional>
#include <sol/forward.hpp>

namespace ccs
{
class step_controller
{

    bounded<int> step;
    bounded<real> time;
    real h_cfl;
    real p_cfl;
    real min_dt;

public:
    step_controller() = default;
    step_controller(
        bounded<int> step, bounded<real> time, real h_cfl, real p_cfl, real min_dt)
        : step{step}, time{time}, h_cfl{h_cfl}, p_cfl{p_cfl}, min_dt{min_dt}
    {
    }

    bool done() const { return !(step || time); }

    std::optional<real> check_timestep_size(real dt) const
    {
        return dt >= min_dt ? std::optional<real>{dt} : std::nullopt;
    }

    operator real() const { return time; }
    operator int() const { return step; }
    operator bool() const { return time && step; }

    real simulation_time() const { return time; }

    int simulation_step() const { return step; }

    void advance(real dt)
    {
        time += dt;
        step += 1;
    }

    real parabolic_cfl() const { return p_cfl; }
    real hyperbolic_cfl() const { return h_cfl; }

    static std::optional<step_controller> from_lua(const sol::table&);
};
} // namespace ccs
