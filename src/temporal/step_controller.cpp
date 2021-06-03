#include "step_controller.hpp"

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

namespace ccs
{
std::optional<step_controller> step_controller::from_lua(const sol::table& tbl)
{
    if (!tbl["step_controller"].valid()) {
        spdlog::info("step_controller not specified.  using default");
        return step_controller{};
    }

    auto c = tbl["step_controller"];
    int max_step = c["max_step"].get_or(0);
    real max_time = c["max_time"].get_or(0.0);
    real min_dt = c["min_dt"].get_or(1e-6);
    real h_cfl = c["cfl"]["hyperbolic"].get_or(1.0);
    real p_cfl = c["cfl"]["parabolic"].get_or(1.0);

    return step_controller{
        bounded<int>{max_step}, bounded<real>{max_time}, h_cfl, p_cfl, min_dt};
}
} // namespace ccs
