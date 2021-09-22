#include "step_controller.hpp"

#include <sol/sol.hpp>

namespace ccs
{
std::optional<step_controller> step_controller::from_lua(const sol::table& tbl,
                                                         const logs& logger)
{
    if (!tbl["step_controller"].valid()) {
        logger(spdlog::level::info, "step_controller not specified.  using default");
        return step_controller{};
    }

    auto c = tbl["step_controller"];
    int max_step = c["max_step"].get_or(std::numeric_limits<int>::max());
    real max_time = c["max_time"].get_or(std::numeric_limits<real>::max());
    real min_dt = c["min_dt"].get_or(1e-6);
    real h_cfl = c["cfl"]["hyperbolic"].get_or(1.0);
    real p_cfl = c["cfl"]["parabolic"].get_or(1.0);

    // if neither are specied (i.e. for eigenvalue analysis then do zero steps)
    if (max_step == std::numeric_limits<int>::max() &&
        max_time == std::numeric_limits<real>::max())
        max_step = 0;

    return step_controller{
        bounded<int>{max_step}, bounded<real>{max_time}, h_cfl, p_cfl, min_dt};
}
} // namespace ccs
