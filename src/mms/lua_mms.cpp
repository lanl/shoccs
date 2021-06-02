#include "lua_mms.hpp"
#include "manufactured_solutions.hpp"
#include <spdlog/spdlog.h>

namespace ccs
{

lua_mms::lua_mms(const sol::table& tbl_) : tbl{tbl_}
{
    if (tbl["call"].get_type() != sol::type::function) {
        spdlog::error("call function not defined for lua_mms");
        return;
    }
    if (tbl["ddt"].get_type() != sol::type::function) {
        spdlog::error("ddt function not defined for lua_mms");
        return;
    }
    if (tbl["grad"].get_type() != sol::type::function) {
        spdlog::error("grad function not defined for lua_mms");
        return;
    }
    if (tbl["div"].get_type() != sol::type::function) {
        spdlog::error("div function not defined for lua_mms");
        return;
    }
    if (tbl["lap"].get_type() != sol::type::function) {
        spdlog::error("lap function not defined for lua_mms");
        return;
    }

    call_ = tbl["call"];
    ddt_ = tbl["ddt"];
    gradient_ = tbl["grad"];
    divergence_ = tbl["div"];
    laplacian_ = tbl["lap"];
}

std::optional<manufactured_solution> lua_mms::from_lua(const sol::table& tbl)
{
    return {lua_mms{tbl}};
}
// std::function<real(real, const real3&)> call_;
// std::function<real(real, const real3&)> ddt_;
// std::function<real3(real, const real3&)> gradient_;
// std::function<real(real, const real3&)> divergence_;
// std::function<real(real, const real3&)> laplacian_;

// auto operator()(real time, const real3& loc) const { return call_(time, loc); }
// auto ddt(real time, const real3& loc) const { return ddt_(time, loc); }
// auto gradient(real time, const real3& loc) const { return gradient_(time, loc); }
// auto divergence(real time, const real3& loc) const { return divergence_(time, loc); }
// auto laplacian(real time, const real3& loc) const { return laplacian_(time, loc); }
// };

// manufactured_solution build_lua_mms(std::function<real(real, const real3&)>&& call,
//                                     std::function<real(real, const real3&)>&& ddt,
//                                     std::function<real3(real, const real3&)>&& grad,
//                                     std::function<real(real, const real3&)>&& div,
//                                     std::function<real(real, const real3&)>&& lap)
// {
//     return {lua_mms{MOVE(call), MOVE(ddt), MOVE(grad), MOVE(div), MOVE(lap)}};
// }
} // namespace ccs
