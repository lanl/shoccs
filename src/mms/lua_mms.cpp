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

} // namespace ccs
