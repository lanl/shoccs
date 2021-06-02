#pragma once

#include "types.hpp"
#include <functional>
#include <optional>
#include <sol/sol.hpp>

namespace ccs
{
class manufactured_solution;

struct lua_mms {
    sol::table tbl;

    std::function<real(real, const real3&)> call_;
    std::function<real(real, const real3&)> ddt_;
    std::function<std::tuple<real, real, real>(real, const real3&)> gradient_;
    std::function<real(real, const real3&)> divergence_;
    std::function<real(real, const real3&)> laplacian_;

    lua_mms() = default;
    lua_mms(const sol::table& tbl);

    lua_mms(const lua_mms& other)
        : tbl{other.tbl},
          call_{tbl["call"]},
          ddt_{tbl["ddt"]},
          gradient_{tbl["grad"]},
          divergence_{tbl["div"]},
          laplacian_{tbl["lap"]}
    {
    }

    lua_mms(lua_mms&& other)
        : tbl{MOVE(other.tbl)},
          call_{tbl["call"]},
          ddt_{tbl["ddt"]},
          gradient_{tbl["grad"]},
          divergence_{tbl["div"]},
          laplacian_{tbl["lap"]}
    {
    }

    lua_mms& operator=(const lua_mms& other)
    {
        tbl = other.tbl;
        call_ = tbl["call"];
        ddt_ = tbl["ddt"];
        gradient_ = tbl["grad"];
        divergence_ = tbl["div"];
        laplacian_ = tbl["lap"];
        return *this;
    }

    lua_mms& operator=(lua_mms&& other)
    {
        tbl = MOVE(other.tbl);
        call_ = tbl["call"];
        ddt_ = tbl["ddt"];
        gradient_ = tbl["grad"];
        divergence_ = tbl["div"];
        laplacian_ = tbl["lap"];
        return *this;
    }

    auto operator()(real time, const real3& loc) const { return call_(time, loc); }
    auto ddt(real time, const real3& loc) const { return ddt_(time, loc); }
    real3 gradient(real time, const real3& loc) const
    {
        auto&& tp = gradient_(time, loc);
        return {std::get<0>(tp), std::get<1>(tp), std::get<2>(tp)};
    }
    auto divergence(real time, const real3& loc) const { return divergence_(time, loc); }
    auto laplacian(real time, const real3& loc) const { return laplacian_(time, loc); }

    static std::optional<manufactured_solution> from_lua(const sol::table& tbl);
};

} // namespace ccs
