#include "gauss.hpp"
#include "lua_mms.hpp"
#include "manufactured_solutions.hpp"

#include <sol/sol.hpp>

#include <string>

#include <range/v3/action/transform.hpp>

#include <cctype>

using namespace std::string_literals;

namespace ccs
{
manufactured_solution build_ms_gauss(int dims,
                                     std::span<const real3> center,
                                     std::span<const real3> variance,
                                     std::span<const real> amplitude,
                                     std::span<const real> frequency)
{
    switch (dims) {
    case 1:
        return build_ms_gauss1d(center, variance, amplitude, frequency);
    case 2:
        return build_ms_gauss2d(center, variance, amplitude, frequency);
    case 3:
        return build_ms_gauss3d(center, variance, amplitude, frequency);
    }
    return {};
}

std::optional<manufactured_solution>
manufactured_solution::from_lua(const sol::table& tbl, int dims, const logs& logger)
{
    auto ms = tbl["manufactured_solution"];
    if (!ms.valid()) {
        logger(spdlog::level::info, "manufactured_solution not specified.");
        return std::nullopt;
    }

    std::string ms_t = ms["type"].get_or(""s) |
                       rs::action::transform([](auto c) { return std::tolower(c); });

    if (ms_t == "gaussian") {
        logger(spdlog::level::info, "building gaussian manufactured solution");
        std::vector<real3> center{};
        std::vector<real3> variance{};
        std::vector<real> amplitude{};
        std::vector<real> frequency{};

        for (int i = 1; ms[i].valid(); ++i) {
            auto t = ms[i];
            center.push_back({t["center"][1].get_or(0.0),
                              t["center"][2].get_or(0.0),
                              t["center"][3].get_or(0.0)});
            variance.push_back({t["variance"][1].get_or(0.0),
                                t["variance"][2].get_or(0.0),
                                t["variance"][3].get_or(0.0)});
            amplitude.emplace_back(t["amplitude"].get_or(0.0));
            frequency.emplace_back(t["frequency"].get_or(0.0));
        }

        if (center.size() > 0)
            return build_ms_gauss(dims, center, variance, amplitude, frequency);

    } else if (ms_t == "lua") {
        logger(spdlog::level::info, "building lua manufactured solution");
        sol::optional<std::function<real(real, const real3&)>> call;
        sol::optional<std::function<real(real, const real3&)>> ddt;
        sol::optional<std::function<real3(real, const real3&)>> grad;
        sol::optional<std::function<real(real, const real3&)>> div;
        sol::optional<std::function<real(real, const real3&)>> lap;

        call = ms["call"];
        ddt = ms["ddt"];
        grad = ms["grad"];
        div = ms["div"];
        lap = ms["lap"];

        if (call && ddt && grad && div && lap) {
            return lua_mms::from_lua(ms);
        } else {
            logger(spdlog::level::err,
                   "`lua` manufactured solution requires call, ddt, grad, div, "
                   "lap functions");
        }

    } else {
        logger(spdlog::level::err,
               "Got manufactured_solution.type = `{}`.  Expected one of: `{}`",
               ms_t,
               "gaussian, lua");
    }

    return std::nullopt;
}
} // namespace ccs
