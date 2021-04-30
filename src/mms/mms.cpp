#include "manufactured_solutions.hpp"
#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

#include <string>

#include <range/v3/action/transform.hpp>

#include <cctype>

using namespace std::string_literals;

namespace ccs
{
std::optional<ManufacturedSolution> ManufacturedSolution::from_lua(const sol::table& tbl,
                                                                   int dims)
{
    auto ms = tbl["manufactured_solution"];
    if (!ms.valid()) {
        spdlog::info("manufactured_solution not specified.");
        return std::nullopt;
    }

    std::string ms_t =
        ms["type"].get_or(""s) | rs::action::transform([](auto c) { return std::tolower(c); });

    if (ms_t == "gaussian") {
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

    } else {
        spdlog::error("Got manufactured_solution.type = `{}`.  Expected one of: `{}`",
                      ms_t,
                      "gaussian");
    }

    return std::nullopt;
}
} // namespace ccs