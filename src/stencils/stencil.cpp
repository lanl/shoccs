#include "stencil.hpp"

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

namespace ccs::stencils
{
std::optional<stencil> stencil::from_lua(const sol::table& tbl)
{

    auto m = tbl["scheme"];
    if (!m.valid()) {
        spdlog::error("simulation.scheme must be specified");
        return std::nullopt;
    }

    std::vector<real> alpha{};
    if (auto as = m["alpha"]; as.valid()) {
        for (int i = 1; as[i].valid(); ++i) alpha.push_back(as[i].get<real>());
    }
    int order = m["order"].get_or(1); // default to first derivative schemes
    std::string type = m["type"].get_or(std::string{});

    if (order == 2) {
        if (type == "E2") {
            spdlog::info("E2 scheme chosen");
            return second::E2;
        }
        if (type == "E4") {
            spdlog::info("E4 scheme chosen");
            return second::E4;
        }
    } else if (order == 1) {
        if (type == "E2") {
            spdlog::info("E2 first scheme chosen");
            return make_E2_1(alpha);
        }
    }

    spdlog::error("scheme.order/type = {} / {} not recognized", order, type);
    return std::nullopt;
}
} // namespace ccs::stencils
