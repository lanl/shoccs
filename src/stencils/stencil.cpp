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

    int order = m["order"].get_or(1); // default to first derivative schemes
    std::string type = m["type"].get_or(std::string{});

    if (order == 2) {
        if (type == "E2") {
            return second::E2;
        } else {
            spdlog::error("scheme order/type combination not recognized");
            return std::nullopt;
        }
    } else {
        spdlog::error("scheme.order not recognized");
        return std::nullopt;
    }
}
} // namespace ccs::stencils
