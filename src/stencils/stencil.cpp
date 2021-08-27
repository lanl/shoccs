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

    // experimenting with different kinds of schemes extends the 'kinds' of
    // free parameters needed.  If we ever figure out what we're doing then
    // this should be cleaned up
    auto read_alpha = [&m](auto&& str, auto& v) {
        if (auto as = m[str]; as.valid()) {
            for (int i = 1; as[i].valid(); ++i) v.push_back(as[i].template get<real>());
        }
    };

    std::vector<real> alpha{}, floating_alpha{}, dirichlet_alpha{};
    read_alpha("alpha", alpha);
    read_alpha("floating_alpha", floating_alpha);
    read_alpha("dirichlet_alpha", dirichlet_alpha);

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
        if (type == "E2-poly") {
            spdlog::info("E2-poly scheme chosen");
            if (alpha.size() > 0) {
                spdlog::info("alpha = {}", fmt::join(alpha, ", "));
                // make the dubious assumption that the user specified the floating +
                // dirichlet alpha in the single array - useful for the optimizer
                // interface
                auto a = std::span{alpha};
                return make_polyE2_1(a.subspan(0, 6), a.subspan(6));
            } else {
                spdlog::info("floating alpha = {}", fmt::join(floating_alpha, ", "));
                spdlog::info("dirichlet alpha = {}", fmt::join(dirichlet_alpha, ", "));

                return make_polyE2_1(floating_alpha, dirichlet_alpha);
            }
        }
    }

    spdlog::error("scheme.order/type = {} / {} not recognized", order, type);
    return std::nullopt;
}
} // namespace ccs::stencils
