#include "boundaries.hpp"

#include <string>

#include <sol/sol.hpp>

namespace ccs::bcs
{

static std::optional<type>
get_bc(const sol::table& tbl, const std::string& side, const logs& logger, int n = 0)
{
    if (auto bc = tbl[side]; bc.valid()) {
        auto t = bc.get_or(std::string{});

        auto check = [n, &side, &logger](auto success) {
            if (n == 1) {
                logger(spdlog::level::warn,
                       "ignoring bc specification on side {} due to domain extents.\n",
                       side);
                return Floating;
            }
            return success;
        };

        if (t == "dirichlet") {
            return check(Dirichlet);
        } else if (t == "floating") {
            return Floating;
        } else if (t == "neumann") {
            return check(Neumann);
        } else {
            logger(spdlog::level::err, "incorrect boundary condition");
            return std::nullopt;
        }
    }
    // a reasonable default?
    return Floating;
}

std::optional<std::pair<Grid, Object>>
from_lua(const sol::table& tbl, index_extents ex, const logs& logger)
{
    Grid g{ff, ff, ff}; // default to floating everwhere if nothing specified
    Object o{};
    if (auto m = tbl["domain_boundaries"]; m.valid()) {
        if (auto d = get_bc(m, "xmin", logger, ex[0]); d)
            g[0].left = *d;
        else
            return std::nullopt;

        if (auto d = get_bc(m, "xmax", logger, ex[0]); d)
            g[0].right = *d;
        else
            return std::nullopt;

        if (auto d = get_bc(m, "ymin", logger, ex[1]); d)
            g[1].left = *d;
        else
            return std::nullopt;

        if (auto d = get_bc(m, "ymax", logger, ex[1]); d)
            g[1].right = *d;
        else
            return std::nullopt;

        if (auto d = get_bc(m, "zmin", logger, ex[2]); d)
            g[2].left = *d;
        else
            return std::nullopt;

        if (auto d = get_bc(m, "zmax", logger, ex[2]); d)
            g[2].right = *d;
        else
            return std::nullopt;
    }

    // object bcs
    if (auto m = tbl["shapes"]; m.valid()) {
        for (int i = 1; m[i].valid(); i++) {
            auto d = get_bc(m[i], "boundary_condition", logger);
            if (d)
                o.push_back(*d);
            else
                return std::nullopt;
        }
    }

    return std::pair{g, o};
}
} // namespace ccs::bcs
