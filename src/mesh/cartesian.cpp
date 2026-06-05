#include "cartesian.hpp"

#include "fields/lazy_views.hpp"

#include <algorithm>

#include <fmt/core.h>
#include <sol/sol.hpp>

namespace ccs
{

cartesian::cartesian(span<const int> n, span<const real> min, span<const real> max)
{
    auto concat_copy = [](auto&& in, auto val, auto&& out) {
        int i = 0;
        for (auto it = std::ranges::begin(in); i < 3 && it != std::ranges::end(in);
             ++it, ++i)
            out[i] = *it;
        for (; i < 3; ++i)
            out[i] = val;
    };
    concat_copy(min, 0.0, min_);
    concat_copy(max, null_v<>, max_);

    int3& n_ = as_extents();
    {
        int i = 0;
        for (auto it = std::ranges::begin(n); i < 3 && it != std::ranges::end(n);
             ++it, ++i)
            n_[i] = *it > 0 ? *it : 1;
        for (; i < 3; ++i)
            n_[i] = 1;
    }

    for (int i = 0; i < 3; ++i)
        h_[i] = (n_[i] - 1) ? (max_[i] - min_[i]) / (n_[i] - 1) : null_v<>;

    dims_ = std::count_if(n_.begin(), n_.end(), [](auto n) { return !!(n - 1); });

    x_ = ccs::linear_distribute(min_[0], max_[0], n_[0]);
    y_ = ccs::linear_distribute(min_[1], max_[1], n_[1]);
    z_ = ccs::linear_distribute(min_[2], max_[2], n_[2]);
}

std::optional<std::pair<index_extents, domain_extents>>
cartesian::from_lua(const sol::table& tbl, const logs& logger)
{
    auto m = tbl["mesh"];
    if (!m.valid()) {
        logger(spdlog::level::err, "simulation.mesh is required");
        return std::nullopt;
    }

    auto ie = m["index_extents"];
    if (!ie.valid()) {
        logger(spdlog::level::err, "simulation.mesh.index_extents is required");
        return std::nullopt;
    }

    auto db = m["domain_bounds"];
    if (!db.valid()) {
        logger(spdlog::level::err, "simulation.mesh.domain_bounds is required");
        return std::nullopt;
    }

    int3 n{ie[1].get_or(1), ie[2].get_or(1), ie[3].get_or(1)};

    real3 lb{};
    real3 ub{};

    if (auto x = db["min"]; x.valid()) {
        lb = real3{x[1].get_or(0.0), x[2].get_or(0.0), x[3].get_or(0.0)};
    } else {
        logger(spdlog::level::info,
               "No explicit domain lower bound set.  Assuming (0, 0, 0)");
    }

    if (auto x = db["max"]; x.valid()) {
        ub = real3{x[1].get_or(lb[0]), x[2].get_or(lb[1]), x[3].get_or(lb[2])};
    } else if (db[1].valid()) {
        ub = real3{db[1].get_or(lb[0]), db[2].get_or(lb[1]), db[3].get_or(lb[2])};
    } else {
        logger(spdlog::level::err,
               "domain_bounds.max = {...} or domain_bounds = {...} must be specified");
        return std::nullopt;
    }

    return std::pair{index_extents{n}, domain_extents{lb, ub}};
}

} // namespace ccs
