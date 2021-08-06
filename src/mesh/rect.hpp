#pragma once

#include "indexing.hpp"
#include "real3_operators.hpp"
#include "shapes.hpp"
#include "types.hpp"

namespace ccs
{

template <int I>
struct rect {
    // corner positions in 'slow/fast' order
    real2 c0;
    real2 c1;

    real plane_coord;
    real fluid_normal; // +1, the normal points in in the +direction

    int id;

    std::optional<hit_info> hit(const ray& r, real t_min, real t_max) const
    {
        auto t = (plane_coord - r.origin[I]) / (r.direction[I]);

        if (t < t_min || t > t_max) return std::nullopt;

        const real3 p = r.position(t);
        auto s = index::dir<I>::slow;
        auto f = index::dir<I>::fast;

        if (p[s] < c0[0] || p[s] > c1[0] || p[f] < c0[1] || p[f] > c1[1])
            return std::nullopt;

        return hit_info{t, p, fluid_normal * r.direction[I] < 0, id};
    }

    real3 normal(const real3&) const
    {
        real3 n{};
        n[I] = fluid_normal;
        return n;
    }
};

} // namespace ccs
