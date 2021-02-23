#pragma once

#include "Cartesian.hpp"
#include "indexing.hpp"

#include <cppcoro/generator.hpp>

// Include some facilities here for traversing/viewing the mesh.  These functions
// make use of ranges and generators so they are separate from the main mesh class
// since most objects will need to know about the mesh but do not need to pay the
// compile penalty for including ranges.

namespace ccs::mesh
{

template <int I = 2>
cppcoro::generator<real3> location_view(const Cartesian& m)
{
    constexpr int F = index::dir<I>::fast;
    constexpr int S = index::dir<I>::slow;
    real3 loc;

    const auto iline = m.line(I);
    const auto fline = m.line(F);
    const auto sline = m.line(S);

    for (int s = 0; s < sline.n; s++) {
        loc[S] = sline.min + sline.h * s;
        for (int f = 0; f < fline.n; f++) {
            loc[F] = fline.min + fline.h * f;
            for (int i = 0; i < iline.n; i++) {
                loc[I] = iline.min + iline.h * i;
                co_yield loc;
            }
        }
    }
}

template <int I>
cppcoro::generator<real3> location_view(const Cartesian& m, int i)
{
    constexpr int F = index::dir<I>::fast;
    constexpr int S = index::dir<I>::slow;
    real3 loc;

    const auto iline = m.line(I);
    const auto fline = m.line(F);
    const auto sline = m.line(S);

    loc[I] = iline.min + iline.h * (i < 0 ? i + iline.n : i);

    for (int s = 0; s < sline.n; s++) {
        loc[S] = sline.min + sline.h * s;
        for (int f = 0; f < fline.n; f++) {
            loc[F] = fline.min + fline.h * f;
            co_yield loc;
        }
    }
}

} // namespace ccs::mesh