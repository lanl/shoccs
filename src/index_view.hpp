#pragma once

#include "indexing.hpp"
#include <cppcoro/generator.hpp>

namespace ccs
{
template <int I = 2>
cppcoro::generator<int3> index_view(int3 extents)
{
    constexpr int F = index::dir<I>::fast;
    constexpr int S = index::dir<I>::slow;
    int3 ijk;

    const auto ni = extents[I];
    const auto nf = extents[F];
    const auto ns = extents[S];

    for (int s = 0; s < ns; s++) {
        ijk[S] = s;
        for (int f = 0; f < nf; f++) {
            ijk[F] = f;
            for (int i = 0; i < ni; i++) {
                ijk[I] = i;
                co_yield ijk;
            }
        }
    }
}

// index view of plane coordinates, if a negative value is specified,
// add extents[I] to it.  This allows for specifying plane boundaries with 0, -1
template <int I>
cppcoro::generator<int3> index_view(int3 extents, int i)
{
    constexpr int F = index::dir<I>::fast;
    constexpr int S = index::dir<I>::slow;
    int3 ijk;

    const auto nf = extents[F];
    const auto ns = extents[S];

    ijk[I] = i >= 0 ? i : i + extents[I];

    for (int s = 0; s < ns; s++) {
        ijk[S] = s;
        for (int f = 0; f < nf; f++) {
            ijk[F] = f;
            co_yield ijk;
        }
    }
}
} // namespace ccs