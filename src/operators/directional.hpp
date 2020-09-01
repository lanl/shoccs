#pragma once

#include "boundaries.hpp"
#include "geometry/geometry.hpp"
#include "matrices/block.hpp"
#include "matrices/csr.hpp"
#include "mesh/mesh.hpp"
#include "stencils/stencils.hpp"
#include "types.hpp"

#include <functional>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/zip.hpp>

namespace ccs::op
{
// a directional operator (i.e. d/dx) - a building block for a full operator (i.e. grad)
//
// Q. should the domain boundaries be handled via csr in the same way as embedded
// object boundaries?  For now we treat them differently and do not put them
// in the boundary operator
//
// To form the operator we need
// 1) The stencil -> boundary and interior stencils
// 2) UnCut Cartesian mesh -> defined by the umesh_lines
// 3) cut-cell intersection points
// 4) solid points we can use to enforce bc's
//
// The result of applying this operator should be a range over the
// whole cartesian domain which one can then use a mapping
// to extract the boundary derivatives.  For this to work,
// The

class directional
{
    std::vector<int> offsets;
    std::vector<int> zeros;
    std::vector<matrix::block> O;
    matrix::csr B;
    std::vector<real> interior_c;

public:
    directional() = default;

    directional(int dir,
                const stencil& st,
                const mesh& m,
                const geometry& g,
                domain_boundaries db,
                std::span<const boundary> object_b);

    template <ranges::random_access_range R>
    auto operator()(R&& rng)
    {
        using namespace ranges::views;
        // is returning a range worth it here?  If we simply take a "out" span we can
        // write directly to it and do away with the explicit zeros here.  Our output
        // range is no longer a random access range either
        return zip_with(
            std::plus{},
            concat(repeat_n(0.0, zeros[0]),
                   for_each(zip(offsets, O, zeros | drop(1)),
                            [rng](auto&& t) {
                                auto&& [off, mat, nz] = t;
                                return ranges::yield_from(
                                    concat(mat * (rng | drop(off)), repeat_n(0.0, nz)));
                            })),
            B * rng);
    }
};
} // namespace ccs::op