#pragma once

#include "boundaries.hpp"
#include "fields/fields.hpp"
#include "geometry/geometry.hpp"
#include "matrices/block.hpp"
#include "matrices/csr.hpp"
#include "mesh/mesh.hpp"
#include "stencils/stencils.hpp"

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
// 2) UnCut cartesian mesh -> defined by the umesh_lines
// 3) cut-cell intersection points
// 4) solid points we can use to enforce bc's
//
// The result of applying this operator should be a range over the
// whole cartesian domain which one can then use a mapping
// to extract the boundary derivatives.  For this to work,
// The

class directional
{
    matrix::block O; // main operator
    matrix::csr B;   // field values for boundary conditions
    matrix::csr N;   // for Neumann
    std::vector<real> interior_c;
    // We only need the transformed coordinates of the solid points to map
    // boundary values to their proper location
    std::vector<int> spts;

public:
    directional() = default;

    directional(int dir,
                const stencil& st,
                const mesh& m,
                const geometry& g,
                domain_boundaries db,
                std::span<const boundary> object_b);

    void operator()(std::span<const real> f,
                    std::span<const real> df,
                    result_view result) const;

    const auto& solid_points() const { return spts; }
};
} // namespace ccs::op