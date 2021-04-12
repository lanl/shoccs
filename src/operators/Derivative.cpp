#include "Derivative.hpp"
#include "fields/Selector.hpp"

#include <range/v3/view/chunk.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/join.hpp>

#include <cassert>
#include <iostream>

namespace ccs::operators
{

Derivative::Derivative(int dir,
                       const mesh::Mesh& mesh,
                       // real h,
                       // std::span<const mesh::Line> lines,
                       const stencils::Stencil& stencil,
                       const bcs::Grid& grid_bcs,
                       const bcs::Object& obj_bcs)
    : dir{dir}
{
    // query the stencil and allocate memory
    auto [p, rmax, tmax, ex_max] = stencil.query_max();
    auto h = mesh.h(dir);
    // set up the interior stencil
    interior_c.resize(2 * p + 1);
    stencil.interior(h, interior_c);

    // allocate maximum amount of memory required by any boundary conditions
    std::vector<real> left(rmax * tmax);
    std::vector<real> right(rmax * tmax);
    std::vector<real> extra(ex_max);

    auto B_builder = matrix::CSR::Builder();
    auto O_builder = matrix::Block::Builder();

    for (auto [stride, start, end] : mesh.lines(dir)) {
        // assert(offset == mesh.ic(start.mesh_coordinate));
        // skip derivatives along line of dirichlet bcs
        if (mesh.dirichlet_line(start.mesh_coordinate, dir, grid_bcs)) continue;

        // start with assumption of square matrix and adjust based on boundary conditions
        auto columns = end.mesh_coordinate[dir] - start.mesh_coordinate[dir] + 1;
        auto rows = columns;
        auto row_offset = mesh.ic(start.mesh_coordinate);
        auto col_offset = mesh.ic(start.mesh_coordinate);

        auto leftMat = matrix::Dense{};

        if (const auto& obj = start.object_boundary; obj) {
            const auto id = obj->objectID;
            assert(id < (integer)obj_bcs.size());
            const auto bc_t = obj_bcs[id];
            assert(bc_t == bcs::Dirichlet);

            auto&& [pLeft, rLeft, tLeft, exLeft] = stencil.query(bc_t);
            stencil.nbs(h, bc_t, obj->psi, false, left, extra);

            // change to allow something other than dirichlet
            leftMat = matrix::Dense{
                rLeft, tLeft - 1, left | vs::chunk(tLeft) | vs::for_each(vs::drop(1))};
            rows--;
            columns--;
            row_offset += stride;
            col_offset += stride;

            // add points to B
            auto b_coeffs = left | vs::stride(tLeft) | vs::take(rLeft);
            for (auto&& [row, val] : vs::enumerate(b_coeffs)) {
                B_builder.add_point(
                    row_offset + stride * row, obj->object_coordinate, val);
            }

        } else {
            auto&& [pLeft, rLeft, tLeft, exLeft] = stencil.query(grid_bcs[dir].left);
            stencil.nbs(h, grid_bcs[dir].left, 1.0, false, left, extra);

            leftMat = matrix::Dense{rLeft, tLeft, left};
            if (grid_bcs[dir].left == bcs::Dirichlet) {
                rows--;
                row_offset += stride;
            }
        }

        auto rightMat = matrix::Dense{};

        if (const auto& obj = end.object_boundary; obj) {
            const auto id = obj->objectID;
            assert(id < (integer)obj_bcs.size());
            const auto bc_t = obj_bcs[id];
            assert(bc_t == bcs::Dirichlet);

            auto&& [pRight, rRight, tRight, exRight] = stencil.query(bc_t);
            stencil.nbs(h, bc_t, obj->psi, true, right, extra);

            rightMat = matrix::Dense{rRight,
                                     tRight - 1,
                                     right | vs::chunk(tRight) |
                                         vs::for_each(vs::take(tRight - 1))};
            rows--;
            columns--;

            // add points to B
            integer boundary_offset =
                mesh.ic(end.mesh_coordinate) - (rRight + 1) * stride;
            auto b_coeffs =
                right | vs::drop(tRight - 1) | vs::stride(tRight) | vs::take(rRight);
            for (auto&& [row, val] : vs::enumerate(b_coeffs)) {
                B_builder.add_point(
                    boundary_offset + stride * row, obj->object_coordinate, val);
            }

        } else {
            auto&& [pRight, rRight, tRight, exRight] = stencil.query(grid_bcs[dir].right);
            stencil.nbs(h, grid_bcs[dir].right, 1.0, true, right, extra);

            rightMat = matrix::Dense{rRight, tRight, right};
            if (grid_bcs[dir].right == bcs::Dirichlet) { rows--; }
        }
        const integer n_interior = rows - leftMat.rows() - rightMat.rows();

        O_builder.add_InnerBlock(columns,
                                 row_offset,
                                 col_offset,
                                 stride,
                                 MOVE(leftMat),
                                 matrix::Circulant{n_interior, interior_c},
                                 MOVE(rightMat));

        // grab the left right portions of the line

        // if the left is a domain boundary all of it stays in a O (till neumann)

        // if the left is an object boundary the first column goes into B, the rest in O

        // if the right is a domain boundary all of it stays in O

        // if the right is an object boundary the last column goes into B, the rest in O
    }

    O = MOVE(O_builder).to_Block();
    B = MOVE(B_builder.to_CSR(mesh.size()));
}

void Derivative::operator()(field::ScalarView_Const u, field::ScalarView_Mutable du) const
{
    using namespace selector::scalar;
    du = 0;
    O(get<D>(u), get<D>(du));
    // This is ugly
    switch (dir) {
    case 0:
        B(get<Rx>(u), get<D>(du));
        break;
    case 1:
        B(get<Ry>(u), get<D>(du));
        break;
    default:
        B(get<Rz>(u), get<D>(du));
    }
}

} // namespace ccs::operators