#include "Derivative.hpp"
#include "fields/Selector.hpp"

#include <range/v3/view/chunk.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/for_each.hpp>

namespace ccs::operators
{

Derivative::Derivative(int dir,
                       const mesh::Mesh& mesh,
                       // real h,
                       // std::span<const mesh::Line> lines,
                       const stencils::Stencil& stencil,
                       const bcs::Grid& grid_bcs,
                       const bcs::Object&)
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

    for (auto [offset, stride, start, end] : mesh.lines(dir)) {
        // skip derivatives along line of dirichlet bcs
        if (mesh.dirichlet_line(start.mesh_coordinate, dir, grid_bcs)) continue;

        auto columns = end.mesh_coordinate[dir] - start.mesh_coordinate[dir] + 1;

        auto&& [pLeft, rLeft, tLeft, exLeft] = stencil.query(grid_bcs[dir].left);
        stencil.nbs(h, grid_bcs[dir].left, 1.0, false, left, extra);

        auto leftMat = matrix::Dense{};
        if (grid_bcs[dir].left == bcs::Dirichlet) {
            leftMat = matrix::Dense{rLeft - 1,
                                    tLeft - 1,
                                    left | vs::chunk(tLeft) | vs::drop(1) |
                                        vs::for_each(vs::drop(1))};
            columns--;
            offset += stride;
        } else {
            leftMat = matrix::Dense{rLeft, tLeft, left};
        }

        auto&& [pRight, rRight, tRight, exRight] = stencil.query(grid_bcs[dir].right);
        stencil.nbs(h, grid_bcs[dir].right, 1.0, true, right, extra);

        auto rightMat = matrix::Dense{};
        if (grid_bcs[dir].right == bcs::Dirichlet) {
            rightMat = matrix::Dense{rRight - 1,
                                     tRight - 1,
                                     right | vs::chunk(tRight) |
                                         vs::for_each(vs::take(tRight - 1))};
            columns--;
        } else {
            rightMat = matrix::Dense{rRight, tRight, right};
        }

        const integer n_interior = columns - leftMat.rows() - rightMat.rows();

        O_builder.add_InnerBlock(columns,
                                 offset,
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
    B = MOVE(B_builder.to_CSR(0));
}

void Derivative::operator()(field::ScalarView_Const u, field::ScalarView_Mutable du) const
{
    using namespace selector::scalar;
    du = 0;
    O(get<D>(u), get<D>(du));
    // This is ugly
    switch (dir) {
    case 0:
        B(get<Rx>(u), get<Rx>(du));
        break;
    case 1:
        B(get<Ry>(u), get<Ry>(du));
        break;
    default:
        B(get<Rz>(u), get<Rz>(du));
    }
}

} // namespace ccs::operators