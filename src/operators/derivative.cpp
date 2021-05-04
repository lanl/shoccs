#include "derivative.hpp"
#include "fields/selector.hpp"

#include <range/v3/view/chunk.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/join.hpp>

#include <cassert>

namespace ccs
{

derivative::derivative(int dir,
                       const mesh& m,
                       // real h,
                       // std::span<const mesh::line> lines,
                       const stencil& st,
                       const bcs::Grid& grid_bcs,
                       const bcs::Object& obj_bcs)
    : dir{dir}
{
    // query the stencil and allocate memory
    auto [p, rmax, tmax, ex_max] = st.query_max();
    auto h = m.h(dir);
    // set up the interior stencil
    interior_c.resize(2 * p + 1);
    st.interior(h, interior_c);

    // allocate maximum amount of memory required by any boundary conditions
    std::vector<real> left(rmax * tmax);
    std::vector<real> right(rmax * tmax);
    std::vector<real> extra(ex_max);

    auto B_builder = matrix::csr::builder();
    auto O_builder = matrix::block::builder();
    auto N_builder = matrix::csr::builder();

    for (auto [stride, start, end] : m.lines(dir)) {
        // assert(offset == m.ic(start.m_coordinate));
        // skip derivatives along line of dirichlet bcs
        if (m.dirichlet_line(start.mesh_coordinate, dir, grid_bcs)) continue;

        // start with assumption of square matrix and adjust based on boundary conditions
        auto columns = end.mesh_coordinate[dir] - start.mesh_coordinate[dir] + 1;
        auto rows = columns;
        auto row_offset = m.ic(start.mesh_coordinate);
        auto col_offset = m.ic(start.mesh_coordinate);

        auto leftMat = matrix::dense{};

        if (const auto& obj = start.object; obj) {
            const auto id = obj->objectID;
            assert(id < (integer)obj_bcs.size());
            const auto bc_t = obj_bcs[id];
            assert(bc_t == bcs::Dirichlet);

            auto&& [pLeft, rLeft, tLeft, exLeft] = st.query(bc_t);
            st.nbs(h, bc_t, obj->psi, false, left, extra);

            // change to allow something other than dirichlet
            leftMat = matrix::dense{
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
            auto&& [pLeft, rLeft, tLeft, exLeft] = st.query(grid_bcs[dir].left);
            st.nbs(h, grid_bcs[dir].left, 1.0, false, left, extra);

            leftMat = matrix::dense{rLeft, tLeft, left};
            if (grid_bcs[dir].left == bcs::Dirichlet) {
                rows--;
                row_offset += stride;
            } else if (grid_bcs[dir].left == bcs::Neumann) {
                // add data to N matrix
                for (int row = 0; row < exLeft; row++) {
                    N_builder.add_point(
                        row_offset + stride * row, row_offset, extra[row]);
                }
            }
        }

        auto rightMat = matrix::dense{};

        if (const auto& obj = end.object; obj) {
            const auto id = obj->objectID;
            assert(id < (integer)obj_bcs.size());
            const auto bc_t = obj_bcs[id];
            assert(bc_t == bcs::Dirichlet);

            auto&& [pRight, rRight, tRight, exRight] = st.query(bc_t);
            st.nbs(h, bc_t, obj->psi, true, right, extra);

            rightMat = matrix::dense{rRight,
                                     tRight - 1,
                                     right | vs::chunk(tRight) |
                                         vs::for_each(vs::take(tRight - 1))};
            rows--;
            columns--;

            // add points to B
            integer boundary_offset = m.ic(end.mesh_coordinate) - rRight * stride;
            auto b_coeffs =
                right | vs::drop(tRight - 1) | vs::stride(tRight) | vs::take(rRight);
            for (auto&& [row, val] : vs::enumerate(b_coeffs)) {
                B_builder.add_point(
                    boundary_offset + stride * row, obj->object_coordinate, val);
            }

        } else {
            auto&& [pRight, rRight, tRight, exRight] = st.query(grid_bcs[dir].right);
            st.nbs(h, grid_bcs[dir].right, 1.0, true, right, extra);

            rightMat = matrix::dense{rRight, tRight, right};
            if (grid_bcs[dir].right == bcs::Dirichlet) {
                rows--;
            } else if (grid_bcs[dir].right == bcs::Neumann) {
                auto ic = m.ic(end.mesh_coordinate);
                integer boundary_offset = ic - (exRight - 1) * stride;

                for (int row = 0; row < exRight; row++) {
                    N_builder.add_point(boundary_offset + stride * row, ic, extra[row]);
                }
            }
        }
        const integer n_interior = rows - leftMat.rows() - rightMat.rows();

        O_builder.add_inner_block(columns,
                                  row_offset,
                                  col_offset,
                                  stride,
                                  MOVE(leftMat),
                                  matrix::circulant{n_interior, interior_c},
                                  MOVE(rightMat));
    }

    O = MOVE(O_builder).to_block();
    B = MOVE(B_builder.to_csr(m.size()));
    N = MOVE(N_builder.to_csr(m.size()));
}

template <typename Op>
requires(!Scalar<Op>) void derivative::operator()(scalar_view u,
                                                  scalar_span du,
                                                  Op op) const
{
    using namespace si;
    O(get<D>(u), get<D>(du), op);
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

template <typename Op>
requires(!Scalar<Op>) void derivative::operator()(scalar_view u,
                                                  scalar_view nu,
                                                  scalar_span du,
                                                  Op op) const
{
    using namespace si;

    (*this)(u, du, op);
    N(get<D>(nu), get<D>(du));
}

template void derivative::operator()<eq_t>(scalar_view, scalar_span, eq_t) const;

template void
derivative::operator()<plus_eq_t>(scalar_view, scalar_span, plus_eq_t) const;

template void
derivative::operator()<eq_t>(scalar_view, scalar_view, scalar_span, eq_t) const;

template void
derivative::operator()<plus_eq_t>(scalar_view, scalar_view, scalar_span, plus_eq_t) const;

} // namespace ccs