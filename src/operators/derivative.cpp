#include "derivative.hpp"
#include "fields/selector.hpp"

#include <range/v3/all.hpp>

#include <cassert>

namespace ccs
{

namespace
{

struct OB_builder {
    matrix::csr::builder O;
    matrix::csr::builder B;

    template <rs::random_access_range R>
    void add_cut_row(integer shape_row, integer solid_ic, integer stride, R&& r)
    {
        B.add_point(shape_row, shape_row, r[0]);
        for (auto&& [i, v] : vs::enumerate(r) | vs::drop(1))
            O.add_point(shape_row, solid_ic + i * stride, v);
    }

    template <rs::random_access_range R>
    void add_cut_point(integer shape_row, R&& r)
    {
        B.add_point(shape_row, shape_row, r[0]);
    }

    // this routine is currently no good for planar objects which result in interior
    // stencils where every point is a solid point
    template <rs::random_access_range R>
    void add_interp_row(integer shape_row,
                        real deriv_coeff,
                        R&& interp_coeffs,
                        const boundary& left,
                        const boundary& right,
                        integer stride,
                        const mesh& m)
    {
        auto left_ic = m.ic(left.mesh_coordinate);
        auto right_ic = m.ic(right.mesh_coordinate);

        auto it = rs::begin(interp_coeffs);
        // handle left point
        if (const auto& obj = left.object; obj)
            B.add_point(shape_row, obj->object_coordinate, deriv_coeff * *it);
        else
            O.add_point(shape_row, left_ic, deriv_coeff * *it);
        ++it;

        // handle interior
        for (int ic = left_ic + stride; ic < right_ic; ic += stride) {
            O.add_point(shape_row, ic, deriv_coeff * *it);
            ++it;
        }

        // handle right point
        if (const auto& obj = right.object; obj)
            B.add_point(shape_row, obj->object_coordinate, deriv_coeff * *it);
        else
            O.add_point(shape_row, right_ic, deriv_coeff * *it);
    }

    void to_csr(matrix::csr& O_matrix, matrix::csr& B_matrix, integer rows)
    {
        O_matrix = MOVE(O.to_csr(rows));
        B_matrix = MOVE(B.to_csr(rows));
    }
};

void cut_discretization(int r,
                        int dir,
                        const mesh& m,
                        const stencil& st,
                        const bcs::Grid& grid_bcs,
                        const bcs::Object& obj_bcs,
                        matrix::csr& O,
                        matrix::csr& B,
                        std::span<const real> interior)
{
    const auto shapes = m.R(r);
    const auto sz = shapes.size();

    if (sz == 0 || rs::accumulate(obj_bcs, true, [](auto&& acc, auto&& cur) {
            return acc && (cur == bcs::Dirichlet);
        }))
        return; // quick exit'

    auto [p, rmax, tmax, ex_max] = st.query_max();
    auto h = m.h(dir);

    OB_builder builder{};

    // allocate maximum amount of memory required by any boundary conditions
    std::vector<real> c(rmax * tmax);
    std::vector<real> interp_c(tmax);
    std::span<real> c_line;
    std::vector<real> extra(ex_max);
    auto stride = m.stride(dir);

    if (dir == r) {
        // no interpolation needed for this case
        for (auto&& [shape_row, obj] : vs::enumerate(shapes)) {
            auto bc_t = obj_bcs[obj.shape_id];
            // nothing to do for dirichlet
            if (bc_t == bcs::Dirichlet) continue;

            auto&& [pObj, rObj, tObj, exObj] = st.query(bc_t);
            st.nbs(h, bc_t, obj.psi, obj.ray_outside, c, extra);

            if (obj.ray_outside) {
                auto rng = c | vs::drop_exactly((rObj - 1) * tObj) |
                           vs::take_exactly(tObj) | vs::reverse;
                builder.add_cut_row(shape_row, m.ic(obj.solid_coord), -stride, rng);
            } else {
                auto rng = c | vs::take_exactly(tObj);
                builder.add_cut_row(shape_row, m.ic(obj.solid_coord), stride, rng);
            }
        }
    } else {
        for (auto&& [shape_row, obj] : vs::enumerate(shapes)) {
            auto bc_t = obj_bcs[obj.shape_id];
            // nothing to do for dirichlet
            if (bc_t == bcs::Dirichlet) continue;

            const bool right_interp_wall = obj.normal[dir] < 0.0;
            // get coefficient for derivative
            auto&& [pObj, rObj, tObj, exObj] = st.query(bc_t);
            st.nbs(h, bc_t, 1.0, right_interp_wall, c, extra);
            // get the line required for the interp point and reorder such that
            // the first point is the intersection point
            if (right_interp_wall) {
                c_line = std::span(rs::begin(c) + (rObj - 1) * tObj, tObj);
                rs::reverse(c_line);
            } else {
                c_line = std::span(rs::begin(c), tObj);
            }
            // get starting coordinates of closest point
            int3 cp = [&obj, &r]() {
                int3 ray = obj.solid_coord;
                if (obj.psi <= 0.5) ray[r] += 1 - 2 * obj.ray_outside;
                return ray;
            }();
            // The first point for all stencils will be the solid_coord.  Set the shift
            // accordingly
            const int cp_shift = right_interp_wall ? -1 : 1;
            // Set interp distance y for interp stencils in (i + y) format
            const real y = [&obj]() {
                real sign = obj.ray_outside ? 1.0 : -1.0;
                return sign * (obj.psi <= 0.5 ? obj.psi : obj.psi - 1);
            }();

            builder.add_cut_point(shape_row, c_line);

            for (int i = 1; i < tObj; i++) {
                cp[dir] += cp_shift;
                auto&& [r_stride, left_bounds, right_bounds] = m.interp_line(r, cp);
                auto&& [interp_v, left, right] =
                    st.interp(r, cp, y, left_bounds, right_bounds, interp_c);
                builder.add_interp_row(
                    shape_row, c_line[i], interp_v, left, right, r_stride, m);
            }
        }
    }

    // construct ray in 'dir` emanative from R(r)

    builder.to_csr(O, B, sz);
}

struct submatrix_size {
    integer rows, columns;
    integer row_offset, col_offset;
    integer stride, last_row;

    submatrix_size() = default;
    submatrix_size(int dir,
                   integer stride,
                   const boundary& start,
                   const boundary& end,
                   const mesh& m)
        : rows{end.mesh_coordinate[dir] - start.mesh_coordinate[dir] + 1},
          columns{this->rows},
          row_offset{m.ic(start.mesh_coordinate)},
          col_offset{this->row_offset},
          stride{stride},
          last_row{m.ic(end.mesh_coordinate)}
    {
    }

    void remove_left_row()
    {
        --rows;
        row_offset += stride;
    }

    void remove_left_row_col()
    {
        remove_left_row();
        --columns;
        col_offset += stride;
    }

    void remove_right_row() { --rows; }

    void remove_right_row_col()
    {
        remove_right_row();
        --columns;
    }

    integer left_row(integer row = 0) const { return row_offset + stride * row; }

    integer right_row(integer row = 0) const { return last_row + stride * row; }
};

void domain_discretization(int dir,
                           const mesh& m,
                           const stencil& st,
                           const bcs::Grid& grid_bcs,
                           const bcs::Object& obj_bcs,
                           matrix::block& O,
                           matrix::csr& B,
                           matrix::csr& N,
                           std::span<const real> interior)
{
    // query the stencil and allocate memory
    auto [p, rmax, tmax, ex_max] = st.query_max();
    auto h = m.h(dir);

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
        auto sub = submatrix_size{dir, stride, start, end, m};

        auto leftMat = matrix::dense{};

        if (const auto& obj = start.object; obj) {
            const auto id = obj->objectID;
            assert(id < (integer)obj_bcs.size());
            const auto bc_t = obj_bcs[id];

            auto&& [pLeft, rLeft, tLeft, exLeft] = st.query(bc_t);
            st.nbs(h, bc_t, obj->psi, false, left, extra);

            // change to allow something other than dirichlet
            // In the case of non-dirichlet bc's on the object, we need to skip the
            // first row as it will be handled in the Rx/y/z operators
            int s = bc_t != bcs::Dirichlet;
            rLeft -= s;
            auto lc = left | vs::drop(s * tLeft);
            leftMat = matrix::dense{
                rLeft, tLeft - 1, lc | vs::chunk(tLeft) | vs::for_each(vs::drop(1))};

            sub.remove_left_row_col();

            // add points to B
            auto b_coeffs = lc | vs::stride(tLeft) | vs::take(rLeft);
            for (auto&& [row, val] : vs::enumerate(b_coeffs)) {
                B_builder.add_point(sub.left_row(row), obj->object_coordinate, val);
            }

        } else {
            auto&& [pLeft, rLeft, tLeft, exLeft] = st.query(grid_bcs[dir].left);
            st.nbs(h, grid_bcs[dir].left, 1.0, false, left, extra);

            leftMat = matrix::dense{rLeft, tLeft, left};
            if (grid_bcs[dir].left == bcs::Dirichlet) {
                sub.remove_left_row();
            } else if (grid_bcs[dir].left == bcs::Neumann) {
                // add data to N matrix
                for (int row = 0; row < exLeft; row++) {
                    N_builder.add_point(sub.left_row(row), sub.left_row(), extra[row]);
                }
            }
        }

        auto rightMat = matrix::dense{};

        if (const auto& obj = end.object; obj) {
            const auto id = obj->objectID;
            assert(id < (integer)obj_bcs.size());
            const auto bc_t = obj_bcs[id];

            auto&& [pRight, rRight, tRight, exRight] = st.query(bc_t);
            st.nbs(h, bc_t, obj->psi, true, right, extra);

            integer s = bc_t != bcs::Dirichlet;
            rRight -= s;
            auto rc = right | vs::take_exactly(rRight * tRight);

            rightMat = matrix::dense{rRight,
                                     tRight - 1,
                                     rc | vs::chunk(tRight) |
                                         vs::for_each(vs::take(tRight - 1))};
            sub.remove_right_row_col();

            // add points to B
            auto b_coeffs =
                rc | vs::drop(tRight - 1) | vs::stride(tRight) | vs::take(rRight);
            for (auto&& [row, val] : vs::enumerate(b_coeffs)) {
                B_builder.add_point(
                    sub.right_row(row - rRight), obj->object_coordinate, val);
            }

        } else {
            auto&& [pRight, rRight, tRight, exRight] = st.query(grid_bcs[dir].right);
            st.nbs(h, grid_bcs[dir].right, 1.0, true, right, extra);

            rightMat = matrix::dense{rRight, tRight, right};
            if (grid_bcs[dir].right == bcs::Dirichlet) {
                sub.remove_right_row();
            } else if (grid_bcs[dir].right == bcs::Neumann) {
                for (int row = 0; row < exRight; row++) {
                    N_builder.add_point(
                        sub.right_row(row - exRight + 1), sub.right_row(), extra[row]);
                }
            }
        }
        const integer n_interior = sub.rows - leftMat.rows() - rightMat.rows();

        O_builder.add_inner_block(sub.columns,
                                  sub.row_offset,
                                  sub.col_offset,
                                  stride,
                                  MOVE(leftMat),
                                  matrix::circulant{n_interior, interior},
                                  MOVE(rightMat));
    }

    O = MOVE(O_builder).to_block();
    B = MOVE(B_builder.to_csr(m.size()));
    N = MOVE(N_builder.to_csr(m.size()));
}
} // namespace

derivative::derivative(int dir,
                       const mesh& m,
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

    domain_discretization(dir, m, st, grid_bcs, obj_bcs, O, B, N, interior_c);

    cut_discretization(0, dir, m, st, grid_bcs, obj_bcs, Bfx, Brx, interior_c);
    cut_discretization(1, dir, m, st, grid_bcs, obj_bcs, Bfy, Bry, interior_c);
    cut_discretization(2, dir, m, st, grid_bcs, obj_bcs, Bfz, Brz, interior_c);
}

template <typename Op>
    requires(!Scalar<Op>)
void derivative::operator()(scalar_view u, scalar_span du, Op op) const
{
    using namespace si;

    // update points in R
    Bfx(get<D>(u), get<Rx>(du));
    Bfy(get<D>(u), get<Ry>(du));
    Bfz(get<D>(u), get<Rz>(du));

    Brx(get<Rx>(u), get<Rx>(du));
    Bry(get<Ry>(u), get<Ry>(du));
    Brz(get<Rz>(u), get<Rz>(du));

    // update fluid domain
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
    requires(!Scalar<Op>)
void derivative::operator()(scalar_view u, scalar_view nu, scalar_span du, Op op) const
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
