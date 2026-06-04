#include "derivative.hpp"

#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <algorithm>
#include <cassert>
#include <ranges>
#include <span>
#include <vector>

namespace ccs
{

namespace
{

struct interp_deriv_info {
    std::span<const real> c;
    int offset;
};

interp_deriv_info interp_deriv_coefficients(int dir,
                                            int n,
                                            real h,
                                            const mesh_object_info& obj,
                                            bcs::type bc,
                                            const stencil& st,
                                            std::span<real> c,
                                            std::span<real> ex)
{
    auto&& [p, r, t, _] = st.query(bc);

    if (std::abs(obj.normal[dir]) > 1e-12) {

        const bool right_interp_wall = obj.normal[dir] < 0.0;
        // get coefficient for derivative
        st.nbs(h, bc, 1.0, right_interp_wall, c, ex);
        // get the line required for the interp point and reorder such that
        // the first point is the intersection point
        if (right_interp_wall) {
            return {c.subspan((r - 1) * t, t), (1 - t)};
        } else {
            return {c.subspan(0, t), 0};
        }
    } else {
        // Here we are assuming convex only bodies which means the only
        // thing to consider here is if the point is near a domain wall
        auto right_dist = n - obj.solid_coord[dir];
        auto left_dist = obj.solid_coord[dir];
        assert(1 + right_dist + left_dist > t);

        if (right_dist >= p && left_dist >= p) {
            // do an interior stencil
            st.interior(h, c);
            return {c.subspan(0, 2 * p + 1), -p};
        } else {
            // do a boundary stencil
            const bool right_interp_wall = right_dist < left_dist;
            st.nbs(h, bc, 1.0, right_interp_wall, c, ex);

            if (right_interp_wall) {
                auto q = 1 + right_dist;
                return {c.subspan((r - q) * t, t), q - t};
            } else {
                auto q = left_dist;
                return {c.subspan(q * t, t), -q};
            }
        }
    }
}

struct OB_builder {
    matrix::csr::builder O;
    matrix::csr::builder B;

    template <std::ranges::random_access_range R>
    void add_cut_row(integer shape_row, integer solid_ic, integer stride, R&& r)
    {
        B.add_point(shape_row, shape_row, r[0]);
        for (int i = 1; i < std::ranges::ssize(r); ++i)
            O.add_point(shape_row, solid_ic + i * stride, r[i]);
    }

    void add_cut_point(integer shape_row, real u)
    {
        B.add_point(shape_row, shape_row, u);
    }

    // this routine is currently no good for planar objects which result in interior
    // stencils where every point is a solid point
    template <std::ranges::random_access_range R>
    void add_interp_row(integer shape_row,
                        real deriv_coeff,
                        R&& interp_coeffs,
                        const boundary& left,
                        const boundary& right,
                        integer stride,
                        const mesh& m,
                        std::string& msg)
    {
        const auto left_ic = m.ic(left.mesh_coordinate);
        const auto right_ic = m.ic(right.mesh_coordinate);

        auto it = std::ranges::begin(interp_coeffs);
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

        if (msg.size()) {
            if (!(left.object || right.object))
                msg += fmt::format(",{},{}", -1, 1.);
            else if (left.object)
                msg += fmt::format(
                    ",{},{}", left.object->object_coordinate, left.object->psi);
            else
                msg += fmt::format(
                    ",{},{}", right.object->object_coordinate, right.object->psi);
        }
    }

    void to_csr(integer r, matrix::csr& O_matrix, matrix::csr& B_matrix, integer rows)
    {
        O_matrix = MOVE(O.to_csr(rows));
        B_matrix = MOVE(B.to_csr(rows));

        // adjust row/col space flags
        // rowspace for both:
        // r = 0 -> rx == 1
        // r = 1 -> ry == 2
        // r = 2 -> rz == 4
        flag row = 1u << r;

        // colspace for O is fluid

        // colspace for B is same as rowspace
        flag col_b = row;

        O_matrix.flags(row << row_shift);
        B_matrix.flags((row << row_shift) | col_b);
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
                        std::span<const real> interior,
                        const logs& logger)
{
    const auto shapes = m.R(r);
    const auto sz = shapes.size();

    if (sz == 0 ||
        std::ranges::all_of(obj_bcs, [](auto bc) { return bc == bcs::Dirichlet; }))
        return; // quick exit'

    auto [p, rmax, tmax, ex_max] = st.query_max();
    auto h = m.h(dir);

    OB_builder builder{};

    // allocate maximum amount of memory required by any boundary conditions
    std::vector<real> c(rmax * tmax);
    std::vector<real> interp_c(tmax);
    std::vector<real> extra(ex_max);
    auto stride = m.stride(dir);

    if (dir == r) {
        // no interpolation needed for this case
        for (integer shape_row = 0; shape_row < (integer)sz; ++shape_row) {
            const auto& obj = shapes[shape_row];
            auto bc_t = obj_bcs[obj.shape_id];
            // nothing to do for dirichlet
            if (bc_t == bcs::Dirichlet) continue;

            auto&& [pObj, rObj, tObj, exObj] = st.query(bc_t);
            st.nbs(h, bc_t, obj.psi, obj.ray_outside, c, extra);

            if (obj.ray_outside) {
                auto sub = std::span{c}.subspan((rObj - 1) * tObj, tObj);
                std::vector<real> rng(sub.begin(), sub.end());
                std::ranges::reverse(rng);
                builder.add_cut_row(shape_row, m.ic(obj.solid_coord), -stride, rng);
            } else {
                auto rng = std::span{c}.subspan(0, tObj);
                builder.add_cut_row(shape_row, m.ic(obj.solid_coord), stride, rng);
            }
        }
    } else {
        for (integer shape_row = 0; shape_row < (integer)sz; ++shape_row) {
            const auto& obj = shapes[shape_row];
            auto bc_t = obj_bcs[obj.shape_id];
            // nothing to do for dirichlet
            if (bc_t == bcs::Dirichlet) continue;

            auto [c_line, cp_shift] = interp_deriv_coefficients(
                dir, m.extents()[dir] - 1, h, obj, bc_t, st, c, extra);

            // get starting coordinates of closest point
            int3 cp = [&obj, r, dir, cp_shift]() {
                int3 ray = obj.solid_coord;
                if (obj.psi <= 0.5) ray[r] += 1 - 2 * obj.ray_outside;
                ray[dir] += cp_shift;
                return ray;
            }();

            // Set interp distance y for interp stencils in (i + y) format
            const real y = [&obj]() {
                real sign = obj.ray_outside ? 1.0 : -1.0;
                return sign * (obj.psi <= 0.5 ? obj.psi : obj.psi - 1);
            }();

            // prepare the log message if we are logging
            std::string msg{};
            if (logger)
                msg = fmt::format("{},{},{},{},{}", dir, r, shape_row, y, obj.psi);

            for (auto&& v : c_line) {
                if (cp[dir] == obj.solid_coord[dir]) {
                    builder.add_cut_point(shape_row, v);
                } else {
                    auto&& [r_stride, left_bounds, right_bounds] = m.interp_line(r, cp);
                    auto&& [interp_v, left, right] =
                        st.interp(r, cp, y, left_bounds, right_bounds, interp_c);
                    builder.add_interp_row(
                        shape_row, v, interp_v, left, right, r_stride, m, msg);
                }
                ++cp[dir];
            }

            logger(spdlog::level::info, msg);
        }
    }

    // construct ray in 'dir` emanative from R(r)
    builder.to_csr(r, O, B, sz);
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
            auto lc = std::span{left}.subspan(s * tLeft);

            // Build dense matrix: skip first column of each row
            std::vector<real> dense_data;
            dense_data.reserve(rLeft * (tLeft - 1));
            for (int row = 0; row < rLeft; ++row) {
                auto row_span = lc.subspan(row * tLeft + 1, tLeft - 1);
                dense_data.insert(dense_data.end(), row_span.begin(), row_span.end());
            }
            leftMat = matrix::dense{rLeft, tLeft - 1, dense_data};

            sub.remove_left_row_col();

            // add points to B (first element of each row = stride by tLeft)
            for (int row = 0; row < rLeft; ++row) {
                B_builder.add_point(sub.left_row(row), obj->object_coordinate, lc[row * tLeft]);
            }

        } else {
            auto&& [pLeft, rLeft, tLeft, exLeft] = st.query(grid_bcs[dir].left);
            st.nbs(h, grid_bcs[dir].left, 1.0, false, left, extra);

            leftMat = matrix::dense{rLeft, tLeft, left};
            if (grid_bcs[dir].left == bcs::Dirichlet) {
                sub.remove_left_row();
                leftMat.flags(ldd);
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
            auto rc = std::span{right}.subspan(0, rRight * tRight);

            // Build dense matrix: take first (tRight-1) columns of each row
            std::vector<real> dense_data;
            dense_data.reserve(rRight * (tRight - 1));
            for (int row = 0; row < rRight; ++row) {
                auto row_span = rc.subspan(row * tRight, tRight - 1);
                dense_data.insert(dense_data.end(), row_span.begin(), row_span.end());
            }
            rightMat = matrix::dense{rRight, tRight - 1, dense_data};
            sub.remove_right_row_col();

            // add points to B (last element of each row)
            for (int row = 0; row < rRight; ++row) {
                auto val = rc[row * tRight + tRight - 1];
                B_builder.add_point(
                    sub.right_row(row - rRight), obj->object_coordinate, val);
            }

        } else {
            auto&& [pRight, rRight, tRight, exRight] = st.query(grid_bcs[dir].right);
            st.nbs(h, grid_bcs[dir].right, 1.0, true, right, extra);

            rightMat = matrix::dense{rRight, tRight, right};
            if (grid_bcs[dir].right == bcs::Dirichlet) {
                sub.remove_right_row();
                rightMat.flags(rdd);
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

    // col_space of B is `R{dir}`
    // 0 -> rx == 1
    // 1 -> ry == 2
    // 2 -> rz == 4
    B.flags(1u << dir);
}
} // namespace

derivative::derivative(int dir,
                       const mesh& m,
                       const stencil& st,
                       const bcs::Grid& grid_bcs,
                       const bcs::Object& obj_bcs,
                       const logs& logger)
    : dir{dir}
{
    if (m.extents()[dir] < 2) return;
    // query the stencil and allocate memory
    auto [p, rmax, tmax, ex_max] = st.query_max();
    auto h = m.h(dir);
    // set up the interior stencil
    auto interior_c = std::vector<real>(2 * p + 1);
    st.interior(h, interior_c);

    domain_discretization(dir, m, st, grid_bcs, obj_bcs, O, B, N, interior_c);

    cut_discretization(0, dir, m, st, grid_bcs, obj_bcs, Bfx, Brx, interior_c, logger);
    cut_discretization(1, dir, m, st, grid_bcs, obj_bcs, Bfy, Bry, interior_c, logger);
    cut_discretization(2, dir, m, st, grid_bcs, obj_bcs, Bfz, Brz, interior_c, logger);
}

template <typename Op>
    requires std::invocable<Op, real&, real>
void derivative::apply_kernels(scalar_view u, scalar_span du, Op op) const
{
    // update points in R
    Bfx(u.D, du.Rx);
    Bfy(u.D, du.Ry);
    Bfz(u.D, du.Rz);

    Brx(u.Rx, du.Rx);
    Bry(u.Ry, du.Ry);
    Brz(u.Rz, du.Rz);

    // update fluid domain
    O(u.D, du.D, op);
    switch (dir) {
    case 0:
        B(u.Rx, du.D);
        break;
    case 1:
        B(u.Ry, du.D);
        break;
    default:
        B(u.Rz, du.D);
    }
}

template <typename Op>
    requires std::invocable<Op, real&, real>
void derivative::operator()(scalar_view u, scalar_span du, Op op) const
{
    Kokkos::Profiling::ScopedRegion region("derivative::operator()");
    apply_kernels(u, du, op);
    Kokkos::fence("derivative::operator() complete");
}

template <typename Op>
    requires std::invocable<Op, real&, real>
void derivative::operator()(scalar_view u, scalar_view nu, scalar_span du, Op op) const
{
    Kokkos::Profiling::ScopedRegion region("derivative::operator()");
    apply_kernels(u, du, op);
    N(nu.D, du.D);
    Kokkos::fence("derivative::operator() with Neumann complete");
}

template <typename Op>
    requires std::invocable<Op, real&, real>
void derivative::build_graph(scalar_view u, scalar_span du, Op op)
{
    const real* u_D = u.D.data();
    const real* u_Rx = u.Rx.data();
    const real* u_Ry = u.Ry.data();
    const real* u_Rz = u.Rz.data();
    real* du_D = du.D.data();
    real* du_Rx = du.Rx.data();
    real* du_Ry = du.Ry.data();
    real* du_Rz = du.Rz.data();
    const real* b_src = (dir == 0) ? u_Rx : (dir == 1) ? u_Ry : u_Rz;

    graph_ = Kokkos::Experimental::create_graph<execution_space>(
        [&](auto root) {
            // R-space chains (3 independent pairs)
            auto bfx = Bfx.graph_node(root, u_D, du_Rx);
            Brx.graph_node(bfx, u_Rx, du_Rx);

            auto bfy = Bfy.graph_node(root, u_D, du_Ry);
            Bry.graph_node(bfy, u_Ry, du_Ry);

            auto bfz = Bfz.graph_node(root, u_D, du_Rz);
            Brz.graph_node(bfz, u_Rz, du_Rz);

            // D-space chain
            auto o = O.graph_node(root, u_D, du_D, op);
            B.graph_node(o, b_src, du_D);
        });

    graph_->instantiate();
}

template <typename Op>
    requires std::invocable<Op, real&, real>
void derivative::build_graph(scalar_view u, scalar_view nu, scalar_span du, Op op)
{
    const real* u_D = u.D.data();
    const real* u_Rx = u.Rx.data();
    const real* u_Ry = u.Ry.data();
    const real* u_Rz = u.Rz.data();
    const real* nu_D = nu.D.data();
    real* du_D = du.D.data();
    real* du_Rx = du.Rx.data();
    real* du_Ry = du.Ry.data();
    real* du_Rz = du.Rz.data();
    const real* b_src = (dir == 0) ? u_Rx : (dir == 1) ? u_Ry : u_Rz;

    graph_ = Kokkos::Experimental::create_graph<execution_space>(
        [&](auto root) {
            // R-space chains (3 independent pairs)
            auto bfx = Bfx.graph_node(root, u_D, du_Rx);
            Brx.graph_node(bfx, u_Rx, du_Rx);

            auto bfy = Bfy.graph_node(root, u_D, du_Ry);
            Bry.graph_node(bfy, u_Ry, du_Ry);

            auto bfz = Bfz.graph_node(root, u_D, du_Rz);
            Brz.graph_node(bfz, u_Rz, du_Rz);

            // D-space chain with Neumann
            auto o = O.graph_node(root, u_D, du_D, op);
            auto b = B.graph_node(o, b_src, du_D);
            N.graph_node(b, nu_D, du_D);
        });

    graph_->instantiate();
}

void derivative::submit_graph()
{
    Kokkos::Profiling::ScopedRegion region("derivative::submit_graph()");
    graph_->submit();
    Kokkos::fence("derivative::submit_graph() complete");
}

template void derivative::operator()<eq_t>(scalar_view, scalar_span, eq_t) const;

template void
derivative::operator()<plus_eq_t>(scalar_view, scalar_span, plus_eq_t) const;

template void
derivative::operator()<eq_t>(scalar_view, scalar_view, scalar_span, eq_t) const;

template void
derivative::operator()<plus_eq_t>(scalar_view, scalar_view, scalar_span, plus_eq_t) const;

template void derivative::build_graph<eq_t>(scalar_view, scalar_span, eq_t);
template void derivative::build_graph<plus_eq_t>(scalar_view, scalar_span, plus_eq_t);
template void derivative::build_graph<eq_t>(scalar_view, scalar_view, scalar_span, eq_t);
template void derivative::build_graph<plus_eq_t>(scalar_view, scalar_view, scalar_span, plus_eq_t);

} // namespace ccs
