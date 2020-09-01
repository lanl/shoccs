#include "directional.hpp"
#include "boundaries.hpp"
#include "indexing.hpp"
#include <cassert>
#include <iostream>

#include <cppcoro/generator.hpp>

#include <range/v3/view/drop.hpp>
#include <range/v3/view/reverse.hpp>
#include <range/v3/view/stride.hpp>
#include <range/v3/view/take.hpp>

namespace ccs::op
{

struct boundary_info {
    int global_i;
    real psi;
    boundary b;
    bool is_domain_b;
};

struct wall_info {
    matrix::dense mat; // boundary matrix to place in O
    int pt;            // first or last point in O for stencil
    int s;             // number of solid points consumed
};

// We need to return lines of boundary_info so the caller
// can build up the appropriate operators
// the lines will be of a few different types:
// [domain, domain]
// [domain, object]
// [object, domain]
// [onbject, object]
//
// As with the solid point identification algorithm, we do not
// properly handle the case of fully solid lines
cppcoro::generator<std::array<boundary_info, 2>> static lines(
    int dir,
    const mesh& m,
    const geometry& g,
    domain_boundaries db,
    std::span<const boundary> ob)
{
    // the fast/slow idiom was used to determine the order
    // of the solid points so it must be used to access them
    auto [ns, nf, n] = m.n_dir(dir);
    auto [fast, slow] = index::dirs(dir);

    auto object_info = g.R(dir);
    auto first = object_info.begin();
    auto last = object_info.end();

    // coordinate transformation functions
    auto ijk2uc = m.ucf_ijk2dir(dir);
    auto uc = m.ucf_dir(dir);

    for (int s = 0; s < ns; s++)
        for (int f = 0; f < nf; f++) {
            auto left_ic = uc(int3{s, f, 0});
            auto right_ic = uc(int3{s, f, n - 1});

            if (first == last) {
                co_yield std::array{boundary_info{left_ic, 1.0, db.left, true},
                                    boundary_info{right_ic, 1.0, db.right, true}};
            } else
                while (first != last && first->solid_coord[slow] == s &&
                       first->solid_coord[fast] == f) {
                    const auto& obj = *first;

                    if (obj.ray_outside) {
                        // assume the left boundary is the domain wall in this case
                        co_yield std::array{boundary_info{left_ic, 1.0, db.left, true},
                                            boundary_info{ijk2uc(obj.solid_coord),
                                                          obj.psi,
                                                          ob[obj.shape_id],
                                                          false}};
                    } else {
                        auto lbi = boundary_info{
                            ijk2uc(obj.solid_coord), obj.psi, ob[obj.shape_id], false};
                        auto next = first + 1;
                        if (next == last || next->solid_coord[slow] != s ||
                            next->solid_coord[fast] != f)
                            co_yield std::array{
                                lbi, boundary_info{right_ic, 1.0, db.right, true}};
                        else {
                            const auto& next_obj = *++first;
                            assert(next_obj.ray_outside);
                            co_yield std::array{
                                lbi,
                                boundary_info{ijk2uc(next_obj.solid_coord),
                                              next_obj.psi,
                                              ob[obj.shape_id],
                                              false}};
                        }
                    }
                    ++first;
                }
        }
}

template <typename R>
static wall_info left_wall(const stencil& st,
                           real h,
                           const boundary_info& bi,
                           R&& solid,
                           bool no_solid,
                           std::span<real> coeffs,
                           std::span<real> extra,
                           matrix::csr_builder& bld)
{

    auto q = st.query(bi.b);
    st.nbs(h, bi.b, bi.psi, false, coeffs, extra);

    if (bi.is_domain_b) return {matrix::dense{q.r, q.t, coeffs}, bi.global_i, 0};

    // should really do something better here
    if (no_solid) throw std::runtime_error("Ran out of solid points");

    // Add boundary column (first) to csr
    auto bpt = *solid;
    using namespace ranges::views;
    for (auto&& [i, v] : coeffs | stride(q.t) | enumerate | drop(1) | take(q.r - 1))
        bld.add_point(bi.global_i + i, bpt, v);

    // Add corner of boundary matrix to csr
    bld.add_point(bpt, bpt, coeffs[0]);
    // Add rest of boundary row to csr
    for (auto&& [i, v] : coeffs | enumerate | drop(1) | take(q.t - 1))
        bld.add_point(bpt, bi.global_i + i, v);

    // range representing the non-boundary portion of the boundary matrix
    auto rng = coeffs | chunk(q.t) | drop(1) | for_each(drop(1));
    return {matrix::dense{q.r - 1, q.t - 1, rng}, bi.global_i + 1, 1};
}

template <typename R>
static wall_info right_wall(const stencil& st,
                            real h,
                            const boundary_info& bi,
                            R&& solid,
                            bool no_solid,
                            std::span<real> coeffs,
                            std::span<real> extra,
                            matrix::csr_builder& bld)
{

    auto q = st.query(bi.b);
    st.nbs(h, bi.b, bi.psi, true, coeffs, extra);

    if (bi.is_domain_b) return {matrix::dense{q.r, q.t, coeffs}, bi.global_i, 0};

    // should really do something better here
    if (no_solid) throw std::runtime_error("Ran out of solid points");

    auto bpt = *solid;
    auto sz = q.r * q.t;

    using namespace ranges::views;
    // reverse the range for easier processing
    auto rev = coeffs | take(sz) | reverse;

    // Add boundary column (last) to csr
    for (auto&& [i, v] : rev | stride(q.t) | enumerate | drop(1))
        bld.add_point(bi.global_i - i, bpt, v);

    // Add corner of boundary matrix to csr
    bld.add_point(bpt, bpt, coeffs[sz - 1]);
    // Add rest of boundary row (last) to csr
    for (auto&& [i, v] : rev | enumerate | drop(1) | take(q.t - 1))
        bld.add_point(bpt, bi.global_i - i, v);

    // range representing the non-boundary portion of the boundary matrix
    auto rng = coeffs | chunk(q.t) | take(q.r - 1) | for_each(take(q.t - 1));
    return {matrix::dense{q.r - 1, q.t - 1, rng}, bi.global_i - 1, 1};
}

directional::directional(int dir,
                         const stencil& st,
                         const mesh& m,
                         const geometry& g,
                         domain_boundaries db,
                         std::span<const boundary> object_b)
{
    const auto ps = m.plane_size(dir);
    const auto h = m.line(dir).h;
    // educated guess for memory usage
    {
        const auto boundary_pts = g.R(dir).size();
        offsets.reserve(ps + boundary_pts);
        O.reserve(ps + boundary_pts);
        zeros.reserve(ps + 1 + boundary_pts);
    }

    auto [p, rmax, tmax, ex_max] = st.query_max();
    // set up the interior stencil
    interior_c.resize(2 * p + 1);
    st.interior(h, interior_c);
    // allocate maximum amount of memory required by any boundary conditions
    std::vector<real> left(rmax * tmax);
    std::vector<real> right(rmax * tmax);
    std::vector<real> extra(ex_max);

    // We will be adding points to the csr via the builder so they can be inserted
    // in any order
    auto bld = matrix::csr::builder(ps);
    auto solid_pts = g.S(dir) | ranges::view::transform(m.ucf_ijk2dir(dir));
    auto solid = ranges::begin(solid_pts);
    auto solid_last = ranges::end(solid_pts);

    // initialize with -1 to ensure calculations work out for first lb point
    int b_prev = -1;

    for (auto&& [lb, rb] : lines(dir, m, g, db, object_b)) {

        auto [left_mat, lpt, li] =
            left_wall(st, h, lb, solid, solid == solid_last, left, extra, bld);
        solid += li;

        auto [right_mat, rpt, ri] =
            right_wall(st, h, rb, solid, solid == solid_last, right, extra, bld);
        solid += ri;

        int n = rpt - lpt + 1;
        int nl = left_mat.rows();
        int nr = right_mat.rows();

        // std::cout << "\n[lpt,rpt]\t[" << lpt << ", " << rpt << "]\n";
        // std::cout << "[nl,n,nr]\t[" << nl << ", " << n << ", " << nr << "]\n";

        O.emplace_back(std::move(left_mat),
                       matrix::circulant{nl - p, n - nr - nl, interior_c},
                       std::move(right_mat));
        offsets.push_back(lpt);

        zeros.push_back(lpt - 1 - b_prev);

        b_prev = rpt;
    }
    // handle trailing zeros
    zeros.push_back(std::max(0, m.size() - 1 - b_prev));

    // finish off csr
    B = bld.to_csr(m.size());
}
} // namespace ccs::op