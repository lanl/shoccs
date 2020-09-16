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
#include <range/v3/view/take_exactly.hpp>

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
};

namespace detail
{
// Encapsulate the logic for building the boundary value and boundary derivative
// CSR matrices.  They each need their own copy of the solid points iterator because
// boundary values are associated with every point in R but derivative values are only
// needed at a few places.  Keeping separate iterators allows the user to simply pass in
// lists of values where the order dictates everything. The boundary derivative CSR matrix
// also contains information on the domain boundaries whereas the boundary value CSR
// matrix does not
using It = typename std::vector<int>::const_iterator;
class builder_
{
    matrix::csr_builder b;
    It spts;

    template <ranges::random_access_range R>
    void add_(R&& rng, int bpt, int r, int t, int uc, int inc)
    {
        using namespace ranges::views;

        // add boundary column to csr
        for (auto&& [i, v] : rng | stride(t) | enumerate | drop(1))
            b.add_point(uc + inc * i, bpt, v);

        // add boundary point to csr
        b.add_point(bpt, bpt, rng[0]);

        // add boundary row to csr
        for (auto&& [i, v] : rng | enumerate | drop(1) | take(t - 1))
            b.add_point(bpt, uc + inc * i, v);
    }

protected:
    builder_() = default;
    builder_(int n, It spts) : b{n}, spts{spts} {}

    void add(std::span<real> c, int r, int t, int uc, int inc, bool solid)
    {
        using namespace ranges::views;
        auto bpt = solid ? *spts++ : uc;
        if (inc > 0)
            add_(c | take_exactly(r * t), bpt, r, t, uc, inc);
        else
            add_(c | take_exactly(r * t) | reverse, bpt, r, t, uc, inc);
    }

    auto to_csr_(int nrows) { return b.to_csr(nrows); }
};
} // namespace detail
class value_builder : detail::builder_
{
public:
    value_builder() = default;
    value_builder(int n, detail::It spts) : detail::builder_{n, spts} {}

    void add_left(std::span<real> c, int r, int t, int uc)
    {
        std::cout << "\nvb left\n";
        add(c, r, t, uc, 1, true);
    }

    void add_right(std::span<real> c, int r, int t, int uc)
    {
        std::cout << "\nvb right\n";
        add(c, r, t, uc, -1, true);
    }

    auto inner_left(std::span<real> c, int r, int t)
    {
        using namespace ranges::views;
        return c | chunk(t) | drop(1) | for_each(drop(1));
    }

    auto inner_right(std::span<real> c, int r, int t)
    {
        using namespace ranges::views;
        return c | chunk(t) | for_each(take(t - 1));
    }

    auto to_csr(int nrows) { return to_csr_(nrows); }
};

class derivative_builder : detail::builder_
{
public:
    derivative_builder() = default;
    derivative_builder(int n, detail::It spts) : detail::builder_{n, spts} {}

    void add_left_domain(std::span<real> c, int r, int uc)
    {
        std::cout << "\ndb left domain\n";
        if (r > 0) add(c, r, 1, uc, 1, false);
    }

    void add_left_solid(std::span<real> c, int r, int uc)
    {
        std::cout << "\ndb left solid\n";
        if (r > 0) add(c, r, 1, uc, 1, true);
    }

    void add_right_domain(std::span<real> c, int r, int uc)
    {
        std::cout << "\ndb right domain\n";
        if (r > 0) add(c, r, 1, uc, -1, false);
    }

    void add_right_solid(std::span<real> c, int r, int uc)
    {
        std::cout << "\ndb right solid\n";
        if (r > 0) add(c, r, 1, uc, -1, true);
    }

    auto to_csr(int nrows) { return to_csr_(nrows); }
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
                                              ob[next_obj.shape_id],
                                              false}};
                        }
                    }
                    ++first;
                }
        }
}

static wall_info left_wall(const stencil& st,
                           real h,
                           const boundary_info& bi,
                           std::span<real> coeffs,
                           std::span<real> extra,
                           value_builder& vb,
                           derivative_builder& db)
{

    auto q = st.query(bi.b);
    st.nbs(h, bi.b, bi.psi, false, coeffs, extra);

    if (bi.is_domain_b) {
        db.add_left_domain(extra, q.nextra, bi.global_i);
        return {matrix::dense{q.r, q.t, coeffs}, bi.global_i};
    }
    if (bi.b == boundary::neumann)
        db.add_left_solid(extra, q.nextra, bi.global_i);
    else
        vb.add_left(coeffs, q.r, q.t, bi.global_i);

    auto inner = vb.inner_left(coeffs, q.r, q.t);

    return {matrix::dense{q.r - 1, q.t - 1, inner}, bi.global_i + 1};
}

static wall_info right_wall(const stencil& st,
                            real h,
                            const boundary_info& bi,
                            std::span<real> coeffs,
                            std::span<real> extra,
                            value_builder& vb,
                            derivative_builder& db)
{

    auto q = st.query(bi.b);
    st.nbs(h, bi.b, bi.psi, true, coeffs, extra);

    if (bi.is_domain_b) {
        db.add_right_domain(extra, q.nextra, bi.global_i);
        return {matrix::dense{q.r, q.t, coeffs}, bi.global_i};
    }

    if (bi.b == boundary::neumann)
        db.add_right_solid(extra, q.nextra, bi.global_i);
    else
        vb.add_right(coeffs, q.r, q.t, bi.global_i);

    auto inner = vb.inner_right(coeffs, q.r, q.t);

    return {matrix::dense{q.r - 1, q.t - 1, inner}, bi.global_i - 1};
}

directional::directional(int dir,
                         const stencil& st,
                         const mesh& m,
                         const geometry& g,
                         domain_boundaries db,
                         std::span<const boundary> object_b)
{
    // keep transformed solid points for mapping values in application function
    const auto b_sz = g.R(dir).size();
    spts = g.S(dir) | ranges::view::transform(m.ucf_ijk2dir(dir)) |
           ranges::view::take(b_sz) | ranges::to<std::vector<int>>();

    const auto ps = m.plane_size(dir);
    const auto h = m.line(dir).h;

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
    auto B_bld = value_builder(ps, spts.begin());
    auto N_bld = derivative_builder(ps, spts.begin());
    auto O_bld = matrix::block::builder(ps + b_sz);

    for (auto&& [lb, rb] : lines(dir, m, g, db, object_b)) {

        auto [left_mat, lpt] = left_wall(st, h, lb, left, extra, B_bld, N_bld);

        auto [right_mat, rpt] = right_wall(st, h, rb, right, extra, B_bld, N_bld);

        int n = rpt - lpt + 1;
        int nl = left_mat.rows();
        int nr = right_mat.rows();

        O_bld.add_inner_block(lpt,
                              std::move(left_mat),
                              matrix::circulant{nl - p, n - nr - nl, interior_c},
                              std::move(right_mat));
    }
    // finish off csr
    B = B_bld.to_csr(m.size());
    N = N_bld.to_csr(m.size());
    O = std::move(O_bld).to_block(m.size());
}
} // namespace ccs::op