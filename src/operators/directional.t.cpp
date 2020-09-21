#include "directional.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>

#include <iostream>
#include <range/v3/algorithm/equal.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/generate_n.hpp>
#include <range/v3/view/stride.hpp>
#include <range/v3/view/iota.hpp>

#include "fields/fields.hpp"

namespace ccs
{
struct identity_stencil {
    stencil_info query(boundary b) const
    {
        if (b == boundary::neumann)
            return {0, 2, 3, 2};
        else
            return {0, 2, 3, 0};
    }
    stencil_info query_max() const { return {0, 2, 3, 2}; }

    void interior(real, std::span<real> c) const { c[0] = 1; }

    void nbs(real,
             boundary b,
             real,
             bool right_wall,
             std::span<real> c,
             std::span<real> x) const
    {
        if (b == boundary::neumann) {
            if (right_wall) {
                x[0] = 0;
                x[1] = 2;

                c[0] = 0;
                c[1] = 1;
                c[2] = 0;
                c[3] = 0;
                c[4] = 0;
                c[5] = -1;
            } else {
                x[0] = 2;
                x[1] = 0;

                c[0] = -1;
                c[1] = 0;
                c[2] = 0;
                c[3] = 0;
                c[4] = 1;
                c[5] = 0;
            }
        } else if (right_wall) {
            c[0] = 0;
            c[1] = 1;
            c[2] = 0;
            c[3] = 0;
            c[4] = 0;
            c[5] = 1;
        } else {
            c[0] = 1;
            c[1] = 0;
            c[2] = 0;
            c[3] = 0;
            c[4] = 1;
            c[5] = 0;
        }
    }
};
} // namespace ccs

TEST_CASE("Identity")
{
    using namespace ccs;
    using namespace ranges::views;

    auto st = stencil{identity_stencil{}};
    const int3 extents {11, 4, 4};
    auto m = mesh{real3{}, real3{1, 1, 1}, extents};

    domain_boundaries db{boundary::dirichlet, boundary::neumann};
    std::vector<boundary> ob{boundary::neumann, boundary::floating};

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis{};
    std::vector<real> rng = generate_n([&gen, &dis]() { return dis(gen); }, m.size()) |
                            ranges::to<std::vector<real>>();

    std::vector<real> rhs(rng.size());
    SECTION("x-domain")
    {

        auto d = op::directional{0, st, m, geometry{}, db, ob};
        scalar_field<real, 0> f{rng, extents};
        scalar_field<real, 0> df(rng.size(), extents);
        result_view result(rhs);
        // set neumann values
        for (int i = m.n(0)-1; i < m.size(); i += m.n(0)) df[i] = rng[i];
        d(f, df, result);

        REQUIRE(ranges::equal(result, rng));
    }
#if 0
    SECTION("x-planes")
    {
        real left = 1e-6;
        real right = 1.0 - 1e-6;
        auto shapes = std::vector<shape>{
            make_yz_rect(0, real3{left, -1, -1}, real3{left, 2, 2}, 1),
            make_yz_rect(1, real3{right, -1, -1}, real3{right, 2, 2}, -1)};
        auto g = geometry(shapes, m);
        auto d = op::directional{0, st, m, g, db, ob};
        auto t = transform([f = m.ucf_ijk2dir(0), &rng](auto&& info) {
            return rng[f(info.solid_coord)];
        });
        // set boundary values
        std::vector<real> bv = g.Rx() | t | ranges::to<std::vector<real>>();
        std::vector<real> nv = g.Rx() | filter([&ob](auto&& info) {
                                   return ob[info.shape_id] == boundary::neumann;
                               }) |
                               t | ranges::to<std::vector<real>>();
        std::cout << "neumann bvals\n";
        for (auto&& v : nv) std::cout << '\t' << v << '\n';
        std::cout << "dirichlet bvals\n";
        for (auto&& v : bv) std::cout << '\t' << v << '\n';
        // REQUIRE(static_cast<int>(nv.size()) == m.plane_size(0));
        // REQUIRE(nv[0] == rng[m.uc_ijk2dir(0, g.Rx(1)[0].solid_coord)]);
        std::vector<real> neumann_rng(rng.size());
        auto res = d(rng, bv, neumann_rng, nv);
        for (int i = 0; auto&& [cmp, ex] : zip(res, rng) | take(2 * m.n(0))) {
            std::cout << "[" << i << "]\t" << cmp << " / " << ex << '\n';
            ++i;
        }
        for (auto&& [cmp, ex] : zip(res, rng)) REQUIRE(cmp == ex);
    }

    SECTION("y-domain")
    {

        auto d = op::directional{1, st, m, geometry{}, db, ob};
        std::vector<real> bv{};
        std::vector<real> nv{};
        std::vector<real> neumann_rng(rng.size());
        // set neumann values
        for (int i = 0; i < m.size(); i += m.n(1)) neumann_rng[i] = rng[i];
        auto res = d(rng, bv, neumann_rng, nv);
        REQUIRE(ranges::equal(res, rng));
    }

    SECTION("y-planes")
    {
        real left = 1e-6;
        real right = 1.0 - 1e-6;
        auto shapes = std::vector<shape>{
            make_xz_rect(0, real3{-1, left, -1}, real3{2, left, 2}, 1),
            make_xz_rect(1, real3{-1, right, -1}, real3{2, right, 2}, -1)};
        auto g = geometry(shapes, m);
        auto d = op::directional{1, st, m, g, db, ob};
        // grab boundary values from planes
        std::vector<real> bv =
            g.Ry() | transform([](auto&& info) { return info.solid_coord; }) |
            transform(m.ucf_ijk2dir(1)) | transform([&rng](auto&& i) { return rng[i]; }) |
            ranges::to<std::vector<real>>();
        auto res = d(rng, bv);
        REQUIRE(ranges::equal(res, rng));
    }


    SECTION("z-domain")
    {
        auto d = op::directional{2, st, m, geometry{}, db, ob};
        std::vector<real> bv{};
        std::vector<real> nv{};
        std::vector<real> neumann_rng(rng.size());
        // set neumann values
        for (int i = 0; i < m.size(); i += m.n(2)) neumann_rng[i] = rng[i];
        auto res = d(rng, bv, neumann_rng, nv);

        REQUIRE(ranges::equal(res, rng));
    }

    SECTION("z-planes")
    {
        real left = 1e-6;
        real right = 1.0 - 1e-6;
        auto shapes = std::vector<shape>{
            make_xy_rect(0, real3{-1, -1, left}, real3{2, 2, left}, 1),
            make_xy_rect(1, real3{-1, -1, right}, real3{2, 2, right}, -1)};
        auto g = geometry(shapes, m);
        auto d = op::directional{2, st, m, g, db, ob};
        std::vector<real> bv =
            g.Rz() | transform([](auto&& info) { return info.solid_coord; }) |
            transform(m.ucf_ijk2dir(2)) | transform([&rng](auto&& i) { return rng[i]; }) |
            ranges::to<std::vector<real>>();
        auto res = d(rng, bv);
        REQUIRE(ranges::equal(res, rng));
    }
#endif
}

#if 0
TEST_CASE("E2_2")
{
    auto st = make_E2_2();

    auto m = mesh{real3{}, real3{1, 1, 1}, int3{31, 1, 1}};
    auto shapes =
        std::vector<shape>{make_yz_rect(0, real3{-1, -1, left}, real3{2, 2, left}, 1),
                           make_yz_rect(1, real3{-1, -1, right}, real3{2, 2, right}, -1)};
    REQUIRE(m.dims() == 1);

    domain_boundaries db{boundary::neumann, boundary::dirichlet};
    std::vector<boundary> ob{boundary::floating, boundary::neumann};

    std::vector<real> rhs{};
}
#endif