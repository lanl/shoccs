#include "directional.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>

#include <iostream>
#include <range/v3/algorithm/equal.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/generate_n.hpp>

namespace ccs
{
struct identity_stencil {
    stencil_info query(boundary) const { return {0, 2, 3, 0}; }
    stencil_info query_max() const { return {0, 2, 3, 0}; }

    void interior(real, std::span<real> c) const { c[0] = 1; }

    void
    nbs(real, boundary, real, bool right_wall, std::span<real> c, std::span<real>) const
    {
        if (right_wall) {
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

TEST_CASE("directional")
{
    using namespace ccs;

    auto st = stencil{identity_stencil{}};
    auto m = mesh{real3{}, real3{1, 1, 1}, int3{11, 22, 13}};
    REQUIRE(m.dims() == 3);

    domain_boundaries db{};
    std::vector<boundary> ob{boundary::dirichlet, boundary::floating};

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis{};
    std::vector<real> rng =
        ranges::views::generate_n([&gen, &dis]() { return dis(gen); }, m.size()) |
        ranges::to<std::vector<real>>();

    SECTION("x-domain")
    {

        auto d = op::directional{0, st, m, geometry{}, db, ob};
        auto res = d(rng);
        REQUIRE(ranges::equal(res, rng));
    }

    SECTION("x-planes")
    {
        real left = 1e-6;
        real right = 1.0 - 1e-6;
        auto shapes = std::vector<shape>{
            make_yz_rect(0, real3{left, -1, -1}, real3{left, 2, 2}, 1),
            make_yz_rect(1, real3{right, -1, -1}, real3{right, 2, 2}, -1)};
        auto g = geometry(shapes, m);
        auto d = op::directional{0, st, m, g, db, ob};
        auto res = d(rng);
        REQUIRE(ranges::equal(res, rng));
    }

    SECTION("y-domain")
    {

        auto d = op::directional{1, st, m, geometry{}, db, ob};
        auto res = d(rng);

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
        auto res = d(rng);
        REQUIRE(ranges::equal(res, rng));
    }

    SECTION("z-domain")
    {
        auto d = op::directional{2, st, m, geometry{}, db, ob};
        auto res = d(rng);

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
        auto res = d(rng);
        REQUIRE(ranges::equal(res, rng));
    }
}