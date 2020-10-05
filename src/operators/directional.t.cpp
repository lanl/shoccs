#include "directional.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <random>
#include <vector>

#include <iostream>
#include <range/v3/algorithm/equal.hpp>
#include <range/v3/algorithm/fill.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/generate_n.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/stride.hpp>

#include "fields/fields.hpp"
#include "identity_stencil.hpp"

TEST_CASE("Identity")
{
    using namespace ccs;
    using Catch::Matchers::Approx;

    auto st = stencil{identity_stencil{}};
    const int3 extents{11, 12, 15};
    auto m = mesh{real3{}, real3{1, 1, 1}, extents};

    domain_boundaries db{boundary::dirichlet, boundary::neumann};
    std::vector<boundary> ob{boundary::neumann, boundary::floating};

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis{};
    std::vector<real> rng =
        vs::generate_n([&gen, &dis]() { return dis(gen); }, m.size()) |
        rs::to<std::vector<real>>();

    std::vector<real> rhs(rng.size());

    auto neumann_coords = [&ob](auto&& f) {
        return vs::filter([&ob](auto&& info) {
                   return ob[info.shape_id] == boundary::neumann;
               }) |
               vs::transform([&f](auto&& info) { return f(info.solid_coord); });
    };

    SECTION("x-domain")
    {

        auto d = op::directional{0, st, m, geometry{}, db, ob};
        scalar_field<real, 0> f{rng, extents};
        scalar_field<real, 0> df(rng, extents);

        d(f, df, result_view(rhs));

        REQUIRE_THAT(rhs, Approx(rng));
    }

    SECTION("x-planes")
    {
        rs::fill(rhs, 0.0);

        real left = 1e-6;
        real right = 1.0 - 1e-6;
        auto shapes = std::vector<shape>{
            make_yz_rect(0, real3{left, -1, -1}, real3{left, 2, 2}, 1),
            make_yz_rect(1, real3{right, -1, -1}, real3{right, 2, 2}, -1)};
        auto g = geometry(shapes, m);
        auto d = op::directional{0, st, m, g, db, ob};

        // field boundary values all have the intuitive ordering here
        scalar_field<real, 0> f{rng, extents};
        // values for df boundaries come from just bpts neumann shapes in g.Rx()
        // indicies for df boundaries come from all solid points
        scalar_field<real, 0> df{extents};

        df >> select(g.Sx()) <<= g.Rx() | neumann_coords(f);

        d(f, df, result_view(rhs));

        REQUIRE_THAT(rhs, Approx(rng));
    }

    SECTION("y-domain")
    {

        auto d = op::directional{1, st, m, geometry{}, db, ob};

        scalar_field<real, 1> f{rng, extents};
        scalar_field<real, 1> df{rng, extents};

        d(f, df, result_view(rhs));

        REQUIRE_THAT(rhs, Approx(rng));
    }

    SECTION("y-planes")
    {
        rs::fill(rhs, 0.0);
        real left = 1e-6;
        real right = 1.0 - 1e-6;
        auto shapes = std::vector<shape>{
            make_xz_rect(0, real3{-1, left, -1}, real3{2, left, 2}, 1),
            make_xz_rect(1, real3{-1, right, -1}, real3{2, right, 2}, -1)};
        auto g = geometry(shapes, m);
        auto d = op::directional{1, st, m, g, db, ob};

        y_field f{rng, extents};
        y_field df{extents};

        df >> select(g.Sy()) <<= g.Ry() | neumann_coords(f);

        d(f, df, result_view(rhs));
        REQUIRE_THAT(rhs, Approx(rng));
    }

    SECTION("z-domain")
    {
        auto d = op::directional{2, st, m, geometry{}, db, ob};

        z_field f{rng, extents};
        z_field df{rng, extents};

        d(f, df, result_view(rhs));

        REQUIRE_THAT(rhs, Approx(rng));
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

        z_field f(rng, extents);
        z_field df(extents);

        df >> select(g.Sz()) <<= g.Rz() | neumann_coords(f);

        d(f, df, result_view(rhs));
        REQUIRE_THAT(rhs, Approx(rng));
    }
}
