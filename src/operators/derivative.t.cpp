#include "derivative.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "fields/selector.hpp"
#include "identity_stencil.hpp"
#include "random/random.hpp"
#include "stencils/stencil.hpp"

#include <range/v3/all.hpp>

using namespace ccs;
using Catch::Matchers::Approx;

constexpr auto g = []() { return pick(); };

// 2nd order polynomial for use with E2
constexpr auto f2 = [](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x * x * (y + z) + y * y * (x + z) + z * z * (x + y) + 3 * x * y * z + x + y +
           z;
};

constexpr auto f2_dx = [](auto&& loc) {
    auto&& [x, y, z] = loc;
    return 2. * x * (y + z) + y * y + z * z + 3. * y * z + 1;
};

constexpr auto f2_dy = [](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x * x + 2. * y * (x + z) + z * z + 3. * x * z + 1;
};

constexpr auto f2_dz = [](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x * x + y * y + 2. * z * (x + y) + 3. * x * y + 1;
};

constexpr auto f2_ddx = [](auto&& loc) {
    auto&& [_, y, z] = loc;
    return 2. * (y + z);
};

constexpr auto f2_ddy = [](auto&& loc) {
    auto&& [x, _, z] = loc;
    return 2. * (x + z);
};

constexpr auto f2_ddz = [](auto&& loc) {
    auto&& [x, y, _] = loc;
    return 2. * (x + y);
};

TEST_CASE("E2_Neumann")
{
    using T = std::vector<real>;

    const auto extents = int3{10, 13, 17};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};

    // initialize fields
    scalar<T> u{};
    u = m.location | vs::transform(f2);
    scalar<T> nu{};
    nu = m.location | vs::transform(f2_dz);
    scalar<T> ex{};
    ex = m.location | vs::transform(f2_ddz);

    ex | m.xmin = 0;
    ex | m.xmax = 0;

    scalar<T> du{m.ss()};
    {
        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::nn};
        auto d = derivative(2, m, stencils::second::E2, gridBcs, objectBcs);
        d(u, nu, du);
        REQUIRE_THAT(get<si::D>(du), Approx(get<si::D>(ex)));

        d(u, nu, du, plus_eq);
        ex *= 2;
        REQUIRE_THAT(get<si::D>(du), Approx(get<si::D>(ex)));
    }
}

TEST_CASE("Identity FFFFFF")
{
    using T = std::vector<real>;

    const auto extents = int3{5, 7, 6};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {-1, -1, 0}, .max = {1, 2, 2.2}}};

    const auto gridBcs = bcs::Grid{bcs::ff, bcs::ff, bcs::ff};
    const auto objectBcs = bcs::Object{};

    // initialize fields
    randomize();
    scalar<T> u{m.ss()};
    u | sel::D = vs::generate_n(g, m.size());

    scalar<T> du{m.ss()};
    REQUIRE((integer)rs::size(du | sel::D) == m.size());

    for (int i = 0; i < 3; i++) {
        auto d = derivative{i, m, stencils::identity, gridBcs, objectBcs};
        d(u, du);

        REQUIRE_THAT(get<si::D>(u), Approx(get<si::D>(du)));
    }
}

TEST_CASE("E2_2 FFFFFF")
{
    using T = std::vector<real>;

    const auto extents = int3{5, 7, 6};

    // shift domain bounds away from zero to avoid problems with Catch::Approx
    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto gridBcs = bcs::Grid{bcs::ff, bcs::ff, bcs::ff};
    const auto objectBcs = bcs::Object{};

    // initialize fields
    scalar<T> u = m.location | vs::transform(f2);
    //    u | sel::D = m.location() | vs::transform(f2);

    scalar<T> du{m.ss()};
    REQUIRE((integer)rs::size(du | sel::D) == m.size());

    // exact
    std::array<scalar<T>, 3> dd{m.location | vs::transform(f2_ddx),
                                m.location | vs::transform(f2_ddy),
                                m.location | vs::transform(f2_ddz)};

    for (int i = 0; i < 3; i++) {
        auto d = derivative{i, m, stencils::second::E2, gridBcs, objectBcs};
        d(u, du);

        REQUIRE_THAT(get<si::D>(dd[i]), Approx(get<si::D>(du)));
    }
}

TEST_CASE("Identity Mixed")
{
    using T = std::vector<real>;

    const auto extents = int3{5, 7, 6};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {-1, -1, 0}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};

    // initialize fields
    randomize();
    scalar<T> u{m.ss()};
    u | sel::D = vs::generate_n(g, m.size());

    SECTION("DDFNFD")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::fn, bcs::fd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        scalar<T> du_exact{u}, nu{u};

        // set zeros for dirichlet at xmin/xmax
        du_exact | m.xmin = 0;
        du_exact | m.xmax = 0;
        du_exact | m.zmax = 0;

        scalar<T> du{u};
        REQUIRE((integer)rs::size(du | sel::D) == m.size());

        for (int i = 0; i < m.dims(); i++) {
            auto d = derivative{i, m, stencils::identity, gridBcs, objectBcs};
            du = 0;
            d(u, nu, du);

            REQUIRE_THAT(get<si::D>(du_exact), Approx(get<si::D>(du)));
        }
    }

    SECTION("NNDDDF")
    {

        const auto gridBcs = bcs::Grid{bcs::nn, bcs::dd, bcs::df};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        scalar<T> du_exact{u}, nu{u};

        // set zeros for dirichlet at xmin/xmax
        du_exact | m.ymin = 0;
        du_exact | m.ymax = 0;
        du_exact | m.zmin = 0;

        scalar<T> du{u};
        REQUIRE((integer)rs::size(du | sel::D) == m.size());

        for (int i = 0; i < m.dims(); i++) {
            auto d = derivative{i, m, stencils::identity, gridBcs, objectBcs};
            du = 0;
            d(u, nu, du);

            REQUIRE_THAT(get<si::D>(du_exact), Approx(get<si::D>(du)));
        }
    }
}

TEST_CASE("E2 Mixed")
{
    using T = std::vector<real>;

    const auto extents = int3{5, 7, 6};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};

    // initialize fields
    scalar<T> u{m.location | vs::transform(f2)};

    SECTION("DDFFFD")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::fd};
        // set the exact du we expect based on zeros assigned to dirichlet locations

        std::array<scalar<T>, 3> dd{m.location | vs::transform(f2_ddx),
                                    m.location | vs::transform(f2_ddy),
                                    m.location | vs::transform(f2_ddz)};

        scalar<T> du{m.ss()};
        REQUIRE((integer)rs::size(du | sel::D) == m.size());

        for (int i = 0; i < m.dims(); i++) {
            auto d = derivative{i, m, stencils::second::E2, gridBcs, objectBcs};
            du = 0;
            d(u, du);

            auto& ex = dd[i];
            // zero boundaries
            ex | m.xmin = 0;
            ex | m.xmax = 0;
            ex | m.zmax = 0;

            REQUIRE_THAT(get<si::D>(ex), Approx(get<si::D>(du)));
        }
    }

    SECTION("FNDDDF")
    {

        const auto gridBcs = bcs::Grid{bcs::fn, bcs::dd, bcs::df};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        scalar<T> nu{m.location | vs::transform(f2_dx)};
        std::array<scalar<T>, 3> dd{m.location | vs::transform(f2_ddx),
                                    m.location | vs::transform(f2_ddy),
                                    m.location | vs::transform(f2_ddz)};

        scalar<T> du{u};
        REQUIRE((integer)rs::size(du | sel::D) == m.size());

        for (int i = 0; i < m.dims(); i++) {
            auto d = derivative{i, m, stencils::second::E2, gridBcs, objectBcs};
            du = 0;
            d(u, nu, du);

            auto& ex = dd[i];
            // zero boundaries
            ex | m.ymin = 0;
            ex | m.ymax = 0;
            ex | m.zmin = 0;

            REQUIRE_THAT(get<si::D>(ex), Approx(get<si::D>(du)));
        }
    }
}

TEST_CASE("Identity with Objects")
{
    using T = std::vector<real>;

    const auto extents = int3{16, 19, 18};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {-1, -1, 0}, .max = {1, 2, 2.2}},
                  std::vector<shape>{make_sphere(0, real3{0.01, -0.01, 0.99}, 0.25)}};

    const auto gridBcs = bcs::Grid{bcs::ff, bcs::ff, bcs::ff};
    const auto objectBcs = bcs::Object{bcs::Dirichlet};

    // initialize fields
    randomize();
    scalar<T> u{m.location | vs::transform([](auto&&) { return pick(); })};

    REQUIRE(rs::size(u | sel::Rx) == m.Rx().size());

    scalar<T> du_x{u};
    scalar<T> du_y{u};
    scalar<T> du_z{u};
    REQUIRE((integer)rs::size(du_x | sel::D) == m.size());

    auto dx = derivative{0, m, stencils::identity, gridBcs, objectBcs};
    auto dy = derivative{1, m, stencils::identity, gridBcs, objectBcs};
    auto dz = derivative{2, m, stencils::identity, gridBcs, objectBcs};

    du_x = 0;
    dx(u, du_x);
    du_y = 0;
    dy(u, du_y);
    du_z = 0;
    dz(u, du_z);

    REQUIRE_THAT(get<si::D>(du_x), Approx(get<si::D>(du_y)));
    REQUIRE_THAT(get<si::D>(du_x), Approx(get<si::D>(du_z)));
}

TEST_CASE("E2 with Objects")
{
    using T = std::vector<real>;

    const auto extents = int3{25, 26, 27};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}},
                  std::vector<shape>{make_sphere(0, real3{0.45, 1.011, 1.31}, 0.25)}};

    const auto gridBcs = bcs::Grid{bcs::nn, bcs::dd, bcs::ff};
    const auto objectBcs = bcs::Object{bcs::Dirichlet};

    // initialize fields
    scalar<T> u = m.location | vs::transform(f2);
    REQUIRE(rs::size(u | sel::Rx) == m.Rx().size());

    scalar<T> nu = m.location | vs::transform(f2_dx);

    scalar<T> du_x{m.ss()}, du_y{m.ss()}, du_z{m.ss()};

    du_x | m.fluid = m.location | vs::transform(f2_ddx);
    du_x | m.ymin = 0;
    du_x | m.ymax = 0;

    du_y | m.fluid = m.location | vs::transform(f2_ddy);
    du_y | m.ymin = 0;
    du_y | m.ymax = 0;

    du_z | m.fluid = m.location | vs::transform(f2_ddz);
    du_z | m.ymin = 0;
    du_z | m.ymax = 0;

    REQUIRE((integer)rs::size(du_x | sel::D) == m.size());

    auto dx = derivative{0, m, stencils::second::E2, gridBcs, objectBcs};
    auto dy = derivative{1, m, stencils::second::E2, gridBcs, objectBcs};
    auto dz = derivative{2, m, stencils::second::E2, gridBcs, objectBcs};

    scalar<T> du{u};

    du = 0;
    dx(u, nu, du);
    REQUIRE_THAT(get<si::D>(du), Approx(get<si::D>(du_x)));

    du = 0;
    dy(u, du);
    REQUIRE_THAT(get<si::D>(du), Approx(get<si::D>(du_y)));

    du = 0;
    dz(u, du);
    REQUIRE_THAT(get<si::D>(du), Approx(get<si::D>(du_z)));
}
