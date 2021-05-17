#include "laplacian.hpp"

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

TEST_CASE("E2_2 Domain")
{
    using T = std::vector<real>;

    const auto extents = int3{5, 6, 7};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};

    // initialize fields
    scalar<T> u{};
    u | sel::D = m.location() | vs::transform(f2);

    SECTION("DDFFFD")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::fd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        scalar<T> ddx{u}, ddy{u}, ddz{u};
        ddx | sel::D = m.location() | vs::transform(f2_ddx);
        ddy | sel::D = m.location() | vs::transform(f2_ddy);
        ddz | sel::D = m.location() | vs::transform(f2_ddz);
        scalar<T> ex = ddx + ddy + ddz;

        // zero boundaries
        ex | m.xmin = 0;
        ex | m.xmax = 0;
        ex | m.zmax = 0;

        scalar<T> du{u};
        REQUIRE((integer)rs::size(du | sel::D) == m.size());

        auto lap = laplacian{m, stencils::second::E2, gridBcs, objectBcs};
        du = lap(u);

        REQUIRE_THAT(get<si::D>(ex), Approx(get<si::D>(du)));
    }

    SECTION("DDFFND")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::nd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        scalar<T> ddx{u}, ddy{u}, ddz{u};
        ddx | sel::D = m.location() | vs::transform(f2_ddx);
        ddy | sel::D = m.location() | vs::transform(f2_ddy);
        ddz | sel::D = m.location() | vs::transform(f2_ddz);
        scalar<T> ex = ddx + ddy + ddz;

        // zero boundaries
        ex | m.xmin = 0;
        ex | m.xmax = 0;
        ex | m.zmax = 0;

        // neumann
        scalar<T> nu{u};
        nu | sel::D = m.location() | vs::transform(f2_dz);

        scalar<T> du{u};
        REQUIRE((integer)rs::size(du | sel::D) == m.size());

        auto lap = laplacian{m, stencils::second::E2, gridBcs, objectBcs};
        du = lap(u, nu);

        REQUIRE_THAT(get<si::D>(ex), Approx(get<si::D>(du)));
    }
}

TEST_CASE("E2 with Objects")
{
    using T = std::vector<real>;

    const auto extents = int3{25, 26, 27};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}},
                  std::vector<shape>{make_sphere(0, real3{0.45, 1.011, 1.31}, 0.25)}};

    const auto gridBcs = bcs::Grid{bcs::df, bcs::nn, bcs::fd};
    const auto objectBcs = bcs::Object{bcs::Dirichlet};

    // initialize fields
    scalar<T> u{};
    u | sel::D = m.location() | vs::transform(f2);
    u | sel::R = m.location() | vs::transform(f2);
    REQUIRE(rs::size(u | sel::Rx) == m.Rx().size());

    // set the exact du we expect based on zeros assigned to dirichlet locations
    scalar<T> ddx{u}, ddy{u}, ddz{u};
    ddx | sel::D = m.location() | vs::transform(f2_ddx);
    ddy | sel::D = m.location() | vs::transform(f2_ddy);
    ddz | sel::D = m.location() | vs::transform(f2_ddz);
    scalar<T> ex{u};
    ex | sel::D = 0;

    ex | m.fluid = (ddx + ddy + ddz) | m.fluid;

    // zero dirichlet boundaries
    ex | m.xmin = 0;
    ex | m.zmax = 0;

    // neumann conditions
    scalar<T> nu{u};
    nu | sel::D = m.location() | vs::transform(f2_dy);

    scalar<T> du{u};

    auto lap = laplacian{m, stencils::second::E2, gridBcs, objectBcs};
    du = lap(u, nu);

    REQUIRE_THAT(get<si::D>(ex), Approx(get<si::D>(du)));
}
