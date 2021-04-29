#include "Laplacian.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "IdentityStencil.hpp"
#include "fields/Selector.hpp"
#include "random/random.hpp"
#include "stencils/Stencils.hpp"

#include <range/v3/all.hpp>

using namespace ccs;

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
    using namespace ccs;
    using namespace mesh;
    using namespace field;
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    const auto extents = int3{5, 6, 7};

    auto m = Mesh{IndexExtents{extents},
                  DomainBounds{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};

    // initialize fields
    Scalar<T> u{};
    u | selector::D = m.location() | vs::transform(f2);

    SECTION("DDFFFD")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::fd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        Scalar<T> ddx{u}, ddy{u}, ddz{u};
        ddx | selector::D = m.location() | vs::transform(f2_ddx);
        ddy | selector::D = m.location() | vs::transform(f2_ddy);
        ddz | selector::D = m.location() | vs::transform(f2_ddz);
        Scalar<T> ex = ddx + ddy + ddz;

        // zero boundaries
        ex | selector::D | m.xmin() = 0;
        ex | selector::D | m.xmax() = 0;
        ex | selector::D | m.zmax() = 0;

        Scalar<T> du{u};
        REQUIRE((integer)rs::size(du | selector::D) == m.size());

        auto lap = operators::Laplacian{m, stencils::second::E2, gridBcs, objectBcs};
        du = lap(u);

        REQUIRE_THAT(get<selector::scalar::D>(ex), Approx(get<selector::scalar::D>(du)));
    }

    SECTION("DDFFND")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::nd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        Scalar<T> ddx{u}, ddy{u}, ddz{u};
        ddx | selector::D = m.location() | vs::transform(f2_ddx);
        ddy | selector::D = m.location() | vs::transform(f2_ddy);
        ddz | selector::D = m.location() | vs::transform(f2_ddz);
        Scalar<T> ex = ddx + ddy + ddz;

        // zero boundaries
        ex | selector::D | m.xmin() = 0;
        ex | selector::D | m.xmax() = 0;
        ex | selector::D | m.zmax() = 0;

        // neumann
        Scalar<T> nu{u};
        nu | selector::D = m.location() | vs::transform(f2_dz);

        Scalar<T> du{u};
        REQUIRE((integer)rs::size(du | selector::D) == m.size());

        auto lap = operators::Laplacian{m, stencils::second::E2, gridBcs, objectBcs};
        du = lap(u, nu);

        REQUIRE_THAT(get<selector::scalar::D>(ex), Approx(get<selector::scalar::D>(du)));
    }
}

TEST_CASE("E2 with Objects")
{
    using namespace ccs;
    using namespace mesh;
    using namespace field;
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    const auto extents = int3{25, 26, 27};

    auto m = Mesh{IndexExtents{extents},
                  DomainBounds{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}},
                  std::vector<shape>{make_sphere(0, real3{0.45, 1.011, 1.31}, 0.25)}};

    const auto gridBcs = bcs::Grid{bcs::df, bcs::nn, bcs::fd};
    const auto objectBcs = bcs::Object{bcs::Dirichlet};

    // initialize fields
    Scalar<T> u{};
    u | selector::D = m.location() | vs::transform(f2);
    u | selector::R = m.location() | vs::transform(f2);
    REQUIRE(rs::size(u | selector::Rx) == m.Rx().size());

    // set the exact du we expect based on zeros assigned to dirichlet locations
    Scalar<T> ddx{u}, ddy{u}, ddz{u};
    ddx | selector::D = m.location() | vs::transform(f2_ddx);
    ddy | selector::D = m.location() | vs::transform(f2_ddy);
    ddz | selector::D = m.location() | vs::transform(f2_ddz);
    Scalar<T> ex{u};
    ex | selector::D = 0;
    {
        auto out = ex | selector::D | m.F();
        auto sum = (ddx + ddy + ddz) | selector::D | m.F();
        rs::copy(sum, rs::begin(out));
    }

    // zero dirichlet boundaries
    ex | selector::D | m.xmin() = 0;
    ex | selector::D | m.zmax() = 0;

    // neumann conditions
    Scalar<T> nu{u};
    nu | selector::D = m.location() | vs::transform(f2_dy);

    Scalar<T> du{u};

    auto lap = operators::Laplacian{m, stencils::second::E2, gridBcs, objectBcs};
    du = lap(u, nu);

    REQUIRE_THAT(get<selector::scalar::D>(ex), Approx(get<selector::scalar::D>(du)));
}
