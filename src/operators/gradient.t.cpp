#include "gradient.hpp"

#include "fields/selector.hpp"
#include "stencils/stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <range/v3/all.hpp>

using namespace ccs;
using Catch::Matchers::Approx;

const std::vector<real> alpha{
    -1.47956280234494, 0.261900367793859, -0.145072532538541, -0.224665713988644};

// 2nd order polynomial for use with E2
constexpr auto f2 = vs::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x * (y + z) + y * (x + z) + z * (x + y) + 3 * x * y * z;
});

constexpr auto f2_dx = vs::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return 2. * (y + z) + 3. * y * z;
});

constexpr auto f2_dy = vs::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return 2. * (x + z) + 3. * x * z;
});

constexpr auto f2_dz = vs::transform([](auto&& loc) {
    auto&& [x, y, z] = loc;
    return 2. * (x + y) + 3. * x * y;
});

TEST_CASE("E2_1 Domain")
{
    using T = std::vector<real>;

    const auto extents = int3{15, 12, 13};

    auto m = mesh{index_extents{extents},
                  domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};
    const auto loc = m.xyz;
    const auto st = stencils::make_E2_1(alpha);

    // initialize fields
    scalar_real u{loc | f2};

    SECTION("DDFFFD")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::fd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        vector_real ex{loc | f2_dx, loc | f2_dy, loc | f2_dz};

        // zero boundaries
        ex | m.dirichlet(gridBcs) = 0;

        vector_real du{u, u, u};
        REQUIRE((integer)rs::size(du | sel::Dx) == m.size());

        auto grad = gradient{m, st, gridBcs, objectBcs};
        du = grad(u);

        REQUIRE_THAT(get<vi::Dx>(ex), Approx(get<vi::Dx>(du)));
        REQUIRE_THAT(get<vi::Dy>(ex), Approx(get<vi::Dy>(du)));
        REQUIRE_THAT(get<vi::Dz>(ex), Approx(get<vi::Dz>(du)));
    }
#if 0
    SECTION("DDFFND")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::nd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        scalar<T> ex = (loc | f2_ddx) + (loc | f2_ddy) + (loc | f2_ddz);

        // zero boundaries
        ex | m.dirichlet(gridBcs) = 0;

        // neumann
        scalar<T> nu{loc | f2_dz};

        scalar<T> du{m.ss()};
        REQUIRE((integer)rs::size(du | sel::D) == m.size());

        auto lap = laplacian{m, stencils::second::E2, gridBcs, objectBcs};
        du = lap(u, nu);

        REQUIRE_THAT(get<si::D>(ex), Approx(get<si::D>(du)));
    }
#endif
}
