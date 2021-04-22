#include "Derivative.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "IdentityStencil.hpp"
#include "fields/Selector.hpp"
#include "random/random.hpp"
#include "stencils/Stencils.hpp"

#include <range/v3/algorithm/mismatch.hpp>
#include <range/v3/all.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/generate_n.hpp>
#include <range/v3/view/stride.hpp>

using namespace ccs;

constexpr auto g = []() { return pick(); };

// 2nd order polynomial for use with E2
constexpr auto f2 = [](auto&& loc) {
    auto&& [x, y, z] = loc;
    return x * x * (y + z) + y * y * (x + z) + z * z * (x + y) + 3 * x * y * z + x + y +
           z;
};

TEST_CASE("Identity FFFFFF")
{
    using namespace ccs;
    using namespace mesh;
    using namespace field;
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    const auto extents = int3{5, 7, 6};

    auto m =
        Mesh{DomainBounds{.min = {-1, -1, 0}, .max = {1, 2, 2.2}}, IndexExtents{extents}};

    const auto gridBcs = bcs::Grid{bcs::ff, bcs::ff, bcs::ff};
    const auto objectBcs = bcs::Object{};

    // initialize fields
    randomize();
    Scalar<T> u{};
    u | selector::D = vs::generate_n(g, m.size());

    Scalar<T> du{u};
    REQUIRE((integer)rs::size(du | selector::D) == m.size());

    for (int i = 0; i < 3; i++) {
        auto d = operators::Derivative{i, m, stencils::Identity, gridBcs, objectBcs};
        d(u, du);

        REQUIRE_THAT(get<selector::scalar::D>(u), Approx(get<selector::scalar::D>(du)));
    }
}

TEST_CASE("E2_2 FFFFFF")
{
    using namespace ccs;
    using namespace mesh;
    using namespace field;
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    const auto extents = int3{5, 7, 6};

    auto m =
        Mesh{DomainBounds{.min = {-1, -1, 0}, .max = {1, 2, 2.2}}, IndexExtents{extents}};

    const auto gridBcs = bcs::Grid{bcs::ff, bcs::ff, bcs::ff};
    const auto objectBcs = bcs::Object{};

    // initialize fields
    Scalar<T> u{};
    u | selector::D = m.location() | vs::transform(f2);

    Scalar<T> du{u};
    REQUIRE((integer)rs::size(du | selector::D) == m.size());

    for (int i = 0; i < 3; i++) {
        auto d = operators::Derivative{i, m, stencils::Identity, gridBcs, objectBcs};
        d(u, du);

        REQUIRE_THAT(get<selector::scalar::D>(u), Approx(get<selector::scalar::D>(du)));
    }
}

TEST_CASE("Identity Mixed")
{
    using namespace ccs;
    using namespace mesh;
    using namespace field;
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    const auto extents = int3{5, 7, 6};

    auto m =
        Mesh{DomainBounds{.min = {-1, -1, 0}, .max = {1, 2, 2.2}}, IndexExtents{extents}};

    const auto objectBcs = bcs::Object{};

    // initialize fields
    randomize();
    Scalar<T> u{};
    u | selector::D = vs::generate_n(g, m.size());

    SECTION("DDFFFD")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::fd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        Scalar<T> du_exact{u};
        {
            // auto [nx, ny, nz] = extents;
            auto domain = du_exact | selector::D;
            // set zeros for dirichlet at xmin/xmax
            domain | m.xmin() = 0;
            domain | m.xmax() = 0;
            domain | m.zmax() = 0;
        }

        Scalar<T> du{u};
        REQUIRE((integer)rs::size(du | selector::D) == m.size());

        for (int i = 0; i < m.dims(); i++) {
            auto d = operators::Derivative{i, m, stencils::Identity, gridBcs, objectBcs};
            d(u, du);

            REQUIRE_THAT(get<selector::scalar::D>(du_exact),
                         Approx(get<selector::scalar::D>(du)));
        }
    }

    SECTION("FFDDDF")
    {

        const auto gridBcs = bcs::Grid{bcs::ff, bcs::dd, bcs::df};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        Scalar<T> du_exact{u};
        {
            // auto [nx, ny, nz] = extents;
            auto domain = du_exact | selector::D;
            // set zeros for dirichlet at xmin/xmax
            domain | m.ymin() = 0;
            domain | m.ymax() = 0;
            domain | m.zmin() = 0;
        }

        Scalar<T> du{u};
        REQUIRE((integer)rs::size(du | selector::D) == m.size());

        for (int i = 0; i < m.dims(); i++) {
            auto d = operators::Derivative{i, m, stencils::Identity, gridBcs, objectBcs};
            d(u, du);

            REQUIRE_THAT(get<selector::scalar::D>(du_exact),
                         Approx(get<selector::scalar::D>(du)));
        }
    }
}

TEST_CASE("Identity with Objects")
{
    using namespace ccs;
    using namespace mesh;
    using namespace field;
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    const auto extents = int3{16, 19, 18};

    auto m = Mesh{DomainBounds{.min = {-1, -1, 0}, .max = {1, 2, 2.2}},
                  IndexExtents{extents},
                  std::vector<shape>{make_sphere(0, real3{0.01, -0.01, 0.99}, 0.25)}};

    const auto gridBcs = bcs::Grid{bcs::ff, bcs::ff, bcs::ff};
    const auto objectBcs = bcs::Object{bcs::Dirichlet};

    // initialize fields
    randomize();
    Scalar<T> u{};
    u | selector::D = vs::generate_n(g, m.size());
    // note that the transform function doesn't do what we want if the simply do something
    // like:
    //
    // return vs::generator_n(g, rs::size(s))
    //
    // There must be some kind of mismatch between the transform/resize_and_copy routines
    // and the kind of view return by this lambda...
    auto tt = tuple::transform(
        [g = g](auto&& s) { return vs::generate_n(g, rs::size(s)) | rs::to<T>(); },
        m.R());
    REQUIRE(rs::size(get<0>(tt)) == rs::size(m.Rx()));

    u | selector::R = tt;
    REQUIRE(rs::size(u | selector::Rx) == m.Rx().size());

    Scalar<T> du_x{u};
    Scalar<T> du_y{u};
    Scalar<T> du_z{u};
    REQUIRE((integer)rs::size(du_x | selector::D) == m.size());

    auto dx = operators::Derivative{0, m, stencils::Identity, gridBcs, objectBcs};
    auto dy = operators::Derivative{1, m, stencils::Identity, gridBcs, objectBcs};
    auto dz = operators::Derivative{2, m, stencils::Identity, gridBcs, objectBcs};

    dx(u, du_x);
    dy(u, du_y);
    dz(u, du_z);

    REQUIRE_THAT(get<selector::scalar::D>(du_x), Approx(get<selector::scalar::D>(du_y)));
    REQUIRE_THAT(get<selector::scalar::D>(du_x), Approx(get<selector::scalar::D>(du_z)));
}