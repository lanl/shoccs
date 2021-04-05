#include "Derivative.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "IdentityStencil.hpp"
#include "fields/Selector.hpp"
#include "random/random.hpp"

#include <range/v3/view/drop.hpp>
#include <range/v3/view/generate_n.hpp>
#include <range/v3/view/stride.hpp>

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
    u | selector::D = vs::generate_n([]() { return pick(); }, m.size());

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
    u | selector::D = vs::generate_n([]() { return pick(); }, m.size());

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