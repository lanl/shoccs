#include "Derivative.hpp"

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

TEST_CASE("E2_Neumann")
{
    using namespace ccs;
    using namespace mesh;
    using namespace field;
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    const auto extents = int3{10, 13, 17};

    auto m = Mesh{IndexExtents{extents},
                  DomainBounds{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};

    // initialize fields
    Scalar<T> u{};
    u | selector::D = m.location() | vs::transform(f2);
    Scalar<T> nu{u};
    nu | selector::D = m.location() | vs::transform(f2_dz);
    Scalar<T> ex{u};
    ex | selector::D = m.location() | vs::transform(f2_ddz);

    ex | selector::D | m.xmin() = 0;
    ex | selector::D | m.xmax() = 0;

    Scalar<T> du{u};
    du = 0;

    {
        const auto gridBcs = bcs::Grid{bcs::dd, bcs::ff, bcs::nn};
        auto d = operators::Derivative(2, m, stencils::second::E2, gridBcs, objectBcs);
        d(u, nu, du);
        REQUIRE_THAT(get<selector::scalar::D>(du), Approx(get<selector::scalar::D>(ex)));

        d(u, nu, du, plus_eq);
        ex *= 2;
        REQUIRE_THAT(get<selector::scalar::D>(du), Approx(get<selector::scalar::D>(ex)));
    }
}

TEST_CASE("Identity FFFFFF")
{
    using namespace ccs;
    using namespace mesh;
    using namespace field;
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    const auto extents = int3{5, 7, 6};

    auto m =
        Mesh{IndexExtents{extents}, DomainBounds{.min = {-1, -1, 0}, .max = {1, 2, 2.2}}};

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
        du = 0;
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

    // shift domain bounds away from zero to avoid problems with Catch::Approx
    auto m = Mesh{IndexExtents{extents},
                  DomainBounds{.min = {0.1, 0.2, 0.3}, .max = {1, 2, 2.2}}};

    const auto gridBcs = bcs::Grid{bcs::ff, bcs::ff, bcs::ff};
    const auto objectBcs = bcs::Object{};

    // initialize fields
    Scalar<T> u{};
    u | selector::D = m.location() | vs::transform(f2);

    Scalar<T> du{u};
    REQUIRE((integer)rs::size(du | selector::D) == m.size());

    // exact
    Scalar<T> dx{u}, dy{u}, dz{u};
    dx | selector::D = m.location() | vs::transform(f2_ddx);
    dy | selector::D = m.location() | vs::transform(f2_ddy);
    dz | selector::D = m.location() | vs::transform(f2_ddz);
    std::array<Scalar<T>*, 3> dd{&dx, &dy, &dz};

    for (int i = 0; i < 3; i++) {
        auto d = operators::Derivative{i, m, stencils::second::E2, gridBcs, objectBcs};
        du = 0;
        d(u, du);

        REQUIRE_THAT(get<selector::scalar::D>(*dd[i]),
                     Approx(get<selector::scalar::D>(du)));
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
        Mesh{IndexExtents{extents}, DomainBounds{.min = {-1, -1, 0}, .max = {1, 2, 2.2}}};

    const auto objectBcs = bcs::Object{};

    // initialize fields
    randomize();
    Scalar<T> u{};
    u | selector::D = vs::generate_n(g, m.size());

    SECTION("DDFNFD")
    {

        const auto gridBcs = bcs::Grid{bcs::dd, bcs::fn, bcs::fd};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        Scalar<T> du_exact{u}, nu{u};
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
            du = 0;
            d(u, nu, du);

            REQUIRE_THAT(get<selector::scalar::D>(du_exact),
                         Approx(get<selector::scalar::D>(du)));
        }
    }

    SECTION("NNDDDF")
    {

        const auto gridBcs = bcs::Grid{bcs::nn, bcs::dd, bcs::df};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        Scalar<T> du_exact{u}, nu{u};
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
            du = 0;
            d(u, nu, du);

            REQUIRE_THAT(get<selector::scalar::D>(du_exact),
                         Approx(get<selector::scalar::D>(du)));
        }
    }
}

TEST_CASE("E2 Mixed")
{
    using namespace ccs;
    using namespace mesh;
    using namespace field;
    using Catch::Matchers::Approx;
    using T = std::vector<real>;

    const auto extents = int3{5, 7, 6};

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
        Scalar<T> dx{u}, dy{u}, dz{u};
        dx | selector::D = m.location() | vs::transform(f2_ddx);
        dy | selector::D = m.location() | vs::transform(f2_ddy);
        dz | selector::D = m.location() | vs::transform(f2_ddz);
        std::array<Scalar<T>*, 3> dd{&dx, &dy, &dz};

        Scalar<T> du{u};
        REQUIRE((integer)rs::size(du | selector::D) == m.size());

        for (int i = 0; i < m.dims(); i++) {
            auto d =
                operators::Derivative{i, m, stencils::second::E2, gridBcs, objectBcs};
            du = 0;
            d(u, du);

            auto& ex = *dd[i];
            // zero boundaries
            ex | selector::D | m.xmin() = 0;
            ex | selector::D | m.xmax() = 0;
            ex | selector::D | m.zmax() = 0;

            REQUIRE_THAT(get<selector::scalar::D>(ex),
                         Approx(get<selector::scalar::D>(du)));
        }
    }

    SECTION("FNDDDF")
    {

        const auto gridBcs = bcs::Grid{bcs::fn, bcs::dd, bcs::df};
        // set the exact du we expect based on zeros assigned to dirichlet locations
        Scalar<T> dx{u}, dy{u}, dz{u}, nu{u};
        dx | selector::D = m.location() | vs::transform(f2_ddx);
        dy | selector::D = m.location() | vs::transform(f2_ddy);
        dz | selector::D = m.location() | vs::transform(f2_ddz);
        nu | selector::D = m.location() | vs::transform(f2_dx);
        std::array<Scalar<T>*, 3> dd{&dx, &dy, &dz};

        Scalar<T> du{u};
        REQUIRE((integer)rs::size(du | selector::D) == m.size());

        for (int i = 0; i < m.dims(); i++) {
            auto d =
                operators::Derivative{i, m, stencils::second::E2, gridBcs, objectBcs};
            du = 0;
            d(u, nu, du);

            auto& ex = *dd[i];
            // zero boundaries
            ex | selector::D | m.ymin() = 0;
            ex | selector::D | m.ymax() = 0;
            ex | selector::D | m.zmin() = 0;

            REQUIRE_THAT(get<selector::scalar::D>(ex),
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

    auto m = Mesh{IndexExtents{extents},
                  DomainBounds{.min = {-1, -1, 0}, .max = {1, 2, 2.2}},
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

    du_x = 0;
    dx(u, du_x);
    du_y = 0;
    dy(u, du_y);
    du_z = 0;
    dz(u, du_z);

    REQUIRE_THAT(get<selector::scalar::D>(du_x), Approx(get<selector::scalar::D>(du_y)));
    REQUIRE_THAT(get<selector::scalar::D>(du_x), Approx(get<selector::scalar::D>(du_z)));
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

    const auto gridBcs = bcs::Grid{bcs::nn, bcs::dd, bcs::ff};
    const auto objectBcs = bcs::Object{bcs::Dirichlet};

    // initialize fields
    Scalar<T> u{};
    u | selector::D = m.location() | vs::transform(f2);
    u | selector::R = m.location() | vs::transform(f2);
    REQUIRE(rs::size(u | selector::Rx) == m.Rx().size());

    Scalar<T> du_x{u}, du_y{u}, du_z{u}, dd{u}, nu{u};
    nu | selector::D = m.location() | vs::transform(f2_dx);
    dd | selector::D = m.location() | vs::transform(f2_ddx);
    dd | selector::D | m.ymin() = 0;
    dd | selector::D | m.ymax() = 0;
    du_x | selector::D = 0;
    {
        auto out = du_x | selector::D | m.F();
        rs::copy(dd | selector::D | m.F(), rs::begin(out));
    }
    // This direct copy doesn't work.. why?
    // du_x | selector::D | m.F() = dd | selector::D | m.F();
    dd | selector::D = m.location() | vs::transform(f2_ddy);
    dd | selector::D | m.ymin() = 0;
    dd | selector::D | m.ymax() = 0;
    du_y | selector::D = 0;
    {
        auto out = du_y | selector::D | m.F();
        rs::copy(dd | selector::D | m.F(), rs::begin(out));
    }

    dd | selector::D = m.location() | vs::transform(f2_ddz);
    dd | selector::D | m.ymin() = 0;
    dd | selector::D | m.ymax() = 0;
    du_z | selector::D = 0;
    {
        auto out = du_z | selector::D | m.F();
        rs::copy(dd | selector::D | m.F(), rs::begin(out));
    }

    REQUIRE((integer)rs::size(du_x | selector::D) == m.size());

    auto dx = operators::Derivative{0, m, stencils::second::E2, gridBcs, objectBcs};
    auto dy = operators::Derivative{1, m, stencils::second::E2, gridBcs, objectBcs};
    auto dz = operators::Derivative{2, m, stencils::second::E2, gridBcs, objectBcs};

    Scalar<T> du{u};

    du = 0;
    dx(u, nu, du);
    REQUIRE_THAT(get<selector::scalar::D>(du), Approx(get<selector::scalar::D>(du_x)));

    du = 0;
    dy(u, du);
    REQUIRE_THAT(get<selector::scalar::D>(du), Approx(get<selector::scalar::D>(du_y)));

    du = 0;
    dz(u, du);
    REQUIRE_THAT(get<selector::scalar::D>(du), Approx(get<selector::scalar::D>(du_z)));
}
