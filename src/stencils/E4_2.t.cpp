#include "stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <vector>

#include <range/v3/all.hpp>

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

using Catch::Matchers::Approx;
using namespace ccs;

// 4th order polynomial for use with E4
constexpr auto f4_f = [](auto&& x) { return x * x * x * x + 2 * x * x * x - 3 * x + 1; };
constexpr auto f4 = vs::transform(f4_f);
constexpr auto f4_ddx = [](auto&& x) { return 12 * x * x + 12 * x; };

constexpr auto f3_f = [](auto&& x) { return x * x * x - 3 * x * x + 12 * x - 4; };
constexpr auto f3 = vs::transform(f3_f);

constexpr auto f2_f = [](auto&& x) { return -2 * x * x + 3 * x - 1; };
constexpr auto f2 = vs::transform(f2_f);
constexpr auto f2_dx = [](auto&& x) { return -4. * x + 3; };
constexpr auto f2_ddx = [](auto&& x) { return -4.; };

constexpr real ymin = -1.;
constexpr real ymax = 5.;
using T = std::vector<real>;

TEST_CASE("interior")
{
    auto st = stencils::second::E4;

    auto [p, r, t, x] = st.query_max();
    REQUIRE(p == 2);
    REQUIRE(r == 3);
    REQUIRE(t == 5);
    REQUIRE(x == 3);

    const auto mesh = vs::linear_distribute(ymin, ymax, 2 * p + 1) | rs::to<T>();
    const auto h = mesh[1] - mesh[0];
    T c(2 * p + 1);

    st.interior(h, c);

    REQUIRE(rs::inner_product(c, mesh | f4, 0.) == Catch::Approx(f4_ddx(mesh[2])));
}

TEST_CASE("floating")
{
    auto st = stencils::second::E4;

    auto [p, r, t, x] = st.query(bcs::Floating);
    REQUIRE(p == 2);
    REQUIRE(r == 3);
    REQUIRE(t == 5);
    REQUIRE(x == 0);

    T c(r * t);
    T ex(x);
    T mesh = vs::linear_distribute(ymin, ymax, t - 1) | rs::to<T>();
    real h = mesh[1] - mesh[0];
    real psi = 0.2;

    {
        T m = vs::concat(vs::single(ymin - psi * h), mesh) | rs::to<T>();
        REQUIRE((int)m.size() == t);

        st.nbs(h, bcs::Floating, psi, false, c, ex);

        for (int i = 0; i < r; i++)
            REQUIRE(rs::inner_product(c | vs::drop(i * t) | vs::take_exactly(t),
                                      m | f2,
                                      0.) == Catch::Approx(f2_ddx(m[i])));
    }

    {
        T m = vs::concat(mesh, vs::single(ymax + psi * h)) | rs::to<T>();
        REQUIRE((int)m.size() == t);

        st.nbs(h, bcs::Floating, psi, true, c, ex);

        for (int i = 0; i < r; i++)
            REQUIRE(rs::inner_product(c | vs::drop(i * t) | vs::take_exactly(t),
                                      m | f2,
                                      0.) == Catch::Approx(f2_ddx(m[t - r + i])));
    }
}

TEST_CASE("dirichlet")
{
    auto st = stencils::second::E4;

    auto [p, r, t, x] = st.query(bcs::Dirichlet);
    REQUIRE(p == 2);
    REQUIRE(r == 2);
    REQUIRE(t == 5);
    REQUIRE(x == 0);

    T c(r * t);
    T ex(x);
    T mesh = vs::linear_distribute(ymin, ymax, t - 1) | rs::to<T>();
    real h = mesh[1] - mesh[0];
    real psi = 0.0;

    {
        T m = vs::concat(vs::single(ymin - psi * h), mesh) | rs::to<T>();
        REQUIRE((int)m.size() == t);

        st.nbs(h, bcs::Dirichlet, psi, false, c, ex);

        for (int i = 0; i < r; i++)
            REQUIRE(rs::inner_product(c | vs::drop(i * t) | vs::take_exactly(t),
                                      m | f2,
                                      0.) == Catch::Approx(f2_ddx(m[i + 1])));
    }

    {
        T m = vs::concat(mesh, vs::single(ymax + psi * h)) | rs::to<T>();
        REQUIRE((int)m.size() == t);

        st.nbs(h, bcs::Dirichlet, psi, true, c, ex);

        for (int i = 0; i < r; i++)
            REQUIRE(rs::inner_product(c | vs::drop(i * t) | vs::take_exactly(t),
                                      m | f2,
                                      0.) == Catch::Approx(f2_ddx(m[t - r - 1 + i])));
    }
}

TEST_CASE("neumann")
{
    auto st = stencils::second::E4;

    auto [p, r, t, x] = st.query(bcs::Neumann);
    REQUIRE(p == 2);
    REQUIRE(r == 3);
    REQUIRE(t == 5);
    REQUIRE(x == 3);

    T c(r * t);
    T ex(x);
    T mesh = vs::linear_distribute(ymin, ymax, t - 1) | rs::to<T>();
    real h = mesh[1] - mesh[0];
    real psi = 0.8;

    {
        T m = vs::concat(vs::single(ymin - psi * h), mesh) | rs::to<T>();
        REQUIRE((int)m.size() == t);

        st.nbs(h, bcs::Neumann, psi, false, c, ex);

        for (int i = 0; i < r; i++)
            REQUIRE(rs::inner_product(c | vs::drop(i * t) | vs::take_exactly(t),
                                      m | f2,
                                      ex[i] * f2_dx(m[0])) ==
                    Catch::Approx(f2_ddx(m[i])));
    }

    {
        T m = vs::concat(mesh, vs::single(ymax + psi * h)) | rs::to<T>();
        REQUIRE((int)m.size() == t);

        st.nbs(h, bcs::Neumann, psi, true, c, ex);

        for (int i = 0; i < r; i++)
            REQUIRE(rs::inner_product(c | vs::drop(i * t) | vs::take_exactly(t),
                                      m | f2,
                                      ex[i] * f2_dx(m[t - 1])) ==
                    Catch::Approx(f2_ddx(m[t - r + i])));
    }
}

TEST_CASE("interp interior")
{
    auto st = stencils::make_E4_2();
    auto p = st.query_max().p;

    T c(2 * p + 1);
    T mesh = vs::linear_distribute(ymin, ymax, 2 * p + 1) | rs::to<T>();
    const real h = mesh[1] - mesh[0];

    {
        real y = -0.4;
        auto&& [v, l, r] = st.interp(2,
                                     int3{1, 2, p},
                                     y,
                                     boundary(int3{1, 2, 0}, std::nullopt),
                                     boundary(int3{1, 2, 2 * p + 1}, std::nullopt),
                                     c);

        REQUIRE(!l.object);
        REQUIRE(!r.object);
        real yi = rs::inner_product(v, mesh | f4, 0.);
        REQUIRE(yi == Catch::Approx(f4_f(mesh[p] + y * h)));
    }
}

TEST_CASE("interp wall")
{
    auto st = stencils::make_E4_2();
    auto t = st.query_max().t;

    T c(t);
    T mesh = vs::linear_distribute(ymin, ymax, t - 1) | rs::to<T>();
    const real h = mesh[1] - mesh[0];

    // left 0
    {
        const real psi = 0.8, y = 0.3;
        auto m = vs::concat(vs::single(ymin - psi * h), mesh) | rs::to<T>();
        auto&& [v, l, r] = st.interp(0,
                                     int3{8, 4, 5},
                                     y,
                                     boundary{int3{8, 4, 5}, object_boundary{0, 1, psi}},
                                     boundary{int3{50, 4, 5}, std::nullopt},
                                     c);
        REQUIRE(l.object);
        REQUIRE(!r.object);
        real yi = rs::inner_product(v, m | f3, 0.);
        REQUIRE(yi == Catch::Approx(f3_f(ymin - h + y * h)));
    }
    // left 1
    {
        const real psi = 0.8, y = -0.2;
        auto m = vs::concat(vs::single(ymin - psi * h), mesh) | rs::to<T>();
        auto&& [v, l, r] = st.interp(0,
                                     int3{9, 4, 5},
                                     y,
                                     boundary{int3{8, 4, 5}, object_boundary{0, 1, psi}},
                                     boundary{int3{50, 4, 5}, std::nullopt},
                                     c);
        REQUIRE(l.object);
        REQUIRE(!r.object);
        real yi = rs::inner_product(v, m | f3, 0.);
        REQUIRE(yi == Catch::Approx(f3_f(ymin + y * h)));
    }
    // left 2
    {
        const real psi = 0.0, y = 0.41;
        auto m = vs::concat(vs::single(ymin - psi * h), mesh) | rs::to<T>();
        auto&& [v, l, r] = st.interp(0,
                                     int3{10, 4, 5},
                                     y,
                                     boundary{int3{8, 4, 5}, object_boundary{0, 1, psi}},
                                     boundary{int3{50, 4, 5}, std::nullopt},
                                     c);
        REQUIRE(l.object);
        REQUIRE(!r.object);
        real yi = rs::inner_product(v, m | f3, 0.);
        REQUIRE(yi == Catch::Approx(f3_f(mesh[1] + y * h)));
    }
    // right 0
    {
        const real psi = 0.9, y = -0.3;
        auto m = vs::concat(mesh, vs::single(ymax + psi * h)) | rs::to<T>();
        auto&& [v, l, r] = st.interp(1,
                                     int3{8, 9, 5},
                                     y,
                                     boundary{int3{8, 0, 5}, std::nullopt},
                                     boundary{int3{8, 9, 5}, object_boundary{0, 0, psi}},
                                     c);
        REQUIRE(!l.object);
        REQUIRE(r.object);
        real yi = rs::inner_product(v, m | f3, 0.);
        REQUIRE(yi == Catch::Approx(f3_f(ymax + h + y * h)));
    }

    // right 1
    {
        const real psi = 0.9, y = 0.3;
        auto m = vs::concat(mesh, vs::single(ymax + psi * h)) | rs::to<T>();
        auto&& [v, l, r] = st.interp(1,
                                     int3{8, 8, 5},
                                     y,
                                     boundary{int3{8, 0, 5}, std::nullopt},
                                     boundary{int3{8, 9, 5}, object_boundary{0, 0, psi}},
                                     c);
        REQUIRE(!l.object);
        REQUIRE(r.object);
        real yi = rs::inner_product(v, m | f3, 0.);
        REQUIRE(yi == Catch::Approx(f3_f(ymax + y * h)));
    }

    // right 2
    {
        const real psi = 1e-3, y = -0.4;
        auto m = vs::concat(mesh, vs::single(ymax + psi * h)) | rs::to<T>();
        auto&& [v, l, r] = st.interp(1,
                                     int3{8, 7, 5},
                                     y,
                                     boundary{int3{8, 0, 5}, std::nullopt},
                                     boundary{int3{8, 9, 5}, object_boundary{0, 0, psi}},
                                     c);
        REQUIRE(!l.object);
        REQUIRE(r.object);
        real yi = rs::inner_product(v, m | f3, 0.);
        REQUIRE(yi == Catch::Approx(f3_f(ymax - h + y * h)));
    }
}
