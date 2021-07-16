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
constexpr auto f4 =
    vs::transform([](auto&& x) { return x * x * x * x + 2 * x * x * x - 3 * x + 1; });

constexpr auto f4_ddx = [](auto&& x) { return 12 * x * x + 12 * x; };

constexpr auto f2 = vs::transform([](auto&& x) { return -2 * x * x + 3 * x - 1; });
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
