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

TEST_CASE("dirichlet")
{
    auto st = stencils::second::E2;

    auto [p, r, t, x] = st.query_max();
    REQUIRE(p == 1);
    REQUIRE(r == 2);
    REQUIRE(t == 4);
    REQUIRE(x == 2);

    auto q = st.query(bcs::Dirichlet);
    REQUIRE(q.p == p);
    REQUIRE(q.r == r - 1);
    REQUIRE(q.t == t);
    REQUIRE(q.nextra == 0);

    std::vector<real> left(q.r * t);
    std::vector<real> right(q.r * t);
    std::vector<real> extra{};

    st.nbs(0.5, bcs::Dirichlet, 0.0, false, left, extra);
    REQUIRE_THAT(left, Approx(std::vector{4., 0., -8., 4.}));

    st.nbs(0.5, bcs::Dirichlet, 0.9, true, right, extra);
    REQUIRE_THAT(right,
                 Approx(std::vector{
                     0.5241379310344828, 2.610526315789474, -7.2, 4.0653357531760435}));
}

TEST_CASE("floating")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(           
            simulation = {
                scheme = {
                    order = 2,
                    type = "E2"
                }
            }
        )");
    auto st_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);
    const auto& st = *st_opt;

    auto [p, r, t, x] = st.query(bcs::Floating);
    REQUIRE(p == 1);
    REQUIRE(r == 2);
    REQUIRE(t == 4);
    REQUIRE(x == 0);

    std::vector<real> c(r * t);
    std::vector<real> extra{};

    st.nbs(0.5, bcs::Floating, 0.0, false, c, extra);
    REQUIRE_THAT(c, Approx(std::vector{0., 4., -8., 4., 4., 0., -8., 4.}));

    st.nbs(0.5, bcs::Floating, 0.5, true, c, extra);
    REQUIRE_THAT(
        c,
        Approx(std::vector{
            2.4, -2.6666666666666665, -4., 4.266666666666667, 3.25, -5.5, 0.25, 2.}));
}

TEST_CASE("neumann")
{
    auto st = stencils::make_E2_2();

    auto [p, r, t, x] = st.query(bcs::Neumann);
    REQUIRE(p == 1);
    REQUIRE(r == 2);
    REQUIRE(t == 4);
    REQUIRE(x == 2);

    std::vector<real> c(r * t);
    std::vector<real> extra(x);

    st.nbs(0.5, bcs::Neumann, 0.0, false, c, extra);
    REQUIRE_THAT(c, Approx(std::vector{-8., 0., 8., 0., 0., -8., 8., 0.}));
    REQUIRE_THAT(extra, Approx(std::vector{-4., -4.}));

    st.nbs(0.5, bcs::Neumann, 0.8, true, c, extra);
    REQUIRE_THAT(c,
                 Approx(std::vector{-0.384,
                                    4.928,
                                    -7.744,
                                    3.2,
                                    -0.45714285714285713,
                                    2.311111111111111,
                                    6.4,
                                    -8.253968253968255}));
    REQUIRE_THAT(extra, Approx(std::vector{0.8, 4.}));
}

TEST_CASE("interp")
{
    using T = std::vector<real>;
    auto st = stencils::make_E2_2();

    T ci(3);

    constexpr auto f = [](auto&& x) { return 3. * x * x - 10. * x + 0.1; };
    const real h = 10.0;
    const auto mesh = vs::linear_distribute(-h, h, 3);

    SECTION("interior direct")
    {

        {
            auto v = st.interp_interior(0.0, ci);
            real r = rs::inner_product(v, mesh | vs::transform(f), 0.0);
            REQUIRE(r == Catch::Approx(f(0.0)));
        }

        {
            auto v = st.interp_interior(0.5, ci);
            real r = rs::inner_product(v, mesh | vs::transform(f), 0.0);
            REQUIRE(r == Catch::Approx(f(h / 2)));
        }

        {
            auto v = st.interp_interior(-0.3, ci);
            real r = rs::inner_product(v, mesh | vs::transform(f), 0.0);
            REQUIRE(r == Catch::Approx(f(-3 * h / 10.)));
        }
    }

    SECTION("interior driver")
    {

        {
            real y = 0.0;
            auto&& [v, left, right] =
                st.interp(2,
                          int3{-10, -10, 1},
                          y,
                          boundary{int3{-5000, -5000, 0}, std::nullopt},
                          boundary{int3{5000, 5000, 2}, std::nullopt},
                          ci);
            REQUIRE(!left.object);
            REQUIRE(!right.object);
            real r = rs::inner_product(v, mesh | vs::transform(f), 0.0);
            REQUIRE(r == Catch::Approx(f(y * h)));
        }

        {
            real y = -0.3;
            auto&& [v, left, right] =
                st.interp(1,
                          int3{-10, -10, 1},
                          y,
                          boundary{int3{-5000, -5000, 0}, std::nullopt},
                          boundary{int3{5000, 5000, 2}, std::nullopt},
                          ci);
            REQUIRE(!left.object);
            REQUIRE(!right.object);
            real r = rs::inner_product(v, mesh | vs::transform(f), 0.0);
            REQUIRE(r == Catch::Approx(f(y * h)));
        }
    }

    T cw(4);

    SECTION("wall direct")
    {

        {
            real psi = 1.0;
            real y = 0.1;
            auto m = vs::concat(vs::single(-h - psi * h), mesh) | rs::to<T>();
            auto v = st.interp_wall(0, y, psi, cw, false);
            real r = rs::inner_product(v, m | vs::transform(f), 0.0);
            REQUIRE(r == Catch::Approx(f(-h - psi * h + y * h)));
        }
        {
            real psi = 0.9;
            real y = 0.0;
            auto m = vs::concat(vs::single(-h - psi * h), mesh) | rs::to<T>();
            auto v = st.interp_wall(0, 0, psi, cw, false);
            real r = rs::inner_product(v, m | vs::transform(f), 0.0);
            REQUIRE(r == Catch::Approx(f(-h - psi * h + y * h)));
        }

        {
            real psi = 0.;
            real y = 0.4;
            auto m = vs::concat(vs::single(-h - psi * h), mesh) | rs::to<T>();
            auto v = st.interp_wall(1, y, psi, cw, false);
            real r = rs::inner_product(v, m | vs::transform(f), 0.0);
            REQUIRE(r == Catch::Approx(f(-h + y * h)));
        }

        {
            real psi = 1.0;
            real y = -0.2;
            auto m = vs::concat(mesh, vs::single(h + psi * h)) | rs::to<T>();
            auto v = st.interp_wall(0, y, psi, cw, true);
            real r = rs::inner_product(v, m | vs::transform(f), 0.0);
            REQUIRE(r == Catch::Approx(f(h + psi * h + y * h)));
        }

        {
            real psi = 0.01;
            real y = -0.3;
            auto m = vs::concat(mesh, vs::single(h + psi * h)) | rs::to<T>();
            auto v = st.interp_wall(1, y, psi, cw, true);
            real r = rs::inner_product(v, m | vs::transform(f), 0.0);
            REQUIRE(r == Catch::Approx(f(h + y * h)));
        }
    }

    SECTION("wall driver")
    {

        {
            real psi = 0.8;
            real y = 0.3;
            auto&& [v, left, right] =
                st.interp(0,
                          int3{0, 1, 1},
                          y,
                          boundary{int3{0, 1, 1}, object_boundary{0, 1, psi}},
                          boundary{int3{3, 1, 1}, std::nullopt},
                          cw);
            REQUIRE(left.object);
            REQUIRE(!right.object);

            auto m = vs::concat(vs::single(-h - psi * h), mesh) | rs::to<T>();
            real r = rs::inner_product(v, m | vs::transform(f), 0.0);

            // y is relataive to the solid point rather than the cut-location
            REQUIRE(r == Catch::Approx(f(-2 * h + y * h)));
        }

        {
            real psi = 1.0;
            real y = 0.1;
            auto&& [v, left, right] = st.interp(0,
                                                int3{0, 1, 1},
                                                y,
                                                boundary{int3{0, 1, 1}, std::nullopt},
                                                boundary{int3{3, 1, 1}, std::nullopt},
                                                cw);
            REQUIRE(!left.object);
            REQUIRE(!right.object);

            auto m = vs::concat(vs::single(-h - psi * h), mesh) | rs::to<T>();
            real r = rs::inner_product(v, m | vs::transform(f), 0.0);

            REQUIRE(r == Catch::Approx(f(-2 * h + y * h)));
        }

        {
            real psi = 0.;
            real y = 0.4;
            auto&& [v, left, right] =
                st.interp(2,
                          int3{10, 10, 1},
                          y,
                          boundary{int3{1, 1, 0}, object_boundary{0, 1, psi}},
                          boundary{int3{1, 1, 3}, std::nullopt},
                          cw);
            REQUIRE(left.object);
            REQUIRE(!right.object);

            auto m = vs::concat(vs::single(-h - psi * h), mesh) | rs::to<T>();
            real r = rs::inner_product(v, m | vs::transform(f), 0.0);

            REQUIRE(r == Catch::Approx(f(-h + y * h)));
        }

        {
            real psi = 1.0;
            real y = -0.2;
            auto&& [v, left, right] = st.interp(0,
                                                int3{10, 1, 1},
                                                y,
                                                boundary{int3{-10, 1, 1}, std::nullopt},
                                                boundary{int3{10, 1, 1}, std::nullopt},
                                                cw);
            REQUIRE(!left.object);
            REQUIRE(!right.object);

            auto m = vs::concat(mesh, vs::single(h + psi * h)) | rs::to<T>();
            real r = rs::inner_product(v, m | vs::transform(f), 0.0);

            REQUIRE(r == Catch::Approx(f(2 * h + y * h)));
        }

        {
            real psi = 0.9;
            real y = -0.2;
            auto&& [v, left, right] =
                st.interp(0,
                          int3{10, 1, 1},
                          y,
                          boundary{int3{-10, 1, 1}, std::nullopt},
                          boundary{int3{10, 1, 1}, object_boundary{0, 3, psi}},
                          cw);
            REQUIRE(!left.object);
            REQUIRE(right.object);

            auto m = vs::concat(mesh, vs::single(h + psi * h)) | rs::to<T>();
            real r = rs::inner_product(v, m | vs::transform(f), 0.0);

            REQUIRE(r == Catch::Approx(f(2 * h + y * h)));
        }

        {
            real psi = 0.01;
            real y = -0.3;
            auto&& [v, left, right] =
                st.interp(2,
                          int3{0, 11, 15},
                          y,
                          boundary{int3{-10, 1, 1}, std::nullopt},
                          boundary{int3{0, 11, 16}, object_boundary{0, 3, psi}},
                          cw);
            REQUIRE(!left.object);
            REQUIRE(right.object);

            auto m = vs::concat(mesh, vs::single(h + psi * h)) | rs::to<T>();
            real r = rs::inner_product(v, m | vs::transform(f), 0.0);

            REQUIRE(r == Catch::Approx(f(h + y * h)));
        }
    }
}
