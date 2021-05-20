#include "stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <vector>

#include <range/v3/view/zip.hpp>

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
