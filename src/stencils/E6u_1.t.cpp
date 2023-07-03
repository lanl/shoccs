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

TEST_CASE("E6u_1")
{
    using T = std::vector<real>;
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            scheme = {
                order = 1,
                type = "E6u",
                alpha = {0.1, 0.2, 0.3, 0.4, 0.5}
            }
        }
    )");

    auto st_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);
    const auto& st = *st_opt;

    {

        auto [p, r, t, x] = st.query(bcs::Floating);
        REQUIRE(p == 3);
        REQUIRE(r == 5);
        REQUIRE(t == 8);
        REQUIRE(x == 0);

        T c(r * t);
        T ex{};

        st.nbs(2, bcs::Floating, 1.0, false, c, ex);
        REQUIRE_THAT(c,
                     Approx(T{-1.0916666666666666,
                              2.2,
                              -1.75,
                              0.6666666666666666,
                              0.12500000000000006,
                              -0.2,
                              0.05,
                              0.0,
                              5.551115123125783e-18,
                              -1.1416666666666666,
                              2.5,
                              -2.5,
                              1.6666666666666667,
                              -0.625,
                              0.1,
                              0.0,
                              0.175,
                              -1.15,
                              2.083333333333333,
                              -2.5,
                              2.125,
                              -0.8833333333333333,
                              0.15,
                              0.0,
                              1.6833333333333333,
                              -9.825,
                              23.5,
                              -30.083333333333332,
                              20.75,
                              -6.475,
                              0.2,
                              0.25,
                              -2.1687367303609344,
                              12.723637650389243,
                              -30.773354564755838,
                              38.79299363057325,
                              -26.92206298655343,
                              9.18067940552017,
                              -0.5610403397027601,
                              -0.27211606510969566}).margin(1.0e-8));
    }
    {
        auto [p, r, t, x] = st.query(bcs::Dirichlet);
        REQUIRE(p == 3);
        REQUIRE(r == 4);
        REQUIRE(t == 8);
        REQUIRE(x == 0);

        T c(r * t);
        T ex{};

        st.nbs(0.5, bcs::Dirichlet, 0.0, false, c, ex);
        REQUIRE_THAT(c,
                     Approx(T{2.2204460492503132e-17,
                              -4.566666666666666,
                              10.0,
                              -10.0,
                              6.666666666666667,
                              -2.5,
                              0.4,
                              0.0,
                              0.7,
                              -4.6,
                              8.333333333333332,
                              -10.0,
                              8.5,
                              -3.533333333333333,
                              0.6,
                              0.0,
                              6.733333333333333,
                              -39.3,
                              94.0,
                              -120.33333333333333,
                              83.0,
                              -25.9,
                              0.8,
                              1.0,
                              -8.674946921443738,
                              50.89455060155697,
                              -123.09341825902335,
                              155.171974522293,
                              -107.68825194621373,
                              36.72271762208068,
                              -2.2441613588110405,
                              -1.0884642604387826})
                         .margin(1.0e-8));
    }
}
