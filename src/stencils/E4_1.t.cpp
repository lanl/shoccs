#include "stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <cmath>
#include <vector>

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

using Catch::Matchers::Approx;
using namespace ccs;

TEST_CASE("E4_1")
{
    using T = std::vector<real>;
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            scheme = {
                order = 1,
                type = "E4",
                alpha = {0.1, 0.7}
            }
        }
    )");

    auto st_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);
    const auto& st = *st_opt;

    {
        auto [p, r, t, x] = st.query(bcs::Floating);
        REQUIRE(p == 2);
        REQUIRE(r == 5);
        REQUIRE(t == 7);
        REQUIRE(x == 0);

        T c(35);
        T ex{};

        st.nbs(2.0, bcs::Floating, 0.9, false, c, ex);
        REQUIRE_THAT(c,
                     Approx(T{4.6526621634741545,
                              41.87395947126741,
                              -212.63681862466535,
                              340.460512074396,
                              -233.80445076088895,
                              59.4541356764168,
                              0.0,
                              0.4084070512442919,
                              3.675663461198627,
                              -19.39157224027908,
                              30.968406579006263,
                              -20.96594483306683,
                              5.3050399818967255,
                              0.0,
                              0.013333333333333329,
                              0.12,
                              -0.7698199999999998,
                              0.7120600000000001,
                              -0.15999333333333335,
                              0.08441999999999998,
                              0.0,
                              -0.016666666666666663,
                              -0.15,
                              0.837275,
                              -1.7025749999999995,
                              1.0749916666666666,
                              -0.043024999999999994,
                              0.0,
                              -3.496227944877181,
                              -35.394622932466035,
                              173.70473447605556,
                              -275.090007820069,
                              187.27582087173366,
                              -46.99969665037699,
                              0.0})
                         .margin(1.0e-8));
    }

    {
        auto [p, r, t, x] = st.query(bcs::Floating);
        REQUIRE(p == 2);
        REQUIRE(r == 5);
        REQUIRE(t == 7);
        REQUIRE(x == 0);

        T c(35);
        T ex{};

        st.nbs(1.0, bcs::Floating, 0.3, false, c, ex);
        REQUIRE_THAT(c,
                     Approx(T{24.485286095950443,
                              10.493694041121616,
                              -179.49582321674225,
                              298.30220055272866,
                              -206.3784378323039,
                              52.593080359245455,
                              0.0,
                              3.0326478495252283,
                              1.2997062212250978,
                              -23.528246195726204,
                              38.76986476744652,
                              -26.194201585573317,
                              6.620228943102668,
                              0.0,
                              0.18666666666666668,
                              0.08,
                              -1.6687066666666663,
                              1.7017200000000003,
                              -0.5199866666666668,
                              0.22030666666666668,
                              0.0,
                              -0.23333333333333334,
                              -0.09999999999999999,
                              1.8358833333333333,
                              -3.7521500000000003,
                              2.3999833333333336,
                              -0.15038333333333334,
                              0.0,
                              -7.673648231189963,
                              -5.023400262346717,
                              61.50107726961133,
                              -99.41829603446094,
                              66.42364870359154,
                              -15.809381445205277,
                              0.0})
                         .margin(1.0e-8));
    }

    {
        auto [p, r, t, x] = st.query(bcs::Dirichlet);
        REQUIRE(p == 2);
        REQUIRE(r == 4);
        REQUIRE(t == 7);
        REQUIRE(x == 0);

        T c(28);
        T ex{};

        st.nbs(0.5, bcs::Dirichlet, 0.7, false, c, ex);
        REQUIRE_THAT(c,
                     Approx(T{3.2734796914563766,
                              7.638119280064874,
                              -59.835759112418,
                              100.21565573965164,
                              -68.86186581215368,
                              17.57037021339881,
                              0.0,
                              0.16000000000000006,
                              0.3733333333333333,
                              -3.4120799999999996,
                              3.6050400000000002,
                              -1.2191733333333337,
                              0.4928800000000001,
                              0.0,
                              -0.2,
                              -0.4666666666666666,
                              3.7651000000000003,
                              -7.756300000000001,
                              5.0239666666666665,
                              -0.3661000000000002,
                              0.0,
                              -14.82664300469518,
                              -40.78597653476493,
                              278.50313430563523,
                              -460.87588278249143,
                              317.0714909092043,
                              -79.08612289288797,
                              0.0})
                         .margin(1.0e-8));
    }

    SECTION("Floating near psi=1 produces finite values")
    {
        auto [p, r, t, x] = st.query(bcs::Floating);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Floating, 1.0 - 1e-12, false, c, ex);
        for (std::size_t i = 0; i < c.size(); ++i) {
            REQUIRE(std::isfinite(c[i]));
        }
    }

    SECTION("Dirichlet near psi=1 produces finite values")
    {
        auto [p, r, t, x] = st.query(bcs::Dirichlet);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Dirichlet, 1.0 - 1e-12, false, c, ex);
        for (std::size_t i = 0; i < c.size(); ++i) {
            REQUIRE(std::isfinite(c[i]));
        }
    }

    SECTION("Floating near psi=snap_tol produces finite values")
    {
        auto [p, r, t, x] = st.query(bcs::Floating);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Floating, 1e-12, false, c, ex);
        for (std::size_t i = 0; i < c.size(); ++i) {
            REQUIRE(std::isfinite(c[i]));
        }
    }

    SECTION("Dirichlet near psi=snap_tol produces finite values")
    {
        auto [p, r, t, x] = st.query(bcs::Dirichlet);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Dirichlet, 1e-12, false, c, ex);
        for (std::size_t i = 0; i < c.size(); ++i) {
            REQUIRE(std::isfinite(c[i]));
        }
    }

    SECTION("Floating near psi=1: magnitude within safe bound")
    {
        auto [p, r, t, x] = st.query(bcs::Floating);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Floating, 1.0 - 1e-12, false, c, ex);
        real max_abs = 0.0;
        for (std::size_t i = 0; i < c.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(c[i]));
        }
        REQUIRE(max_abs < 1e8);
    }

    SECTION("Dirichlet near psi=1: magnitude within safe bound")
    {
        auto [p, r, t, x] = st.query(bcs::Dirichlet);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Dirichlet, 1.0 - 1e-12, false, c, ex);
        real max_abs = 0.0;
        for (std::size_t i = 0; i < c.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(c[i]));
        }
        REQUIRE(max_abs < 1e8);
    }

    SECTION("Floating near psi=0: magnitude within safe bound")
    {
        auto [p, r, t, x] = st.query(bcs::Floating);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Floating, 1e-12, false, c, ex);
        real max_abs = 0.0;
        for (std::size_t i = 0; i < c.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(c[i]));
        }
        REQUIRE(max_abs < 1e8);
    }

    SECTION("Dirichlet near psi=0: magnitude within safe bound")
    {
        auto [p, r, t, x] = st.query(bcs::Dirichlet);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Dirichlet, 1e-12, false, c, ex);
        for (std::size_t i = 0; i < c.size(); ++i) {
            REQUIRE(std::abs(c[i]) < 1e8);
        }
    }

    SECTION("No interior polynomial denominator singularity with alpha[1] >= 197/288")
    {
        // The denominator D(psi) = 1728*alpha[1] + 1584*alpha[1]*psi +
        //   864*alpha[1]*psi^2 + 144*alpha[1]*psi^3 + 12*psi^6 + 162*psi^5 +
        //   1464*psi^4 + 5617*psi^3 + 8070*psi^2 + 1721*psi - 1182.
        // With alpha[1]=0.7 >= 197/288 ~ 0.684, D(0) = 1728*0.7 - 1182 = 27.6 > 0.
        // D'(psi) > 0 for all psi >= 0 and alpha[1] > 0, so D is strictly
        // increasing; D(0) >= 0 ensures D(psi) > 0 for all psi in (0, 1).
        constexpr real alpha1 = 0.7;
        auto D = [](real psi) {
            return 1728 * alpha1 + 1584 * alpha1 * psi +
                   864 * alpha1 * psi * psi +
                   144 * alpha1 * psi * psi * psi +
                   12 * std::pow(psi, 6) + 162 * std::pow(psi, 5) +
                   1464 * std::pow(psi, 4) + 5617 * psi * psi * psi +
                   8070 * psi * psi + 1721 * psi - 1182;
        };

        // Verify D(psi) > 0 at several sample points across (0, 1)
        for (real psi : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
            REQUIRE(D(psi) > 0.0);
        }

        // Verify D(0) > 0 -- the critical condition
        REQUIRE(D(0.0) > 0.0);

        // Evaluate Floating stencil across the range -- all coefficients bounded
        {
            auto [p, r, t, x] = st.query(bcs::Floating);
            T c(r * t);
            T ex{};

            for (real psi : {0.1, 0.3, 0.5, 0.7, 0.9}) {
                st.nbs(1.0, bcs::Floating, psi, false, c, ex);
                real max_abs = 0.0;
                for (std::size_t i = 0; i < c.size(); ++i) {
                    max_abs = std::max(max_abs, std::abs(c[i]));
                }
                REQUIRE(max_abs < 1e8);
            }
        }

        // Evaluate Dirichlet stencil across the range -- all coefficients bounded
        {
            auto [p, r, t, x] = st.query(bcs::Dirichlet);
            T c(r * t);
            T ex{};

            for (real psi : {0.1, 0.3, 0.5, 0.7, 0.9}) {
                st.nbs(1.0, bcs::Dirichlet, psi, false, c, ex);
                real max_abs = 0.0;
                for (std::size_t i = 0; i < c.size(); ++i) {
                    max_abs = std::max(max_abs, std::abs(c[i]));
                }
                REQUIRE(max_abs < 1e8);
            }
        }
    }

    SECTION("alpha[1] < 197/288 throws")
    {
        // Single alpha (zero-padded to alpha[1]=0) should throw
        std::array<real, 1> short_alpha{0.1};
        REQUIRE_THROWS_AS(stencils::make_E4_1(short_alpha), std::invalid_argument);

        // Explicit alpha[1]=0 should throw
        std::array<real, 2> zero_alpha{0.1, 0.0};
        REQUIRE_THROWS_AS(stencils::make_E4_1(zero_alpha), std::invalid_argument);

        // alpha[1] just below the bound should throw
        std::array<real, 2> below_bound{0.1, 197.0 / 288.0 - 0.001};
        REQUIRE_THROWS_AS(stencils::make_E4_1(below_bound), std::invalid_argument);

        // alpha[1] at the bound should NOT throw
        std::array<real, 2> at_bound{0.1, 197.0 / 288.0};
        REQUIRE_NOTHROW(stencils::make_E4_1(at_bound));

        // alpha[1]=0.7 (test default, above bound) should NOT throw
        std::array<real, 2> above_bound{0.1, 0.7};
        REQUIRE_NOTHROW(stencils::make_E4_1(above_bound));
    }
}
