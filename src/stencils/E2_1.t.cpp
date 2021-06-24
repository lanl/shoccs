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

TEST_CASE("E2_1")
{
    using T = std::vector<real>;
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            scheme = {
                order = 1,
                type = "E2",
                alpha = {1, 2, 3, -1}
            }
        }
    )");

    auto st_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);
    const auto& st = *st_opt;

    {

        auto [p, r, t, x] = st.query(bcs::Floating);
        REQUIRE(p == 1);
        REQUIRE(r == 4);
        REQUIRE(t == 5);
        REQUIRE(x == 0);

        T c(r * t);
        T ex{};

        st.nbs(2, bcs::Floating, 1.0, false, c, ex);
        REQUIRE_THAT(c,
                     Approx(T{3,
                              -5,
                              0.5,
                              1.5,
                              0,
                              -0.5,
                              0,
                              1,
                              -0.5,
                              0,
                              0.02631578947368421,
                              -0.32894736842105265,
                              0.07894736842105263,
                              0.2236842105263158,
                              0,
                              0,
                              0,
                              -0.25,
                              0,
                              0.25}));
    }
    {
        auto [p, r, t, x] = st.query(bcs::Dirichlet);
        REQUIRE(p == 1);
        REQUIRE(r == 3);
        REQUIRE(t == 5);
        REQUIRE(x == 0);

        T c(r * t);
        T ex{};

        st.nbs(0.5, bcs::Dirichlet, 0.0, true, c, ex);
        REQUIRE_THAT(c,
                     Approx(T{-0.8947368421052632,
                              -0.3157894736842105,
                              1.3157894736842106,
                              -0.10526315789473684,
                              0.,
                              2.,
                              -4.,
                              0.,
                              2.,
                              0.,
                              -6.,
                              -2.,
                              20.,
                              0.,
                              -12.}));
    }
}
