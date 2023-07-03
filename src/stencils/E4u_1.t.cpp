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

TEST_CASE("E4u_1")
{
    using T = std::vector<real>;
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            scheme = {
                order = 1,
                type = "E4u",
                alpha = {-0.7733323791884821, 0.1623961700641681}
            }
        }
    )");

    auto st_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);
    const auto& st = *st_opt;

    {

        auto [p, r, t, x] = st.query(bcs::Floating);
        REQUIRE(p == 2);
        REQUIRE(r == 3);
        REQUIRE(t == 5);
        REQUIRE(x == 0);

        T c(r * t);
        T ex{};

        st.nbs(2, bcs::Floating, 1.0, false, c, ex);
        REQUIRE_THAT(c,
                     Approx(T{-1.3033328562609077,
                              3.046664758376964,
                              -3.069997137565446,
                              1.713331425043631,
                              -0.38666618959424104,
                              -0.08546858163458262,
                              -0.5747923401283361,
                              0.9871885101925043,
                              -0.4081256734616695,
                              0.08119808503208405,
                              0.0923093909615862,
                              -0.5359042305130115,
                              0.3038563457695172,
                              0.13076243615365518,
                              0.00897605762825287})
                         .margin(1.0e-8));
    }
    {
        auto [p, r, t, x] = st.query(bcs::Dirichlet);
        REQUIRE(p == 2);
        REQUIRE(r == 2);
        REQUIRE(t == 5);
        REQUIRE(x == 0);

        T c(r * t);
        T ex{};

        st.nbs(0.5, bcs::Dirichlet, 0.0, false, c, ex);
        REQUIRE_THAT(c,
                     Approx(T{-0.3418743265383305,
                              -2.2991693605133445,
                              3.9487540407700172,
                              -1.632502693846678,
                              0.3247923401283362,
                              0.3692375638463448,
                              -2.143616922052046,
                              1.2154253830780688,
                              0.5230497446146207,
                              0.03590423051301148})
                         .margin(1.0e-8));
    }
}
