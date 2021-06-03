#include "step_controller.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sol/sol.hpp>

using namespace ccs;

TEST_CASE("default")
{
    auto s = step_controller{};
    REQUIRE(!s);

    int step = s;
    real time = s;

    REQUIRE(step == 0);
    REQUIRE(time == 0.0);
}

TEST_CASE("from_lua")
{
    sol::state lua;
    lua.script(R"(
        simulation = {
            step_controller = {
                max_step = 2,
                max_time = 4.0,
                min_dt = 0.1,
                cfl = {
                    hyperbolic = 0.5,
                    parabolic = 0.2
                }
            }
        }
    )");

    auto step_opt = step_controller::from_lua(lua["simulation"]);
    REQUIRE(step_opt);
    auto& step = *step_opt;

    REQUIRE(step);
    REQUIRE((int)step == 0);
    REQUIRE((real)step == 0.0);

    {
        auto dt_opt = step.check_timestep_size(0.01);
        REQUIRE(!dt_opt);
    }

    auto dt_opt = step.check_timestep_size(0.2);
    REQUIRE(dt_opt);

    step.advance(*dt_opt);
    REQUIRE((int)step == 1);
    REQUIRE((real)step == *dt_opt);
    REQUIRE(step);
    step.advance(*dt_opt);
    REQUIRE((int)step == 2);
    REQUIRE((real)step == 0.4);
    REQUIRE(!step);
}
