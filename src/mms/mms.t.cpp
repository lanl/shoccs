#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fields/tuple_utils.hpp"
#include "manufactured_solutions.hpp"
#include "std_matchers.hpp"

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>
#include <string>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/single.hpp>

using namespace ccs;
using Catch::Matchers::Approx;

TEST_CASE("gauss1d")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
            simulation = {
                manufactured_solution = {
                        type = "gaussian",
                        
                        {
                                center = {1},
                                variance = {0.5},
                                amplitude = 2,
                                frequency = 0.1
                        },
                        {
                                center = {2},
                                variance = {0.3},
                                amplitude = 1.2,
                                frequency = 0.2
                        }
                }
            }
        )");

    auto ms_opt = manufactured_solution::from_lua(lua["simulation"], 1);
    REQUIRE(!!ms_opt);
    auto& ms = *ms_opt;

    const real3 loc{3.0, 0.0, 0.0};
    const real time = 8.0;

    REQUIRE(ms(time, loc) == Catch::Approx(0.0003319785015967778));

    auto t = vs::single(loc) | ms(time) | rs::to<std::vector<real>>();
    REQUIRE(t[0] == Catch::Approx(0.0003319785015967778));

    REQUIRE(ms.ddt(time, loc) == Catch::Approx(-0.000975554445371058));

    const real3 grad{-0.0022343980664847485, 0, 0};
    const auto ms_grad = ms.gradient(time, loc);

    REQUIRE_THAT(ms_grad, Approx(grad));
    REQUIRE(ms.laplacian(time, loc) == Catch::Approx(0.01282798401538059));
}

TEST_CASE("gauss2d")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
            simulation = {
                manufactured_solution = {
                        type = "gaussian",
                        
                        {
                                center = {1, 1.2},
                                variance = {0.5, 0.8},
                                amplitude = 2,
                                frequency = 0.1
                        },
                        {
                                center = {2, -1},
                                variance = {0.3, 0.6},
                                amplitude = 1.2,
                                frequency = 0.2
                        }
                }
            }
        )");
    auto ms_opt = manufactured_solution::from_lua(lua["simulation"], 2);
    REQUIRE(!!ms_opt);
    auto& ms = *ms_opt;

    const real3 loc{3.0, -0.5, 0.0};
    const real time = 8.0;

    REQUIRE(ms(time, loc) == Catch::Approx(-0.000046838098638663583));

    REQUIRE(ms.ddt(time, loc) == Catch::Approx(-0.0006603967369586969));

    const real3 grad{0.0006725075348924356, 0.0002627963438362986, 0};
    // const auto ms_grad = ms.gradient(time, loc);
    REQUIRE_THAT(ms.gradient(time, loc), Approx(grad));
    REQUIRE(ms.laplacian(time, loc) == Catch::Approx(-0.007471160503672486));
}

TEST_CASE("gauss3d")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
            simulation = {
                manufactured_solution = {
                        type = "gaussian",
                        
                        {
                                center = {1, 1.2, -3.5},
                                variance = {0.5, 0.8, 2.0},
                                amplitude = 2,
                                frequency = 0.1
                        },
                        {
                                center = {2, -1},
                                variance = {0.3, 0.6, 0.1},
                                amplitude = 1.2,
                                frequency = 0.2
                        }
                }
            }
        )");
    auto ms_opt = manufactured_solution::from_lua(lua["simulation"], 3);
    REQUIRE(!!ms_opt);
    auto& ms = *ms_opt;

    const real3 loc{3.0, -0.5, -2.0};
    const real time = 8.0;

    REQUIRE(ms(time, loc) == Catch::Approx(0.00003689973951150938));

    REQUIRE(ms.ddt(time, loc) == Catch::Approx(-3.7993394546164832e-6));

    const real3 grad{
        -0.00029519791609207505, 0.00009801493307744677, -0.000013837402316816019};
    // const auto ms_grad = ms->gradient(time, loc);
    REQUIRE_THAT(ms.gradient(time, loc), Approx(grad));
    REQUIRE(ms.laplacian(time, loc) == Catch::Approx(0.002412644784681726));
}

TEST_CASE("lua")
{

    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
            simulation = {
                manufactured_solution = {
                        type = "lua",
                        
                        call = function (time, loc)
                                return loc[3] * time + math.sin(loc[1]) * math.cos(loc[2])
                        end,

                        ddt = function (time, loc)
                                return loc[3]
                        end,

                        grad = function (time, loc)
                                return  math.sin(loc[1]*loc[2]),
                                        math.cos(loc[1]*loc[3]),
                                        math.sin(loc[2]*loc[3])
                                
                        end,

                        div = function (time, loc)
                                return loc[1] * loc[2] * loc[3]
                        end,

                        lap = function (time, loc)
                                return loc[1] + loc[2] + loc[3]
                        end                       
                }
            }
        )");
    auto ms_opt = manufactured_solution::from_lua(lua["simulation"]);
    REQUIRE(!!ms_opt);
    auto& ms = *ms_opt;
    REQUIRE(ms);

    const real3 loc{3.0, -0.5, -2.0};
    const real time = 8.0;

    sol::table t = lua["simulation"]["manufactured_solution"];
    sol::protected_function call = t["call"];

    REQUIRE(ms(time, loc) == Catch::Approx(call(time, loc)));

    sol::protected_function ddt = t["ddt"];
    REQUIRE(ms.ddt(time, loc) == Catch::Approx(ddt(time, loc)));

    sol::protected_function grad = t["grad"];
    std::tuple<real, real, real> res = grad(time, loc);
    real3 g = to<real3>(res);
    REQUIRE_THAT(ms.gradient(time, loc), Approx(g));
    REQUIRE(ms.laplacian(time, loc) == Catch::Approx(t["lap"](time, loc)));
}
