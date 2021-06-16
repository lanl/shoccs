#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating.hpp>

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

#include "simulation_builder.hpp"
#include "systems/system.hpp"

#include <range/v3/all.hpp>

using namespace ccs;

TEST_CASE("cycle")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {21, 22, 23},
                domain_bounds = {
                    min = {1, 1.1, 0.3},
                    max = {3, 3.3, 2.2}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                ymin = "neumann",
                ymax = "neumann",
                zmax = "dirichlet"
            },
            shapes = {
                {
                    type = "sphere",
                    center = {2.0001, 2.5656565, 1.313131311},
                    radius = 0.25,
                    boundary_condition = "dirichlet"
                }
            },
            scheme = {
                order = 2,
                type = "E2"
            },
            system = {
                type = "heat",
                diffusivity = 1.0
            },
            integrator = {
                type = "rk4",                
            },
            step_controller = {
                max_step = 1,
            },
            manufactured_solution = {
                type = "lua",
                call = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return (time + 
                        x * x * (y + z) + y * y * (x + z) + z * z * (x + y) + 
                        3 * x * y * z + x + y + z)                     
                end,
                ddt = function(time, loc)
                    return 1.0
                end,
                grad = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return 2. * x * (y + z) + y * y + z * z + 3. * y * z + 1,
                            x * x + 2. * y * (x + z) + z * z + 3. * x * z + 1,
                            x * x + y * y + 2. * z * (x + y) + 3. * x * y + 1
                end,
                lap = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return 2. * (y + z) + 2. * (x + z) + 2. * (x + y)
                end,
                div = function(time, loc)
                    return 0.0
                end
            }
        }
    )");

    auto cycle_opt = simulation_cycle::from_lua(lua["simulation"]);
    REQUIRE(!!cycle_opt);

    auto res = cycle_opt->run();
    REQUIRE_THAT(res[0], Catch::Matchers::WithinAbs(0.0, 1e-13));
}
