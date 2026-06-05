#include <Kokkos_Core.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

#include "simulation_builder.hpp"
#include "systems/system.hpp"

using namespace ccs;

// Custom main: Kokkos must be initialized before any test allocates Views.
int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

TEST_CASE("cycle - 2D")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {21, 22},
                domain_bounds = {
                    min = {1, 1.1},
                    max = {3, 3.3}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                ymin = "neumann",
                ymax = "neumann",
            },
            shapes = {
                {
                    type = "sphere",
                    center = {2.0001, 2.5656565},
                    radius = 0.25,
                    boundary_condition = "floating"
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
                max_step = 5,
            },
            manufactured_solution = {
                type = "lua",
                call = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return (time +
                        x * x * y + y * y * x + 3 * x * y + x + y)
                end,
                ddt = function(time, loc)
                    return 1.0
                end,
                grad = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return 2. * x * y + y * y + 3. * y + 1,
                            x * x + 2. * y * x + 3. * x + 1,
                            0
                end,
                lap = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return 2. * y + 2. * x
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
    // res = {time, Linf_error, Linf_error}
    // res[0] is the final simulation time; res[1] is the L∞ error.
    // For a 2nd-order scheme on a coarse 21×22 grid over 5 steps the spatial
    // discretisation error dominates — verify it stays bounded.
    REQUIRE_THAT(res[0], Catch::Matchers::WithinAbs(0.0125, 1e-10));
    REQUIRE(res[1] < 0.05);
}

TEST_CASE("cycle - 2D euler")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {21, 22},
                domain_bounds = {
                    min = {1, 1.1},
                    max = {3, 3.3}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                ymin = "neumann",
                ymax = "neumann",
            },
            shapes = {
                {
                    type = "sphere",
                    center = {2.0001, 2.5656565},
                    radius = 0.25,
                    boundary_condition = "floating"
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
                type = "euler",
            },
            step_controller = {
                max_step = 5,
            },
            manufactured_solution = {
                type = "lua",
                call = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return (time +
                        x * x * y + y * y * x + 3 * x * y + x + y)
                end,
                ddt = function(time, loc)
                    return 1.0
                end,
                grad = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return 2. * x * y + y * y + 3. * y + 1,
                            x * x + 2. * y * x + 3. * x + 1,
                            0
                end,
                lap = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return 2. * y + 2. * x
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
    // Euler is first-order in time; for this polynomial MMS the spatial
    // discretisation error dominates at this grid resolution.  Use the same
    // tolerance as the RK4 2D test.
    REQUIRE(res[1] < 0.05);
}
