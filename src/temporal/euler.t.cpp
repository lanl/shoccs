#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

#include "integrator.hpp"
#include "systems/system.hpp"

#include <range/v3/all.hpp>

using namespace ccs;

TEST_CASE("integrator - euler")
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
                type = "euler",
            },
            step_controller = {
                max_step = 1,
            },
            manufactured_solution = {
                type = "lua",
                call = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    -- return time + x * x * (y + z) + x - 1
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
    using namespace si;

    auto sys_opt = system::from_lua(lua["simulation"]);
    REQUIRE(!!sys_opt);
    auto& sys = *sys_opt;

    auto it_opt = integrator::from_lua(lua["simulation"]);
    REQUIRE(!!it_opt);
    auto& it = *it_opt;

    auto st_opt = step_controller::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);
    auto& step = *st_opt;

    field f{sys(step)};
    sys.update_boundary(f, step);

    const real dt = *sys.timestep_size(f, step);

    field g{sys.size()};
    g = it(sys, f, step, dt);

    step.advance(dt);
    sys.update_boundary(g, step);

    // at this point, all fluid points in g should have a value of m_sol(time, loc)
    auto stats = sys.stats(f, g, step);
    REQUIRE_THAT(stats.stats[0], Catch::Matchers::WithinAbs(0.0, 1e-13));
}
