#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

#include "system.hpp"

#include <range/v3/all.hpp>

using namespace ccs;

TEST_CASE("heat - E2")
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
                diffusivity = 0.1
            },
            manufactured_solution = {
                type = "lua",
                call = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return (math.sin(time) + 
                        x * x * (y + z) + y * y * (x + z) + z * z * (x + y) + 
                        3 * x * y * z + x + y + z)                     
                end,
                ddt = function(time, loc)
                    return math.cos(time)
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

    auto sys_opt = system::from_lua(lua["simulation"]);
    REQUIRE(!!sys_opt);
    auto& sys = *sys_opt;
    step_controller step{};

    // Initialize array with system and ensure zero error
    field f{sys(step)};
    // only solid points will be zero
    const integer solid_points = rs::count(f.scalars(0) | sel::D, 0.0);
    // maximum error should be zero
    auto st = sys.stats(f, f, step);
    REQUIRE(st.stats[0] == 0);

    // prepare for rhs calculation
    sys.update_boundary(f, step);
    field rhs{sys.size()};
    rhs = sys.rhs(f, step);

    auto&& u_rhs = rhs.scalars(0);

    // at this point, all fluid points in rhs should have a value of cos(time) -> 1
    // and solid points should remain at zero
    const integer rhs_solid_points = rs::count(u_rhs | sel::D, 0.0);
    // these zeros include the zeroed rhs contributions to the dirichlet planar bcs
    int3 n{21, 22, 23};
    integer x_sz = n[1] * n[2];
    integer z_sz = n[0] * n[1];
    REQUIRE(solid_points == rhs_solid_points - (x_sz + z_sz - n[1]));

    real sum = rs::accumulate(u_rhs | sel::D, 0.0);

    REQUIRE(sum == Catch::Approx((real)n[0] * n[1] * n[2] - rhs_solid_points));
}
