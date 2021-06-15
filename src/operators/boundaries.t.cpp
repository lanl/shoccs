#include "boundaries.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

using namespace ccs;

TEST_CASE("from_lua")
{

    sol::state lua;
    lua.script(R"(
        simulation = {
            domain_boundaries = {
                xmin = "dirichlet",
                zmax = "neumann"
            },
            shapes = {
                {
                    boundary_condition = "dirichlet",                    
                },
                {
                    type = "sphere"
                }
            }
        }
    )");

    auto bc_opt = bcs::from_lua(lua["simulation"], index_extents{.extents = {2, 2, 2}});
    REQUIRE(!!bc_opt);

    auto&& [grid_bcs, object_bcs] = *bc_opt;

    REQUIRE(grid_bcs == bcs::Grid{bcs::df, bcs::ff, bcs::fn});
    REQUIRE(object_bcs == bcs::Object{bcs::Dirichlet, bcs::Floating});
}
