#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "io/logging.hpp"
#include <sol/sol.hpp>

#include "system.hpp"

using namespace ccs;

TEST_CASE("hyperbolic_eigenvalues")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {21},
                domain_bounds = {1}
            },
            shapes = {
                    {
                        type = "yz_rect",
                        psi = 0.001,
                        normal = 1,
                        boundary_condition = "dirichlet"
                    },
                    {
                        type = "yz_rect",
                        psi = 0.9,
                        normal = -1,
                        boundary_condition = "floating"
                    }
                },
                scheme = {
                    order = 1,
                    type = "E2-poly",
                    floating_alpha = {13/100, 7/50, 3/20, 4/25, 17/100, 9/50},
                    dirichlet_alpha = {3/25, 13/100, 7/50}
                },
            system = {
                type = "eigenvalues",
            },
        }
    )");

    auto sys_opt = system::from_lua(lua["simulation"], logs{});
    REQUIRE(!!sys_opt);
    auto& sys = *sys_opt;
    step_controller step{};

    // Initialize array with system and ensure zero error
    field f{sys(step)};

    // maximum eigenvalue is very close to 0
    auto st = sys.stats(f, f, step);
    REQUIRE(st.stats[0] + 1.0 == Catch::Approx(1.0));
}
