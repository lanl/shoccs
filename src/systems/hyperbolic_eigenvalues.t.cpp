#include "system.hpp"

#include "fields/field_registry.hpp"
#include "io/logging.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sol/sol.hpp>

using namespace ccs;

// Custom main: Kokkos must be initialized before any test allocates Views.
int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

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

    // Set up registry — hyperbolic_eigenvalues has size {0, 0, ss},
    // so no scalars/vectors are allocated. field_refs keep default slot indices.
    sim_registry reg;
    field_ref u0_ref{0};
    sys.initialize(reg, u0_ref, step);

    // maximum eigenvalue is very close to 0
    auto st = sys.stats(reg, u0_ref, u0_ref, step);
    REQUIRE(st.stats[0] + 1.0 == Catch::Approx(1.0));
}
