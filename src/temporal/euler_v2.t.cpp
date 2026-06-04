#include <Kokkos_Core.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sol/sol.hpp>

#include "integrator.hpp"
#include "systems/system.hpp"

using namespace ccs;

// ---------------------------------------------------------------------------
// Custom main: Kokkos must be initialized before any test allocates Views.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

// ---------------------------------------------------------------------------
// Registry-based euler integration test using the heat system.
// Mirrors the existing euler.t.cpp but uses sim_registry + field_ref.
// ---------------------------------------------------------------------------

TEST_CASE("euler registry-based step")
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

    auto sys_opt = system::from_lua(lua["simulation"]);
    REQUIRE(!!sys_opt);
    auto& sys = *sys_opt;

    auto st_opt = step_controller::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);
    auto& step = *st_opt;

    // Set up registry with 3 slots: u0(0), u1(1), system_rhs(2)
    sim_registry reg;
    auto sz = sys.size();

    // Extract scalar sizes from system_size
    int d_sz  = sz.d_size;
    int rx_sz = sz.rx_size;
    int ry_sz = sz.ry_size;
    int rz_sz = sz.rz_size;

    // Allocate slots with matching scalar layout
    field_ref u0_ref{0}, u1_ref{1}, srhs_ref{2};
    for (int s = 0; s < sz.nscalars; ++s) {
        u0_ref   = reg.allocate_scalar(0, s, d_sz, rx_sz, ry_sz, rz_sz);
        u1_ref   = reg.allocate_scalar(1, s, d_sz, rx_sz, ry_sz, rz_sz);
        srhs_ref = reg.allocate_scalar(2, s, d_sz, rx_sz, ry_sz, rz_sz);
    }

    // Initialize u0 with the system's initial condition
    sys.initialize(reg, u0_ref, step);
    sys.update_boundary(reg, u0_ref, step);

    // Get timestep
    const real dt = *sys.timestep_size(reg, u0_ref, step);

    // Build RHS graph: bind to (u1_ref, srhs_ref) matching the production
    // convention in simulation_cycle. Euler deep-copies u0→u1 before submit.
    sys.build_rhs_graph(reg, u1_ref, reg, srhs_ref);

    // Perform one euler step using the registry-based interface
    integrators::euler euler_integrator;
    euler_integrator(sys, reg, u0_ref, u1_ref, srhs_ref, step, dt);

    step.advance(dt);
    sys.update_boundary(reg, u1_ref, step);

    // At this point, all fluid points in u1 should match the manufactured solution
    auto stats = sys.stats(reg, u0_ref, u1_ref, step);
    REQUIRE_THAT(stats.stats[0], Catch::Matchers::WithinAbs(0.0, 1e-13));
}
