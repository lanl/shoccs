#include "system.hpp"

#include "fields/field_registry.hpp"
#include "fields/selection_desc.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <sol/sol.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

using namespace ccs;

// Custom main: Kokkos must be initialized before any test allocates Views.
int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

// Helper: set up a sim_registry with 2 slots (u0 and rhs) from system size.
static std::pair<field_ref, field_ref> setup_registry(sim_registry& reg, ccs::system& sys)
{
    auto sz = sys.size();
    int d_sz  = sz.d_size;
    int rx_sz = sz.rx_size;
    int ry_sz = sz.ry_size;
    int rz_sz = sz.rz_size;

    field_ref u0_ref{0}, rhs_ref{1};
    for (int s = 0; s < sz.nscalars; ++s) {
        u0_ref  = reg.allocate_scalar(0, s, d_sz, rx_sz, ry_sz, rz_sz);
        rhs_ref = reg.allocate_scalar(1, s, d_sz, rx_sz, ry_sz, rz_sz);
    }
    return {u0_ref, rhs_ref};
}

TEST_CASE("scalar_wave - update_boundary")
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
                type = "scalar wave"
            }
        }
    )");

    auto sys_opt = system::from_lua(lua["simulation"]);
    REQUIRE(!!sys_opt);
    auto& sys = *sys_opt;
    step_controller step{};

    // Set up registry and initialize
    sim_registry reg;
    auto [u0_ref, rhs_ref] = setup_registry(reg, sys);
    sys.initialize(reg, u0_ref, step);

    // After initialize at t=0, maximum error should be zero
    auto st = sys.stats(reg, u0_ref, u0_ref, step);
    REQUIRE_THAT(st.stats[0], Catch::Matchers::WithinAbs(0.0, 1e-13));

    // Call update_boundary at t=0.25 — boundary values should change from
    // the t=0 values written by initialize.  Directly verify that D-buffer
    // entries on Dirichlet faces match the expected manufactured solution.
    constexpr real time = 0.25;
    sys.update_boundary(reg, u0_ref, time);

    // Mesh / solution parameters (must match the Lua config above)
    constexpr int nx = 21, ny = 22, nz = 23;
    const index_extents ext{{nx, ny, nz}};
    constexpr real xmin_v = 1.0, ymin_v = 1.1, zmin_v = 0.3;
    constexpr real xmax_v = 3.0, ymax_v = 3.3, zmax_v = 2.2;
    constexpr real dx = (xmax_v - xmin_v) / (nx - 1);
    constexpr real dy = (ymax_v - ymin_v) / (ny - 1);
    constexpr real dz = (zmax_v - zmin_v) / (nz - 1);
    constexpr real3 center{2.0001, 2.5656565, 1.313131311};
    constexpr real rad = 0.25;
    constexpr real twoPI = 2 * std::numbers::pi_v<real>;

    auto expected_sol = [&](real x, real y, real z) {
        real rx = x - center[0], ry = y - center[1], rz = z - center[2];
        real r = std::sqrt(rx * rx + ry * ry + rz * rz);
        return std::sin(twoPI * (r - rad - time));
    };

    constexpr auto sh = scalar_handle{0};
    const real* u_D = reg.data(u0_ref, sh.D());

    // xmin face (x = 0, Dirichlet)
    {
        auto desc = make_x_plane_desc(ext, 0);
        REQUIRE(desc.count() == ny * nz);
        for (int n = 0; n < desc.count(); ++n) {
            int flat = desc.element(n);
            int j = (flat / nz) % ny;
            int k = flat % nz;
            real expected = expected_sol(xmin_v, ymin_v + j * dy, zmin_v + k * dz);
            REQUIRE(u_D[flat] == Catch::Approx(expected).epsilon(1e-10));
        }
    }

    // zmax face (z = nz-1, Dirichlet)
    {
        auto desc = make_z_plane_desc(ext, nz - 1);
        REQUIRE(desc.count() == nx * ny);
        for (int n = 0; n < desc.count(); ++n) {
            int flat = desc.element(n);
            int i = flat / (ny * nz);
            int j = (flat / nz) % ny;
            real expected = expected_sol(xmin_v + i * dx, ymin_v + j * dy, zmax_v);
            REQUIRE(u_D[flat] == Catch::Approx(expected).epsilon(1e-10));
        }
    }

    // Exercise the full pipeline: update_boundary → rhs
    sys.rhs(reg, u0_ref, reg, rhs_ref, time);

    // Verify rhs produces finite values (no NaN/Inf from bad boundary data)
    auto u_rhs = extract_scalar_span(reg, rhs_ref, sh);
    bool all_finite =
        std::ranges::all_of(u_rhs.D, [](real v) { return std::isfinite(v); });
    REQUIRE(all_finite);
}

// Uses the E2 setup with Dirichlet + Neumann grid BCs and a Dirichlet object,
// exercising all graph branches: gradient, dot product with wave speed coefficients.
TEST_CASE("scalar_wave - graph matches eager")
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
                type = "scalar wave"
            }
        }
    )");

    auto sw_opt = systems::scalar_wave::from_lua(lua["simulation"]);
    REQUIRE(!!sw_opt);
    auto& sw = *sw_opt;
    step_controller step{};

    auto sz = sw.size();
    sim_registry reg;

    // Allocate 3 slots: u0 (input), rhs_eager, rhs_graph
    field_ref u0_ref{0}, rhs_ref{1}, rhs2_ref{2};
    for (int s = 0; s < sz.nscalars; ++s) {
        u0_ref = reg.allocate_scalar(
            0, s, sz.d_size, sz.rx_size, sz.ry_size, sz.rz_size);
        rhs_ref = reg.allocate_scalar(
            1, s, sz.d_size, sz.rx_size, sz.ry_size, sz.rz_size);
        rhs2_ref = reg.allocate_scalar(
            2, s, sz.d_size, sz.rx_size, sz.ry_size, sz.rz_size);
    }

    sw.initialize(reg, u0_ref, step);
    sw.update_boundary(reg, u0_ref, (real)step);

    // Eager path
    sw.rhs(reg, u0_ref, reg, rhs_ref, (real)step);
    constexpr auto sh = scalar_handle{0};
    auto eager = extract_scalar_span(reg, rhs_ref, sh);

    // Save eager results
    std::vector<real> exp_d(eager.D.begin(), eager.D.end());
    std::vector<real> exp_rx(eager.Rx.begin(), eager.Rx.end());
    std::vector<real> exp_ry(eager.Ry.begin(), eager.Ry.end());
    std::vector<real> exp_rz(eager.Rz.begin(), eager.Rz.end());

    // Graph path
    auto u = extract_scalar_view(reg, u0_ref, sh);
    auto du = extract_scalar_span(reg, rhs2_ref, sh);

    sw.build_rhs_graph(u, du);
    sw.submit_rhs_graph();

    // Compare all 4 buffers
    for (int i = 0; i < sz.d_size; ++i) {
        INFO("D[" << i << "]");
        REQUIRE(du.D[i] == Catch::Approx(exp_d[i]));
    }
    for (int i = 0; i < sz.rx_size; ++i) {
        INFO("Rx[" << i << "]");
        REQUIRE(du.Rx[i] == Catch::Approx(exp_rx[i]));
    }
    for (int i = 0; i < sz.ry_size; ++i) {
        INFO("Ry[" << i << "]");
        REQUIRE(du.Ry[i] == Catch::Approx(exp_ry[i]));
    }
    for (int i = 0; i < sz.rz_size; ++i) {
        INFO("Rz[" << i << "]");
        REQUIRE(du.Rz[i] == Catch::Approx(exp_rz[i]));
    }

    SECTION("resubmit produces same result")
    {
        // Zero graph output, resubmit
        std::ranges::fill(du.D, 0.0);
        std::ranges::fill(du.Rx, 0.0);
        std::ranges::fill(du.Ry, 0.0);
        std::ranges::fill(du.Rz, 0.0);

        sw.submit_rhs_graph();

        for (int i = 0; i < sz.d_size; ++i) {
            INFO("D[" << i << "] resubmit");
            REQUIRE(du.D[i] == Catch::Approx(exp_d[i]));
        }
        for (int i = 0; i < sz.rx_size; ++i) {
            INFO("Rx[" << i << "] resubmit");
            REQUIRE(du.Rx[i] == Catch::Approx(exp_rx[i]));
        }
    }
}
