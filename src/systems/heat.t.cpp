#include "system.hpp"

#include "fields/field_registry.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <numeric>

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

    // Set up registry and initialize
    sim_registry reg;
    auto [u0_ref, rhs_ref] = setup_registry(reg, sys);
    sys.initialize(reg, u0_ref, step);

    constexpr auto sh = scalar_handle{0};
    auto u0_scalar = extract_scalar_span(reg, u0_ref, sh);

    // only solid points will be zero
    const integer solid_points = std::ranges::count(u0_scalar.D, 0.0);
    // maximum error should be zero
    auto st = sys.stats(reg, u0_ref, u0_ref, step);
    REQUIRE_THAT(st.stats[0], Catch::Matchers::WithinAbs(0.0, 1e-13));

    // prepare for rhs calculation
    sys.update_boundary(reg, u0_ref, (real)step);
    sys.rhs(reg, u0_ref, reg, rhs_ref, (real)step);

    auto u_rhs = extract_scalar_span(reg, rhs_ref, sh);

    // at this point, all fluid points in rhs should have a value of cos(time) -> 1
    // and solid points should remain at zero
    const integer rhs_solid_points = std::ranges::count(u_rhs.D, 0.0);
    // these zeros include the zeroed rhs contributions to the dirichlet planar bcs
    int3 n{21, 22, 23};
    integer x_sz = n[1] * n[2];
    integer z_sz = n[0] * n[1];
    REQUIRE(solid_points == rhs_solid_points - (x_sz + z_sz - n[1]));

    auto d_rng = u_rhs.D;
    real sum = std::accumulate(std::ranges::begin(d_rng), std::ranges::end(d_rng), 0.0);

    REQUIRE(sum == Catch::Approx((real)n[0] * n[1] * n[2] - rhs_solid_points));
}

TEST_CASE("heat - E2 - floating")
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
                    boundary_condition = "floating"
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
                        x * (y + z) + y * (x + z) + z * (x + y) +
                        x + y + z)
                end,
                ddt = function(time, loc)
                    return math.cos(time)
                end,
                grad = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return 2. * (y + z) + 1,
                           2. * (x + z) + 1,
                           2. * (x + y) + 1
                end,
                lap = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return 0.0
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

    // Set up registry and initialize
    sim_registry reg;
    auto [u0_ref, rhs_ref] = setup_registry(reg, sys);
    sys.initialize(reg, u0_ref, step);

    constexpr auto sh = scalar_handle{0};
    auto u0_scalar = extract_scalar_span(reg, u0_ref, sh);

    // only solid points will be zero
    const integer solid_points = std::ranges::count(u0_scalar.D, 0.0);
    // maximum error should be zero
    auto st = sys.stats(reg, u0_ref, u0_ref, step);
    REQUIRE_THAT(st.stats[0], Catch::Matchers::WithinAbs(0.0, 1e-13));

    // prepare for rhs calculation
    sys.update_boundary(reg, u0_ref, (real)step);
    sys.rhs(reg, u0_ref, reg, rhs_ref, (real)step);

    auto u_rhs = extract_scalar_span(reg, rhs_ref, sh);

    // at this point, all fluid points in rhs should have a value of cos(time) -> 1
    // and solid points should remain at zero
    const integer rhs_solid_points = std::ranges::count(u_rhs.D, 0.0);
    // these zeros include the zeroed rhs contributions to the dirichlet planar bcs
    int3 n{21, 22, 23};
    integer x_sz = n[1] * n[2];
    integer z_sz = n[0] * n[1];
    REQUIRE(solid_points == rhs_solid_points - (x_sz + z_sz - n[1]));

    real sum_d  = std::accumulate(u_rhs.D.begin(),  u_rhs.D.end(),  0.0);
    real sum_rx = std::accumulate(u_rhs.Rx.begin(), u_rhs.Rx.end(), 0.0);
    real sum_ry = std::accumulate(u_rhs.Ry.begin(), u_rhs.Ry.end(), 0.0);
    real sum_rz = std::accumulate(u_rhs.Rz.begin(), u_rhs.Rz.end(), 0.0);

    REQUIRE(sum_d ==
            Catch::Approx((real)n[0] * n[1] * n[2] - rhs_solid_points));

    auto sz2 = sys.size();

    REQUIRE(sum_rx == Catch::Approx((real)sz2.rx_size));
    REQUIRE(sum_ry == Catch::Approx((real)sz2.ry_size));
    REQUIRE(sum_rz == Catch::Approx((real)sz2.rz_size));
}

TEST_CASE("2D heat - E2 - floating")
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
                diffusivity = 0.1
            },
            manufactured_solution = {
                type = "lua",
                call = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return (math.sin(time) + x * y + x + y)
                end,
                ddt = function(time, loc)
                    return math.cos(time)
                end,
                grad = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return y + 1, x + 1, 0.0
                end,
                lap = function(time, loc)
                    local x, y, z = loc[1], loc[2], loc[3]
                    return 0.0
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

    // Set up registry and initialize
    sim_registry reg;
    auto [u0_ref, rhs_ref] = setup_registry(reg, sys);
    sys.initialize(reg, u0_ref, step);

    constexpr auto sh = scalar_handle{0};
    auto u0_scalar = extract_scalar_span(reg, u0_ref, sh);

    // only solid points will be zero
    const integer solid_points = std::ranges::count(u0_scalar.D, 0.0);
    // maximum error should be zero
    auto st = sys.stats(reg, u0_ref, u0_ref, step);
    REQUIRE_THAT(st.stats[0], Catch::Matchers::WithinAbs(0.0, 1e-13));

    // prepare for rhs calculation
    sys.update_boundary(reg, u0_ref, (real)step);
    sys.rhs(reg, u0_ref, reg, rhs_ref, (real)step);

    auto u_rhs = extract_scalar_span(reg, rhs_ref, sh);

    // at this point, all fluid points in rhs should have a value of cos(time) -> 1
    // and solid points should remain at zero
    const integer rhs_solid_points = std::ranges::count(u_rhs.D, 0.0);
    // these zeros include the zeroed rhs contributions to the dirichlet planar bcs
    int3 n{21, 22, 1};
    integer x_sz = n[1];
    REQUIRE(solid_points == rhs_solid_points - x_sz);

    real sum_d  = std::accumulate(u_rhs.D.begin(),  u_rhs.D.end(),  0.0);
    real sum_rx = std::accumulate(u_rhs.Rx.begin(), u_rhs.Rx.end(), 0.0);
    real sum_ry = std::accumulate(u_rhs.Ry.begin(), u_rhs.Ry.end(), 0.0);

    REQUIRE(sum_d ==
            Catch::Approx((real)n[0] * n[1] * n[2] - rhs_solid_points));

    auto sz2 = sys.size();

    REQUIRE(sum_rx == Catch::Approx((real)sz2.rx_size));
    REQUIRE(sum_ry == Catch::Approx((real)sz2.ry_size));
}

// 16.1a: Verify eval_at_locations fills D and R buffers with correct point-wise
// values. Uses a simple manufactured solution f(t,x,y,z) = x + y + z (linear,
// so laplacian=0 and ddt=0). After initialize() at t=0, D at fluid indices and
// all R buffers should match the expected values exactly.
TEST_CASE("heat - eval_at_locations correctness")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {8, 9, 10},
                domain_bounds = {
                    min = {0.0, 0.0, 0.0},
                    max = {1.0, 1.0, 1.0}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                xmax = "dirichlet",
            },
            shapes = {
                {
                    type = "sphere",
                    center = {0.5, 0.5, 0.5},
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
            manufactured_solution = {
                type = "lua",
                call = function(time, loc)
                    return loc[1] + loc[2] + loc[3]
                end,
                ddt = function(time, loc)
                    return 0.0
                end,
                grad = function(time, loc)
                    return 1.0, 1.0, 1.0
                end,
                lap = function(time, loc)
                    return 0.0
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

    sim_registry reg;
    auto [u0_ref, rhs_ref] = setup_registry(reg, sys);
    sys.initialize(reg, u0_ref, step);

    // Stats error at t=0 must be zero (aggregate check)
    auto st = sys.stats(reg, u0_ref, u0_ref, step);
    REQUIRE_THAT(st.stats[0], Catch::Matchers::WithinAbs(0.0, 1e-13));

    // Verify individual D buffer values at fluid indices.
    // Mesh coords: x_i = i * dx, y_j = j * dy, z_k = k * dz
    // D layout: flat index = i * ny * nz + j * nz + k
    constexpr int nx = 8, ny = 9, nz = 10;
    constexpr real dx = 1.0 / (nx - 1);
    constexpr real dy = 1.0 / (ny - 1);
    constexpr real dz = 1.0 / (nz - 1);

    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_view(reg, u0_ref, sh);

    // Spot-check several D-buffer interior points.
    // Solid points (inside sphere) will be zero; fluid points should equal x+y+z.
    // We check a few known-fluid points (corners of the domain are always fluid
    // since the sphere center is at 0.5,0.5,0.5 with radius 0.25).
    auto check_d = [&](int i, int j, int k) {
        int flat = i * ny * nz + j * nz + k;
        real x = i * dx, y = j * dy, z = k * dz;
        real expected = x + y + z;
        INFO("D[" << i << "," << j << "," << k << "] flat=" << flat);
        REQUIRE(u.D[flat] == Catch::Approx(expected).epsilon(1e-12));
    };

    // Corner points — guaranteed to be far from sphere and thus fluid
    check_d(0, 0, 0);                 // (0,0,0)   -> 0
    check_d(nx - 1, ny - 1, nz - 1);  // (1,1,1)   -> 3
    check_d(0, 0, nz - 1);            // (0,0,1)   -> 1
    check_d(nx - 1, 0, 0);            // (1,0,0)   -> 1
    check_d(0, ny - 1, 0);            // (0,1,0)   -> 1

    // Verify all Rx/Ry/Rz values match f(pos) = pos.x + pos.y + pos.z.
    // For floating BC, all R points are initialized by eval_at_locations.
    auto sz = sys.size();

    // R buffers are populated from mesh_object_info::position.
    // We can't access the mesh directly, but we can verify consistency:
    // each R value should be in range [0, 3] (since domain is [0,1]^3, max x+y+z=3)
    // and stats error is zero confirms they all match.
    // Additionally verify every R value is finite and in expected range.
    for (int i = 0; i < sz.rx_size; ++i) {
        INFO("Rx[" << i << "]");
        REQUIRE(std::isfinite(u.Rx[i]));
        REQUIRE(u.Rx[i] >= -0.1);
        REQUIRE(u.Rx[i] <= 3.1);
    }
    for (int i = 0; i < sz.ry_size; ++i) {
        INFO("Ry[" << i << "]");
        REQUIRE(std::isfinite(u.Ry[i]));
        REQUIRE(u.Ry[i] >= -0.1);
        REQUIRE(u.Ry[i] <= 3.1);
    }
    for (int i = 0; i < sz.rz_size; ++i) {
        INFO("Rz[" << i << "]");
        REQUIRE(std::isfinite(u.Rz[i]));
        REQUIRE(u.Rz[i] >= -0.1);
        REQUIRE(u.Rz[i] <= 3.1);
    }
}

// 16.3a: Verify stats() returns correct min, max, and per-component errors.
// After initialize at t=0 (u = sol, error = 0), we perturb known fluid points
// and verify that stats() reports the correct error magnitudes, indices, and
// min/max values. This test locks in correctness before stats() is migrated
// to Kokkos::parallel_reduce.
TEST_CASE("heat - stats reduction correctness")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {8, 9, 10},
                domain_bounds = {
                    min = {0.0, 0.0, 0.0},
                    max = {1.0, 1.0, 1.0}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                xmax = "dirichlet",
            },
            shapes = {
                {
                    type = "sphere",
                    center = {0.5, 0.5, 0.5},
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
            manufactured_solution = {
                type = "gaussian",
                {
                    center = {0.0, 0.0, 0.0},
                    variance = {1.0, 1.0, 1.0},
                    amplitude = 1.0,
                    frequency = 0.1
                }
            }
        }
    )");

    auto sys_opt = system::from_lua(lua["simulation"]);
    REQUIRE(!!sys_opt);
    auto& sys = *sys_opt;
    step_controller step{};

    sim_registry reg;
    auto [u0_ref, rhs_ref] = setup_registry(reg, sys);
    sys.initialize(reg, u0_ref, step);

    // Baseline: error = 0 at t=0 (u == sol everywhere)
    auto st0 = sys.stats(reg, u0_ref, u0_ref, step);
    REQUIRE_THAT(st0.stats[0], Catch::Matchers::WithinAbs(0.0, 1e-13));

    // Gaussian MMS at t=0: f(x,y,z) = exp(-0.5*(x^2 + y^2 + z^2))
    // Corner (0,0,0) → D[0] = exp(0) = 1.0 (the global max)
    // Corner (1,1,1) → D[719] = exp(-1.5) ≈ 0.22313 (a small value)
    constexpr int nx = 8, ny = 9, nz = 10;
    constexpr int idx_origin = 0;                                       // (0,0,0)
    constexpr int idx_far_corner = (nx - 1) * ny * nz + (ny - 1) * nz + (nz - 1); // (1,1,1)
    static_assert(idx_far_corner == 719);

    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_span(reg, u0_ref, sh);

    // Sanity-check baseline D values at the two corners
    const real val_origin = std::exp(0.0);          // 1.0
    const real val_far = std::exp(-1.5);            // ≈ 0.22313
    REQUIRE(u.D[idx_origin] == Catch::Approx(val_origin).epsilon(1e-12));
    REQUIRE(u.D[idx_far_corner] == Catch::Approx(val_far).epsilon(1e-12));

    // Perturb D buffer at two known fluid points
    const real delta_up = 0.42;    // makes D[0] = 1.42
    const real delta_down = 0.35;  // makes D[719] ≈ -0.127
    u.D[idx_origin] += delta_up;
    u.D[idx_far_corner] -= delta_down;

    // Perturb Rx[0] if available (floating BC → all R are non-dirichlet)
    auto sz = sys.size();
    const real delta_rx = 0.13;
    if (sz.rx_size > 0) {
        u.Rx[0] += delta_rx;
    }

    // Call stats() after perturbation
    auto st1 = sys.stats(reg, u0_ref, u0_ref, step);

    // stats layout: [0]=Linf, [1]=min, [2]=max, [3]=err_d, [4]=err_d_idx,
    //   [5]=err_rx, [6]=idx_rx, [7]=err_ry, [8]=idx_ry, [9]=err_rz, [10]=idx_rz

    // D-buffer error: max(|0.42|, |0.35|) = 0.42 at index 0
    REQUIRE(st1.stats[3] == Catch::Approx(delta_up).epsilon(1e-12));
    REQUIRE(st1.stats[4] == Catch::Approx(0.0));  // err_d_idx = 0

    // Rx error = 0.13 at index 0 (if Rx exists)
    if (sz.rx_size > 0) {
        REQUIRE(st1.stats[5] == Catch::Approx(delta_rx).epsilon(1e-12));
        REQUIRE(st1.stats[6] == Catch::Approx(0.0));  // idx_rx = 0
    }

    // Ry/Rz error = 0 (no perturbation)
    REQUIRE_THAT(st1.stats[7], Catch::Matchers::WithinAbs(0.0, 1e-13));
    REQUIRE_THAT(st1.stats[9], Catch::Matchers::WithinAbs(0.0, 1e-13));

    // Overall Linf = max(0.42, 0.13, 0, 0) = 0.42
    REQUIRE(st1.stats[0] == Catch::Approx(delta_up).epsilon(1e-12));

    // u_max = 1.0 + 0.42 = 1.42 (perturbed D[0] is the new maximum)
    REQUIRE(st1.stats[2] == Catch::Approx(val_origin + delta_up).epsilon(1e-12));

    // u_min = exp(-1.5) - 0.35 ≈ -0.127 (perturbed D[719] is the new minimum)
    REQUIRE(st1.stats[1] == Catch::Approx(val_far - delta_down).epsilon(1e-12));
}

// 16.1b-fix: Exercise the parallel eval_at_locations path (Kokkos::parallel_for).
// The 16.1a test uses Lua MMS (is_thread_safe()=false, serial fallback).
// This test uses Gaussian MMS (is_thread_safe()=true) so the parallel_for branch
// is actually taken. Verifies D-buffer values match the Gaussian formula at corner
// points and that stats error at t=0 is zero.
TEST_CASE("heat - eval_at_locations parallel path (gaussian MMS)")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {8, 9, 10},
                domain_bounds = {
                    min = {0.0, 0.0, 0.0},
                    max = {1.0, 1.0, 1.0}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                xmax = "dirichlet",
            },
            shapes = {
                {
                    type = "sphere",
                    center = {0.5, 0.5, 0.5},
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
            manufactured_solution = {
                type = "gaussian",
                {
                    center = {0.0, 0.0, 0.0},
                    variance = {1.0, 1.0, 1.0},
                    amplitude = 1.0,
                    frequency = 0.1
                }
            }
        }
    )");

    auto sys_opt = system::from_lua(lua["simulation"]);
    REQUIRE(!!sys_opt);
    auto& sys = *sys_opt;
    step_controller step{};

    sim_registry reg;
    auto [u0_ref, rhs_ref] = setup_registry(reg, sys);
    sys.initialize(reg, u0_ref, step);

    // Stats error at t=0 must be zero — this confirms the parallel eval_at_locations
    // computed the same values as the reference (also computed via parallel path).
    auto st = sys.stats(reg, u0_ref, u0_ref, step);
    REQUIRE_THAT(st.stats[0], Catch::Matchers::WithinAbs(0.0, 1e-13));

    // Verify individual D-buffer values at fluid corner points.
    // Gaussian formula at t=0: f(x,y,z) = amp * cos(0) * exp(-0.5*(x^2+y^2+z^2)/1)
    //                        = exp(-0.5*(x^2+y^2+z^2))
    constexpr int nx = 8, ny = 9, nz = 10;
    constexpr real dx = 1.0 / (nx - 1);
    constexpr real dy = 1.0 / (ny - 1);
    constexpr real dz = 1.0 / (nz - 1);

    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_view(reg, u0_ref, sh);

    auto gauss_val = [](real x, real y, real z) {
        return std::exp(-0.5 * (x * x + y * y + z * z));
    };

    auto check_d = [&](int i, int j, int k) {
        int flat = i * ny * nz + j * nz + k;
        real x = i * dx, y = j * dy, z = k * dz;
        real expected = gauss_val(x, y, z);
        INFO("D[" << i << "," << j << "," << k << "] flat=" << flat);
        REQUIRE(u.D[flat] == Catch::Approx(expected).epsilon(1e-12));
    };

    // Corner points — far from sphere, guaranteed fluid
    check_d(0, 0, 0);                 // (0,0,0) -> exp(0) = 1.0
    check_d(nx - 1, ny - 1, nz - 1);  // (1,1,1) -> exp(-1.5)
    check_d(0, 0, nz - 1);            // (0,0,1) -> exp(-0.5)
    check_d(nx - 1, 0, 0);            // (1,0,0) -> exp(-0.5)
    check_d(0, ny - 1, 0);            // (0,1,0) -> exp(-0.5)

    // Verify R-buffer values are finite and in valid range.
    // Gaussian with center at origin, amplitude 1: values in (0, 1].
    auto sz = sys.size();
    for (int i = 0; i < sz.rx_size; ++i) {
        INFO("Rx[" << i << "]");
        REQUIRE(std::isfinite(u.Rx[i]));
        REQUIRE(u.Rx[i] > 0.0);
        REQUIRE(u.Rx[i] <= 1.0 + 1e-10);
    }
    for (int i = 0; i < sz.ry_size; ++i) {
        INFO("Ry[" << i << "]");
        REQUIRE(std::isfinite(u.Ry[i]));
        REQUIRE(u.Ry[i] > 0.0);
        REQUIRE(u.Ry[i] <= 1.0 + 1e-10);
    }
    for (int i = 0; i < sz.rz_size; ++i) {
        INFO("Rz[" << i << "]");
        REQUIRE(std::isfinite(u.Rz[i]));
        REQUIRE(u.Rz[i] > 0.0);
        REQUIRE(u.Rz[i] <= 1.0 + 1e-10);
    }
}

// 17d.5c: Verify graph-based RHS matches eager RHS for heat system.
// Uses the E2 setup with Dirichlet + Neumann grid BCs and a Dirichlet object,
// exercising all graph branches: laplacian, diffusivity scaling, source scatter,
// and BC fill (both grid Dirichlet and object Dirichlet).
TEST_CASE("heat - graph matches eager")
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

    auto heat_opt = systems::heat::from_lua(lua["simulation"]);
    REQUIRE(!!heat_opt);
    auto& h = *heat_opt;
    step_controller step{};

    auto sz = h.size();
    sim_registry reg;

    // Allocate 3 slots: u0 (input), rhs_eager, rhs_graph
    field_ref u0_ref{0}, rhs_ref{1}, rhs2_ref{2};
    for (int s = 0; s < sz.nscalars; ++s) {
        u0_ref = reg.allocate_scalar(0, s, sz.d_size, sz.rx_size, sz.ry_size, sz.rz_size);
        rhs_ref = reg.allocate_scalar(1, s, sz.d_size, sz.rx_size, sz.ry_size, sz.rz_size);
        rhs2_ref = reg.allocate_scalar(2, s, sz.d_size, sz.rx_size, sz.ry_size, sz.rz_size);
    }

    h.initialize(reg, u0_ref, step);
    h.update_boundary(reg, u0_ref, (real)step);

    // Eager path
    h.rhs(reg, u0_ref, reg, rhs_ref, (real)step);
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

    h.fill_source((real)step);
    h.build_rhs_graph(u, du);
    h.submit_rhs_graph();

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

    SECTION("resubmit produces same result") {
        // Zero graph output, refill source, resubmit
        std::ranges::fill(du.D, 0.0);
        std::ranges::fill(du.Rx, 0.0);
        std::ranges::fill(du.Ry, 0.0);
        std::ranges::fill(du.Rz, 0.0);

        h.fill_source((real)step);
        h.submit_rhs_graph();

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
