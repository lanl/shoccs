// Benchmark: full Heat RHS evaluation via graph submit
//
// Exercises the complete heat::build_rhs_graph() + submit_rhs_graph() path:
// laplacian (3 derivative axes), diffusivity scaling, source scatter from
// manufactured solution, and Dirichlet BC fill — the end-to-end kernel most
// relevant for regression tracking.
//
// Parameterized by mesh size (N³ cubic grid) with E2 stencil and Dirichlet BCs.
// Uses Gaussian MMS (thread-safe, pre-evaluated into member buffers before the
// timed loop, so MMS cost is setup-only).

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <sol/sol.hpp>

#include "fields/field_registry.hpp"
#include "fields/scalar.hpp"
#include "systems/heat.hpp"
#include "temporal/step_controller.hpp"

#include <string>

using namespace ccs;

namespace
{

// Build a heat system from Lua for a cubic N³ mesh with Gaussian MMS.
// Dirichlet BCs on xmin/xmax, Floating on the rest — exercises the full
// graph path including source scatter and BC fill.
systems::heat build_heat(int N)
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);

    // Use string concatenation to inject N into the Lua script.
    std::string script = R"(
        simulation = {
            mesh = {
                index_extents = {)" +
                             std::to_string(N) + ", " + std::to_string(N) + ", " +
                             std::to_string(N) + R"(},
                domain_bounds = {
                    min = {0.0, 0.0, 0.0},
                    max = {1.0, 1.0, 1.0}
                }
            },
            domain_boundaries = {
                xmin = "dirichlet",
                xmax = "dirichlet"
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
                type = "gaussian",
                {
                    center = {0.5, 0.5, 0.5},
                    variance = {0.3, 0.3, 0.3},
                    amplitude = 1.0,
                    frequency = 1.0
                }
            }
        }
    )";

    lua.script(script);

    auto opt = systems::heat::from_lua(lua["simulation"]);
    if (!opt) throw std::runtime_error("Failed to build heat system");
    return std::move(*opt);
}

void BM_heat_rhs(benchmark::State& state)
{
    const auto N = static_cast<int>(state.range(0));
    const auto total = static_cast<std::size_t>(N) * N * N;

    auto heat = build_heat(N);
    auto sz = heat.size();

    // Allocate registry with 2 slots: u0 (input) and du (output).
    sim_registry reg;
    field_ref u0_ref{0}, du_ref{1};
    for (int s = 0; s < sz.nscalars; ++s) {
        u0_ref = reg.allocate_scalar(0, s, sz.d_size, sz.rx_size, sz.ry_size, sz.rz_size);
        du_ref = reg.allocate_scalar(1, s, sz.d_size, sz.rx_size, sz.ry_size, sz.rz_size);
    }

    // Initialize field and boundary conditions.
    step_controller step{};
    heat.initialize(reg, u0_ref, step);
    heat.update_boundary(reg, u0_ref, (real)step);
    heat.fill_source((real)step);

    // Extract scalar views for graph construction.
    constexpr auto sh = scalar_handle{0};
    auto u = extract_scalar_view(reg, u0_ref, sh);
    auto du = extract_scalar_span(reg, du_ref, sh);

    // Build graph once (setup cost, not benchmarked).
    heat.build_rhs_graph(u, du);

    // Warm up.
    heat.submit_rhs_graph();

    for (auto _ : state) {
        heat.submit_rhs_graph();
    }

    // Effective bandwidth estimate.
    // The RHS consists of 3 derivative applications (laplacian) plus scaling
    // and source terms.  Each derivative axis reads ~(2p+1) inputs and writes
    // 1 output per grid point (stencil_width=3 for E2, p=1).  The scale step
    // reads+writes 1 value per point, and source scatter touches a subset.
    // Approximate lower bound: 3 axes × (3 reads + 1 write) + 1 read+write
    // for scale = 14 doubles per point.
    constexpr int stencil_width = 3; // E2: 2*1+1
    const auto n_points = static_cast<double>(total);
    const auto bytes_per_point =
        static_cast<double>(3 * (stencil_width + 1) + 2) * sizeof(real);
    state.counters["BW(GB/s)"] = benchmark::Counter(
        n_points * bytes_per_point,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1024);
    state.counters["points"] = n_points;
}

BENCHMARK(BM_heat_rhs)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Unit(benchmark::kMillisecond);

} // namespace

// Custom main: Kokkos must be initialized before any Kokkos calls.
int main(int argc, char** argv)
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
