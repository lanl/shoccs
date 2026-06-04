// Benchmark: derivative operator (full chain)
//
// Measures the full derivative::operator() which applies the block (circulant +
// dense boundary) matrix O, plus CSR matrices B/N for boundary contributions.
// This is the complete per-axis differentiation kernel used in the solver.
//
// Parameterized by mesh size (N³ cubic grid) and stencil order (E2, E4).
// No embedded objects — pure Cartesian grid with Floating BCs on all faces.
// Reports time/iteration and effective memory bandwidth.

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

#include "fields/scalar.hpp"
#include "mesh/mesh.hpp"
#include "operators/derivative.hpp"
#include "stencils/stencil.hpp"
#include "types.hpp"

#include <cmath>
#include <vector>

using namespace ccs;

namespace
{

// Owning scalar with implicit conversion to scalar_view / scalar_span.
struct owned_scalar {
    std::vector<real> d_vec, rx_vec, ry_vec, rz_vec;

    operator scalar_view() const { return {d_vec, rx_vec, ry_vec, rz_vec}; }
    operator scalar_span() { return {d_vec, rx_vec, ry_vec, rz_vec}; }
};

owned_scalar make_scalar(const mesh& m)
{
    return {std::vector<real>(m.size()),
            std::vector<real>(m.Rx().size()),
            std::vector<real>(m.Ry().size()),
            std::vector<real>(m.Rz().size())};
}

// Stencil selector: range(0) = mesh size, range(1) = stencil order (2 or 4).
const stencil& select_stencil(int order)
{
    switch (order) {
    case 4:
        return stencils::second::E4;
    default:
        return stencils::second::E2;
    }
}

void BM_derivative_apply(benchmark::State& state)
{
    const auto N = static_cast<int>(state.range(0));
    const auto order = static_cast<int>(state.range(1));
    const auto total = static_cast<std::size_t>(N) * N * N;

    // Build a cubic mesh with uniform spacing.
    auto m = mesh{index_extents{int3{N, N, N}},
                  domain_extents{.min = {0.0, 0.0, 0.0}, .max = {1.0, 1.0, 1.0}}};

    // Floating BCs on all faces — no Dirichlet rows removed.
    const auto gridBcs = bcs::Grid{bcs::ff, bcs::ff, bcs::ff};
    const auto objectBcs = bcs::Object{};

    // Differentiate along axis 0 (representative; all axes are symmetric).
    const auto& st = select_stencil(order);
    auto d = derivative{0, m, st, gridBcs, objectBcs};

    auto u = make_scalar(m);
    auto du = make_scalar(m);

    // Fill input with a smooth function for realistic cache patterns.
    for (std::size_t i = 0; i < total; ++i)
        u.d_vec[i] = std::sin(2.0 * M_PI * static_cast<real>(i) /
                               static_cast<real>(total));

    // Warm up.
    d(u, du);
    Kokkos::fence();

    for (auto _ : state) {
        d(u, du);
        Kokkos::fence();
    }

    // Stencil half-width determines reads per output point.
    const auto p = (order == 4) ? 2 : 1;
    const auto stencil_width = 2 * p + 1;

    // Effective bandwidth: each output point reads stencil_width input values
    // and writes 1 output value (all doubles).
    const auto n_points = static_cast<double>(total);
    const auto bytes_per_point =
        static_cast<double>(stencil_width + 1) * sizeof(real);
    state.counters["BW(GB/s)"] = benchmark::Counter(
        n_points * bytes_per_point,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1024);
    state.counters["points"] = n_points;
    state.counters["order"] = static_cast<double>(order);
}

// Parameterize: {mesh_size, stencil_order}.
BENCHMARK(BM_derivative_apply)
    ->Args({16, 2})
    ->Args({32, 2})
    ->Args({64, 2})
    ->Args({16, 4})
    ->Args({32, 4})
    ->Args({64, 4})
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
