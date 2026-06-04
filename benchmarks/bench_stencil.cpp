// Benchmark: circulant convolution (stencil apply)
//
// Measures the core inner loop of the finite-difference operators: a 1-D
// sliding-window dot product of a short coefficient vector against a long
// data vector, implemented via Kokkos::parallel_for in circulant::operator().
//
// Parameterized by grid size (number of interior rows).  The stencil width
// is fixed at 5 (4th-order, p=2) which is the most commonly used order.

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

#include "matrices/circulant.hpp"
#include "types.hpp"

#include <cmath>
#include <vector>

using namespace ccs;

namespace
{

// 4th-order centered first-derivative stencil coefficients (h = 1)
constexpr int stencil_width = 5;
constexpr real coeffs_data[stencil_width] = {
    1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0};

void BM_circulant_apply(benchmark::State& state)
{
    const auto n = static_cast<integer>(state.range(0));
    const int half_w = stencil_width / 2;

    // Total vector length: row_offset + rows + (half_w - 1) for the right overhang.
    // The simple constructor sets row_offset = half_w, so total = half_w + n + half_w - 1.
    const auto total = n + 2 * half_w;

    std::vector<real> x(total);
    std::vector<real> b(total, 0.0);

    // Fill input with a smooth function to give realistic cache patterns
    for (integer i = 0; i < static_cast<integer>(x.size()); ++i)
        x[i] = std::sin(2.0 * M_PI * i / static_cast<real>(total));

    const std::span<const real> coeffs_span{coeffs_data, stencil_width};
    const auto A = matrix::circulant{n, coeffs_span};

    // Warm up
    A(x, b);
    Kokkos::fence();

    for (auto _ : state) {
        A(x, b);
        Kokkos::fence();
    }

    // Report metrics
    // FLOPs per output point: stencil_width multiplies + (stencil_width - 1) adds
    const auto flops_per_point = 2.0 * stencil_width - 1.0;
    state.counters["FLOP/s"] = benchmark::Counter(
        static_cast<double>(n) * flops_per_point,
        benchmark::Counter::kIsIterationInvariantRate);
    state.counters["points"] = static_cast<double>(n);
}

BENCHMARK(BM_circulant_apply)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond);

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
