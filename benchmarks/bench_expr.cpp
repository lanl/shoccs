// Benchmark: expression template assign and compound-assign
//
// Measures the core element-wise expression evaluation kernels used by the
// solver's field update paths.  These are thin wrappers around
// Kokkos::parallel_for that evaluate expression trees built from handle_expr,
// scalar_literal_expr, and binary_expr nodes.
//
// Benchmarks:
//   - assign(dst, N, a + b * c)   — ternary expression, aliasing-safe path
//   - plus_assign(dst, N, a * b)  — compound-assign (no alias check)
//
// Parameterized by vector length N (1K .. 1M elements).
// Reports effective memory bandwidth (GB/s).
//
// Note: reduce_max / reduce_sum were removed in Phase 18 as dead code.

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

#include "fields/expr.hpp"
#include "types.hpp"

#include <cmath>
#include <functional>
#include <vector>

using namespace ccs;

namespace
{

// assign(dst, N, a + b * c)
//
// Memory traffic per point:
//   reads:  a[i], b[i], c[i]  = 3 × sizeof(real)
//   writes: dst[i]            = 1 × sizeof(real)
//   total:  4 × sizeof(real)  = 32 bytes (double)
void BM_assign_fma(benchmark::State& state)
{
    const auto n = static_cast<int>(state.range(0));

    std::vector<real> a_vec(n), b_vec(n), c_vec(n), dst_vec(n);

    // Fill with smooth data for realistic cache patterns.
    for (int i = 0; i < n; ++i) {
        const auto t = static_cast<real>(i) / static_cast<real>(n);
        a_vec[i] = std::sin(2.0 * M_PI * t);
        b_vec[i] = std::cos(2.0 * M_PI * t);
        c_vec[i] = t;
    }

    // Build expression tree:  a + b * c
    auto expr = binary_expr{std::plus<>{},
                            handle_expr{a_vec.data()},
                            binary_expr{std::multiplies<>{},
                                        handle_expr{b_vec.data()},
                                        handle_expr{c_vec.data()}}};

    // Warm up.
    assign(dst_vec.data(), n, expr);
    Kokkos::fence();

    for (auto _ : state) {
        assign(dst_vec.data(), n, expr);
        Kokkos::fence();
    }

    const auto bytes_per_point = 4.0 * sizeof(real);
    state.counters["BW(GB/s)"] = benchmark::Counter(
        static_cast<double>(n) * bytes_per_point,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1024);
    state.counters["points"] = static_cast<double>(n);
}

// plus_assign(dst, N, a * b)
//
// Memory traffic per point:
//   reads:  dst[i], a[i], b[i]  = 3 × sizeof(real)
//   writes: dst[i]              = 1 × sizeof(real)
//   total:  4 × sizeof(real)    = 32 bytes (double)
void BM_plus_assign_mul(benchmark::State& state)
{
    const auto n = static_cast<int>(state.range(0));

    std::vector<real> a_vec(n), b_vec(n), dst_vec(n);

    for (int i = 0; i < n; ++i) {
        const auto t = static_cast<real>(i) / static_cast<real>(n);
        a_vec[i] = std::sin(2.0 * M_PI * t);
        b_vec[i] = std::cos(2.0 * M_PI * t);
        dst_vec[i] = t;
    }

    auto expr =
        binary_expr{std::multiplies<>{},
                    handle_expr{a_vec.data()},
                    handle_expr{b_vec.data()}};

    // Warm up.
    plus_assign(dst_vec.data(), n, expr);
    Kokkos::fence();

    for (auto _ : state) {
        plus_assign(dst_vec.data(), n, expr);
        Kokkos::fence();
    }

    const auto bytes_per_point = 4.0 * sizeof(real);
    state.counters["BW(GB/s)"] = benchmark::Counter(
        static_cast<double>(n) * bytes_per_point,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1024);
    state.counters["points"] = static_cast<double>(n);
}

BENCHMARK(BM_assign_fma)
    ->Arg(1 << 10)   //    1K
    ->Arg(1 << 14)   //   16K
    ->Arg(1 << 17)   //  128K
    ->Arg(1 << 20)   //    1M
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_plus_assign_mul)
    ->Arg(1 << 10)   //    1K
    ->Arg(1 << 14)   //   16K
    ->Arg(1 << 17)   //  128K
    ->Arg(1 << 20)   //    1M
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
