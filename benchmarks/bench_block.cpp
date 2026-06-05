// Benchmark: block matvec (TeamPolicy over lines)
//
// Measures the full block::operator() which dispatches a TeamPolicy kernel
// with one team per mesh line.  Each line applies dense left/right boundary
// matrices and a circulant interior stencil — the same pattern used by the
// derivative operator in the solver.
//
// Parameterized by the number of points per side N of a cubic N³ mesh.
// Number of teams = N², points per line = N, stride = N² (slowest axis).
// Reports time/iteration and effective memory bandwidth.

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

#include "matrices/block.hpp"
#include "matrices/circulant.hpp"
#include "matrices/dense.hpp"
#include "matrices/inner_block.hpp"
#include "types.hpp"

#include <cmath>
#include <vector>

using namespace ccs;

namespace
{

// 4th-order centred first-derivative stencil (h = 1)
constexpr int stencil_width = 5;
constexpr int boundary_rows = 2; // dense rows per boundary

// Build a block matrix that mimics one axis of a 3D derivative operator
// on an N x N x N mesh with stride = N² (differencing along the slowest axis).
matrix::block build_block(int N)
{
    const int N2 = N * N;
    const int n_lines = N2;
    const int pts_per_line = N;
    const int interior_rows = pts_per_line - 2 * boundary_rows;
    const int stride = N2;

    // 4th-order interior stencil coefficients
    const std::vector<real> int_c{
        1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0};

    // Dense left boundary: boundary_rows x stencil_width, one-sided coefficients
    const std::vector<real> left_c{
        -25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -1.0 / 4.0,
        -1.0 / 4.0, -5.0 / 6.0, 3.0 / 2.0, -1.0 / 2.0, 1.0 / 12.0};

    // Dense right boundary: boundary_rows x stencil_width, one-sided coefficients
    const std::vector<real> right_c{
        -1.0 / 12.0, 1.0 / 2.0, -3.0 / 2.0, 5.0 / 6.0, 1.0 / 4.0,
        1.0 / 4.0, -4.0 / 3.0, 3.0, -4.0, 25.0 / 12.0};

    std::vector<matrix::inner_block> blocks;
    blocks.reserve(n_lines);

    for (int line = 0; line < n_lines; ++line) {
        const int row_offset = line; // first element of this line in the flat array
        const int col_offset = row_offset;

        blocks.emplace_back(
            static_cast<integer>(pts_per_line),
            static_cast<integer>(row_offset),
            static_cast<integer>(col_offset),
            static_cast<integer>(stride),
            matrix::dense{boundary_rows, stencil_width, left_c},
            matrix::circulant{interior_rows, int_c},
            matrix::dense{boundary_rows, stencil_width, right_c});
    }

    return matrix::block{std::move(blocks)};
}

void BM_block_matvec(benchmark::State& state)
{
    const auto N = static_cast<int>(state.range(0));
    const auto total = static_cast<std::size_t>(N) * N * N;

    const auto A = build_block(N);

    std::vector<real> x(total);
    std::vector<real> b(total, 0.0);

    // Fill input with a smooth function
    for (std::size_t i = 0; i < total; ++i)
        x[i] = std::sin(2.0 * M_PI * static_cast<real>(i) / static_cast<real>(total));

    // Warm up
    A(x, b);
    Kokkos::fence();

    for (auto _ : state) {
        A(x, b);
        Kokkos::fence();
    }

    // Effective bandwidth: each output point reads stencil_width input values
    // and writes 1 output value.  For interior rows the stencil_width is 5;
    // boundary rows read up to stencil_width values.  Approximate with
    // stencil_width reads + 1 write per point, all doubles.
    const auto n_points = static_cast<double>(N) * N * N;
    const auto bytes_per_point =
        static_cast<double>(stencil_width + 1) * sizeof(real);
    state.counters["BW(GB/s)"] = benchmark::Counter(
        n_points * bytes_per_point,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1024);
    state.counters["points"] = n_points;
    state.counters["lines"] = static_cast<double>(N) * N;
}

BENCHMARK(BM_block_matvec)
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
