// Benchmark: selection descriptor scatter (assign_selected)
//
// Measures the cost of scatter-write patterns used by boundary condition
// application: assign_selected(dst, desc, expr) writes to elements described
// by a selection descriptor.  Three descriptor types are benchmarked:
//
//   - contiguous_selection : dense range [offset, offset + count)
//   - strided_selection    : blocked-strided pattern (y/z-plane access)
//   - gather_selection     : arbitrary index list (fluid / object BCs)
//
// Parameterized by selection size (number of elements written).
// Reports effective memory bandwidth (GB/s).

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

#include "fields/expr.hpp"
#include "fields/selection_desc.hpp"

#include <cmath>
#include <vector>

using namespace ccs;

namespace
{

// Total buffer size: selection elements live inside a buffer of this size.
// Large enough that selected indices are a realistic subset.
constexpr int buffer_size = 1 << 21; // 2M elements

// Memory traffic per point:
//   reads:  src[idx]  = 1 × sizeof(real)
//   writes: dst[idx]  = 1 × sizeof(real)
//   total:  2 × sizeof(real) = 16 bytes (double)
constexpr double bytes_per_point = 2.0 * sizeof(real);

// ---------------------------------------------------------------------------
// Contiguous selection: elements [offset, offset + count)
// This is the best-case access pattern (fully coalesced).
// ---------------------------------------------------------------------------
void BM_assign_selected_contiguous(benchmark::State& state)
{
    const auto n = static_cast<int>(state.range(0));

    std::vector<real> src(buffer_size), dst(buffer_size, 0.0);
    for (int i = 0; i < buffer_size; ++i)
        src[i] = std::sin(2.0 * M_PI * i / static_cast<real>(buffer_size));

    // Place selection in the middle of the buffer.
    const int offset = (buffer_size - n) / 2;
    contiguous_selection desc{offset, n};
    handle_expr expr{src.data()};

    // Warm up.
    assign_selected(dst.data(), desc, expr);
    Kokkos::fence();

    for (auto _ : state) {
        assign_selected(dst.data(), desc, expr);
        Kokkos::fence();
    }

    const int actual = desc.count();
    state.counters["BW(GB/s)"] = benchmark::Counter(
        static_cast<double>(actual) * bytes_per_point,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1024);
    state.counters["points"] = static_cast<double>(actual);
}

// ---------------------------------------------------------------------------
// Strided selection: outer_count blocks of inner_count contiguous elements,
// separated by outer_stride.  Mimics y-plane access in a 3-D mesh.
// ---------------------------------------------------------------------------
void BM_assign_selected_strided(benchmark::State& state)
{
    const auto n = static_cast<int>(state.range(0));

    std::vector<real> src(buffer_size), dst(buffer_size, 0.0);
    for (int i = 0; i < buffer_size; ++i)
        src[i] = std::sin(2.0 * M_PI * i / static_cast<real>(buffer_size));

    // Choose dimensions that yield ~n selected elements.
    // Model a y-plane: inner_count = nz, outer_count = nx, outer_stride = ny*nz.
    // We pick nz = sqrt(n), nx = n/nz, ny chosen so the pattern fits in buffer.
    // Note: nz*nx may be less than n when n is not a perfect square.
    int nz = static_cast<int>(std::sqrt(static_cast<double>(n)));
    if (nz < 1) nz = 1;
    int nx = n / nz;
    if (nx < 1) nx = 1;
    // Adjust nz so nx*nz <= n (integer division).
    nz = n / nx;
    // Choose ny so that highest accessed index fits in buffer:
    //   (nx - 1) * ny * nz + nz - 1 < buffer_size
    int ny = nx > 1 ? (buffer_size - nz) / ((nx - 1) * nz) : 2;
    if (ny < 2) ny = 2;
    int outer_stride = ny * nz;

    strided_selection desc{0, nz, nx, outer_stride};
    handle_expr expr{src.data()};

    assign_selected(dst.data(), desc, expr);
    Kokkos::fence();

    for (auto _ : state) {
        assign_selected(dst.data(), desc, expr);
        Kokkos::fence();
    }

    // Use desc.count() (= nz*nx) rather than n: when n is not a perfect
    // square the integer factorisation drops a few elements.
    const int actual = desc.count();
    state.counters["BW(GB/s)"] = benchmark::Counter(
        static_cast<double>(actual) * bytes_per_point,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1024);
    state.counters["points"] = static_cast<double>(actual);
}

// ---------------------------------------------------------------------------
// Gather selection: arbitrary index list.
// Uses a pseudo-random permutation to stress irregular memory access.
// ---------------------------------------------------------------------------
void BM_assign_selected_gather(benchmark::State& state)
{
    const auto n = static_cast<int>(state.range(0));

    std::vector<real> src(buffer_size), dst(buffer_size, 0.0);
    for (int i = 0; i < buffer_size; ++i)
        src[i] = std::sin(2.0 * M_PI * i / static_cast<real>(buffer_size));

    // Build a gather index list: pick n indices spread across the buffer.
    // Use a strided sampling to avoid trivial sequential access while
    // keeping the pattern deterministic.
    std::vector<int> host_indices(n);
    const int stride = buffer_size / n;
    for (int i = 0; i < n; ++i)
        host_indices[i] = (i * stride) % buffer_size;

    // Copy to device view.
    Kokkos::View<int*, memory_space> dev_indices("gather_indices", n);
    auto h_indices = Kokkos::create_mirror_view(dev_indices);
    for (int i = 0; i < n; ++i)
        h_indices(i) = host_indices[i];
    Kokkos::deep_copy(dev_indices, h_indices);

    gather_selection desc{dev_indices};
    handle_expr expr{src.data()};

    assign_selected(dst.data(), desc, expr);
    Kokkos::fence();

    for (auto _ : state) {
        assign_selected(dst.data(), desc, expr);
        Kokkos::fence();
    }

    const int actual = desc.count();
    state.counters["BW(GB/s)"] = benchmark::Counter(
        static_cast<double>(actual) * bytes_per_point,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1024);
    state.counters["points"] = static_cast<double>(actual);
}

BENCHMARK(BM_assign_selected_contiguous)
    ->Arg(1 << 10)   //    1K
    ->Arg(1 << 14)   //   16K
    ->Arg(1 << 17)   //  128K
    ->Arg(1 << 20)   //    1M
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_assign_selected_strided)
    ->Arg(1 << 10)   //    1K
    ->Arg(1 << 14)   //   16K
    ->Arg(1 << 17)   //  128K
    ->Arg(1 << 20)   //    1M
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_assign_selected_gather)
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
