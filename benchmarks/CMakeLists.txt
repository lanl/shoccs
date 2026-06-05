# Google Benchmark suite for SHOCCS critical kernels
#
# Enabled via -DBUILD_BENCHMARKS=ON at configure time.
# Requires the `benchmark` spack package (Google Benchmark).

# Helper: define a benchmark executable linked to common dependencies.
# Does NOT link benchmark_main — each benchmark provides its own main()
# to initialise Kokkos via ScopeGuard before any parallel work.
function(add_bench name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name}
    benchmark::benchmark
    Kokkos::kokkos
    ${ARGN}
  )
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
endfunction()

add_bench(bench_stencil shoccs-matrices)
add_bench(bench_block shoccs-matrices)
add_bench(bench_derivative shoccs-operators shoccs-stencils)
add_bench(bench_expr fields)
add_bench(bench_selection fields)
add_bench(bench_rhs shoccs-system)
