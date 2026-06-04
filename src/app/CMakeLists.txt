add_executable(shoccs-exe shoccs.cpp)
target_link_libraries(shoccs-exe cxxopts::cxxopts shoccs-run_sol spdlog::spdlog Kokkos::kokkos)
set_target_properties(shoccs-exe PROPERTIES OUTPUT_NAME "shoccs")

install(TARGETS shoccs-exe)
