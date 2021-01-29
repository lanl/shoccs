add_executable(shoccs shoccs.cpp)
target_link_libraries(shoccs cxxopts::cxxopts shoccs-run_sol spdlog::spdlog)