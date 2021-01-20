add_executable(shoccs shoccs.cpp)
target_link_libraries(shoccs cxxopts::cxxopts lua sol2::sol2)