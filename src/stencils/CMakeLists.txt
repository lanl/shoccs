add_library(stencils E2_2.cpp)

target_include_directories(stencils PUBLIC ..)
target_link_libraries(stencils PRIVATE ranges)

add_unit_test(E2_2 "stencils" stencils ranges)