add_library(shoccs-stencils E2_2.cpp)

target_include_directories(shoccs-stencils PUBLIC ..)
target_link_libraries(shoccs-stencils PRIVATE range-v3::range-v3)

add_unit_test(E2_2 "stencils" shoccs-stencils range-v3::range-v3)