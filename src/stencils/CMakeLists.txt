add_library(shoccs-stencils
    E2_2.cpp
    E4_2.cpp
    E2_1.cpp
    polyE2_1.cpp
    stencil.cpp)

target_include_directories(shoccs-stencils PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(shoccs-stencils PUBLIC sol2::sol2 lua spdlog::spdlog fmt::fmt PRIVATE range-v3::range-v3)

add_unit_test(E2_2 "stencils" shoccs-stencils range-v3::range-v3)
add_unit_test(E4_2 "stencils" shoccs-stencils range-v3::range-v3)
add_unit_test(E2_1 "stencils" shoccs-stencils range-v3::range-v3)
