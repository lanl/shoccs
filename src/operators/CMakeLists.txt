add_library(shoccs-bcs boundaries.cpp)
target_link_libraries(shoccs-bcs PUBLIC sol2::sol2 lua spdlog::spdlog)
add_unit_test(boundaries "bcs" shoccs-bcs)

#add_library(operators directional.cpp derivative.cpp gradient.cpp divergence.cpp)
add_library(shoccs-operators gradient.cpp laplacian.cpp derivative.cpp)
#target_link_libraries(operators PUBLIC geometry mesh range-v3::range-v3 matrices fields)
target_link_libraries(shoccs-operators PUBLIC shoccs-mesh matrices)
target_include_directories(shoccs-operators PUBLIC ..)

#add_unit_test(directional "operators" operators)
add_unit_test(derivative "operators" shoccs-operators shoccs-random shoccs-stencils)
#add_unit_test(gradient "operators" operators random)
#add_unit_test(divergence "operators" operators random)

add_unit_test(gradient "operators" shoccs-operators shoccs-random)
add_unit_test(laplacian "operators" shoccs-operators shoccs-stencils shoccs-bcs)
