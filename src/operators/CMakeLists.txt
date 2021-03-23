#add_library(operators directional.cpp derivative.cpp gradient.cpp divergence.cpp)
add_library(shoccs-operators Gradient.cpp Derivative.cpp)
#target_link_libraries(operators PUBLIC geometry mesh range-v3::range-v3 matrices fields)
target_link_libraries(shoccs-operators PUBLIC shoccs-mesh matrices)
target_include_directories(shoccs-operators PUBLIC ..)

#add_unit_test(directional "operators" operators)
#add_unit_test(derivative "operators" operators random)
#add_unit_test(gradient "operators" operators random)
#add_unit_test(divergence "operators" operators random)

add_unit_test(Gradient "operators" shoccs-operators shoccs-random)