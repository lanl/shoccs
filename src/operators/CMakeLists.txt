add_library(operators directional.cpp gradient.cpp derivative.cpp)

target_link_libraries(operators PUBLIC geometry mesh range-v3::range-v3 matrices
PRIVATE cppcoro)
target_include_directories(operators PUBLIC ..)

add_unit_test(directional "operators" operators)
add_unit_test(gradient "operators" operators random)