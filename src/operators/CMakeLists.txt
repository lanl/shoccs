add_library(operators directional.cpp)

target_link_libraries(operators PUBLIC geometry mesh ranges matrices
PRIVATE cppcoro)
target_include_directories(operators PUBLIC ..)

add_unit_test(directional "operators" operators)