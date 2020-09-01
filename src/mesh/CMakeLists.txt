add_library(mesh mesh.cpp)

target_include_directories(mesh PUBLIC ..)
target_link_libraries(mesh PRIVATE ranges)

add_unit_test(mesh "mesh" mesh ranges)