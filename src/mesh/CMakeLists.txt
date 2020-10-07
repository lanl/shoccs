add_library(mesh mesh.cpp)

target_include_directories(mesh PUBLIC ..)
target_link_libraries(mesh PRIVATE range-v3::range-v3)

add_library(mesh_view INTERFACE)
target_link_libraries(mesh_view INTERFACE mesh cppcoro)

add_unit_test(mesh "mesh" mesh range-v3::range-v3)
add_unit_test(mesh_view "mesh" mesh_view range-v3::range-v3)