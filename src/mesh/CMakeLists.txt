add_library(shoccs-mesh cartesian.cpp object_geometry.cpp rect.cpp sphere.cpp mesh.cpp)

target_include_directories(shoccs-mesh PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(shoccs-mesh PUBLIC fields sol2::sol2 lua shoccs-logging)

# add_library(mesh_view INTERFACE)
# target_link_libraries(mesh_view INTERFACE shoccs-mesh cppcoro)

add_unit_test(cartesian "mesh" shoccs-mesh)
# add_unit_test(mesh_view "mesh" mesh_view range-v3::range-v3)
add_unit_test(object_geometry "mesh" shoccs-mesh)
add_unit_test(mesh "mesh" shoccs-mesh shoccs-random)
add_unit_test(shapes "mesh" shoccs-mesh shoccs-random)
