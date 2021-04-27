add_library(shoccs-mesh Cartesian.cpp CutGeometry.cpp Rect.cpp Sphere.cpp Mesh.cpp)

target_include_directories(shoccs-mesh PUBLIC ..)
target_link_libraries(shoccs-mesh PUBLIC fields)

add_library(mesh_view INTERFACE)
target_link_libraries(mesh_view INTERFACE shoccs-mesh cppcoro)

add_unit_test(Cartesian "mesh" shoccs-mesh)
# add_unit_test(mesh_view "mesh" mesh_view range-v3::range-v3)
add_unit_test(CutGeometry "mesh" shoccs-mesh)
add_unit_test(Mesh "mesh" shoccs-mesh shoccs-random)