add_library(shoccs-mesh cartesian.cpp object_geometry.cpp rect.cpp sphere.cpp mesh.cpp)

target_include_directories(shoccs-mesh PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(shoccs-mesh PUBLIC fields sol2::sol2 lua shoccs-logging)

add_unit_test(cartesian "mesh" shoccs-mesh)
add_unit_test(object_geometry "mesh" shoccs-mesh)
if (BUILD_TESTING)
  add_executable(t-mesh mesh.t.cpp)
  target_link_libraries(t-mesh Catch2::Catch2 shoccs-mesh shoccs-random Kokkos::kokkos)
  add_test(NAME t-mesh COMMAND t-mesh)
  set_tests_properties(t-mesh PROPERTIES LABELS "mesh")
endif()
add_unit_test(shapes "mesh" shoccs-mesh shoccs-random)
