add_library(shoccs-bcs boundaries.cpp)
target_include_directories(shoccs-bcs PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(shoccs-bcs PUBLIC sol2::sol2 lua shoccs-logging)
add_unit_test(boundaries "bcs" shoccs-bcs)

add_library(shoccs-operators
    gradient.cpp
    laplacian.cpp
    derivative.cpp
    eigenvalue_visitor.cpp)

target_link_libraries(shoccs-operators
    PUBLIC
        shoccs-mesh shoccs-matrices shoccs-logging
    PRIVATE
        lapackpp)
target_include_directories(shoccs-operators PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)


#add_unit_test(divergence "operators" operators random)

if (BUILD_TESTING)
  add_executable(t-derivative derivative.t.cpp)
  target_link_libraries(t-derivative Catch2::Catch2 shoccs-operators shoccs-random shoccs-stencils fmt::fmt Kokkos::kokkos)
  add_test(NAME t-derivative COMMAND t-derivative)
  set_tests_properties(t-derivative PROPERTIES LABELS "operators")

  add_executable(t-gradient gradient.t.cpp)
  target_link_libraries(t-gradient Catch2::Catch2 shoccs-operators shoccs-stencils shoccs-bcs Kokkos::kokkos)
  add_test(NAME t-gradient COMMAND t-gradient)
  set_tests_properties(t-gradient PROPERTIES LABELS "operators")

  add_executable(t-laplacian laplacian.t.cpp)
  target_link_libraries(t-laplacian Catch2::Catch2 shoccs-operators shoccs-stencils shoccs-bcs Kokkos::kokkos)
  add_test(NAME t-laplacian COMMAND t-laplacian)
  set_tests_properties(t-laplacian PROPERTIES LABELS "operators")

  add_executable(t-eigenvalue_visitor eigenvalue_visitor.t.cpp)
  target_link_libraries(t-eigenvalue_visitor Catch2::Catch2 shoccs-operators shoccs-stencils shoccs-bcs Kokkos::kokkos)
  add_test(NAME t-eigenvalue_visitor COMMAND t-eigenvalue_visitor)
  set_tests_properties(t-eigenvalue_visitor PROPERTIES LABELS "operators")
endif()
