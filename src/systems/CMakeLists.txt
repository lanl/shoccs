add_library(shoccs-system
  system.cpp 
  empty_system.cpp 
  scalar_wave.cpp 
  inviscid_vortex.cpp 
  heat.cpp
  hyperbolic_eigenvalues.cpp)
target_include_directories(shoccs-system PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(shoccs-system 
  PUBLIC
    fields 
    shoccs-io 
    shoccs-operators 
    shoccs-bcs 
    shoccs-stencils 
    shoccs-mms
    sol2::sol2 lua spdlog::spdlog fmt::fmt)


if (BUILD_TESTING)
  add_executable(t-heat heat.t.cpp)
  target_link_libraries(t-heat Catch2::Catch2 shoccs-system Kokkos::kokkos)
  add_test(NAME t-heat COMMAND t-heat)
  set_tests_properties(t-heat PROPERTIES LABELS "systems")

  add_executable(t-scalar_wave scalar_wave.t.cpp)
  target_link_libraries(t-scalar_wave Catch2::Catch2 shoccs-system Kokkos::kokkos)
  add_test(NAME t-scalar_wave COMMAND t-scalar_wave)
  set_tests_properties(t-scalar_wave PROPERTIES LABELS "systems")

  add_executable(t-hyperbolic_eigenvalues hyperbolic_eigenvalues.t.cpp)
  target_link_libraries(t-hyperbolic_eigenvalues Catch2::Catch2 shoccs-system Kokkos::kokkos)
  add_test(NAME t-hyperbolic_eigenvalues COMMAND t-hyperbolic_eigenvalues)
  set_tests_properties(t-hyperbolic_eigenvalues PROPERTIES LABELS "systems")

endif()
