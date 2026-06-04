add_library(shoccs-integrate
  integrator.cpp rk4.cpp euler.cpp step_controller.cpp)
target_include_directories(shoccs-integrate PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(shoccs-integrate
  PUBLIC
    fields shoccs-system sol2::sol2 lua spdlog::spdlog)

add_unit_test(step_controller "temporal" shoccs-integrate)

if (BUILD_TESTING)
  add_executable(t-rk4_v2 rk4_v2.t.cpp)
  target_link_libraries(t-rk4_v2 Catch2::Catch2 shoccs-integrate Kokkos::kokkos)
  add_test(NAME t-rk4_v2 COMMAND t-rk4_v2)
  set_tests_properties(t-rk4_v2 PROPERTIES LABELS "temporal")

  add_executable(t-euler_v2 euler_v2.t.cpp)
  target_link_libraries(t-euler_v2 Catch2::Catch2 shoccs-integrate Kokkos::kokkos)
  add_test(NAME t-euler_v2 COMMAND t-euler_v2)
  set_tests_properties(t-euler_v2 PROPERTIES LABELS "temporal")
endif()
