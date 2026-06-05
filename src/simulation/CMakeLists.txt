add_library(shoccs-simulation simulation_builder.cpp simulation_cycle.cpp)
target_include_directories(shoccs-simulation PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(shoccs-simulation 
    PUBLIC
    lua
    sol2::sol2 
    shoccs-system
    shoccs-integrate)

if (BUILD_TESTING)
  add_executable(t-simulation_cycle simulation_cycle.t.cpp)
  target_link_libraries(t-simulation_cycle Catch2::Catch2 shoccs-simulation Kokkos::kokkos)
  add_test(NAME t-simulation_cycle COMMAND t-simulation_cycle)
  set_tests_properties(t-simulation_cycle PROPERTIES LABELS "simulation")
endif()
