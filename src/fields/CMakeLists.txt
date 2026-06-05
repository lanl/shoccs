add_library(fields INTERFACE)

target_include_directories(fields INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(fields INTERFACE Boost::boost Kokkos::kokkos)

add_unit_test(handle "fields" fields)

if (BUILD_TESTING)
  add_executable(t-field_registry field_registry.t.cpp)
  target_link_libraries(t-field_registry Catch2::Catch2 fields Kokkos::kokkos)
  add_test(NAME t-field_registry COMMAND t-field_registry)
  set_tests_properties(t-field_registry PROPERTIES LABELS "fields")

  add_executable(t-expr expr.t.cpp)
  target_link_libraries(t-expr Catch2::Catch2 fields Kokkos::kokkos)
  add_test(NAME t-expr COMMAND t-expr)
  set_tests_properties(t-expr PROPERTIES LABELS "fields")

  add_executable(t-selection_desc selection_desc.t.cpp)
  target_link_libraries(t-selection_desc Catch2::Catch2 fields shoccs-logging Kokkos::kokkos)
  add_test(NAME t-selection_desc COMMAND t-selection_desc)
  set_tests_properties(t-selection_desc PROPERTIES LABELS "fields")

  add_executable(t-graph_poc graph_poc.t.cpp)
  target_link_libraries(t-graph_poc Catch2::Catch2 fields shoccs-matrices Kokkos::kokkos)
  add_test(NAME t-graph_poc COMMAND t-graph_poc)
  set_tests_properties(t-graph_poc PROPERTIES LABELS "fields")
endif()
