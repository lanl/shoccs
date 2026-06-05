add_library(shoccs-matrices 
    dense.cpp
    circulant.cpp
    inner_block.cpp 
    csr.cpp 
    unit_stride_visitor.cpp 
    coefficient_visitor.cpp)

target_include_directories(shoccs-matrices PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(shoccs-matrices PUBLIC fields)


if (BUILD_TESTING)
  add_executable(t-csr csr.t.cpp)
  target_link_libraries(t-csr Catch2::Catch2 shoccs-matrices shoccs-random Kokkos::kokkos)
  add_test(NAME t-csr COMMAND t-csr)
  set_tests_properties(t-csr PROPERTIES LABELS "matrices")
endif()

if (BUILD_TESTING)
  add_executable(t-dense dense.t.cpp)
  target_link_libraries(t-dense Catch2::Catch2 shoccs-matrices shoccs-random Kokkos::kokkos)
  add_test(NAME t-dense COMMAND t-dense)
  set_tests_properties(t-dense PROPERTIES LABELS "matrices")
endif()

if (BUILD_TESTING)
  add_executable(t-unit_stride_visitor unit_stride_visitor.t.cpp)
  target_link_libraries(t-unit_stride_visitor Catch2::Catch2 shoccs-matrices Kokkos::kokkos)
  add_test(NAME t-unit_stride_visitor COMMAND t-unit_stride_visitor)
  set_tests_properties(t-unit_stride_visitor PROPERTIES LABELS "matrices")

  add_executable(t-coefficient_visitor coefficient_visitor.t.cpp)
  target_link_libraries(t-coefficient_visitor Catch2::Catch2 shoccs-matrices Kokkos::kokkos)
  add_test(NAME t-coefficient_visitor COMMAND t-coefficient_visitor)
  set_tests_properties(t-coefficient_visitor PROPERTIES LABELS "matrices")

  add_executable(t-block block.t.cpp)
  target_link_libraries(t-block Catch2::Catch2 shoccs-matrices shoccs-random Kokkos::kokkos)
  add_test(NAME t-block COMMAND t-block)
  set_tests_properties(t-block PROPERTIES LABELS "matrices")

  add_executable(t-circulant circulant.t.cpp)
  target_link_libraries(t-circulant Catch2::Catch2 shoccs-matrices shoccs-random Kokkos::kokkos)
  add_test(NAME t-circulant COMMAND t-circulant)
  set_tests_properties(t-circulant PROPERTIES LABELS "matrices")

  add_executable(t-inner_block inner_block.t.cpp)
  target_link_libraries(t-inner_block Catch2::Catch2 shoccs-matrices shoccs-random Kokkos::kokkos)
  add_test(NAME t-inner_block COMMAND t-inner_block)
  set_tests_properties(t-inner_block PROPERTIES LABELS "matrices")
endif()

