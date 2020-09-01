add_library(matrices INTERFACE)

target_include_directories(matrices INTERFACE ..)
target_link_libraries(matrices INTERFACE ranges)

add_unit_test(dense "dense" matrices)
add_unit_test(circulant "circulant" matrices)
add_unit_test(block "block" matrices)
add_unit_test(csr "csr" matrices)
add_unit_test(zeros "zeros" matrices)

