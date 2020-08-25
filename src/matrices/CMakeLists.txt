add_library(matrices INTERFACE)

target_include_directories(matrices INTERFACE ..)

add_unit_test(dense "dense" matrices ranges)
add_unit_test(circulant "circulant" matrices ranges)
add_unit_test(block "block" matrices ranges)
add_unit_test(csr "csr" matrices ranges)

