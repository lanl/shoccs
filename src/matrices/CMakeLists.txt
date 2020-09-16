add_library(matrices INTERFACE)

target_include_directories(matrices INTERFACE ..)
target_link_libraries(matrices INTERFACE ranges)

add_unit_test(dense "matrices" matrices)
add_unit_test(circulant "matrices" matrices)
add_unit_test(inner_block "matrices" matrices)
add_unit_test(block "matrices" matrices)
add_unit_test(csr "matrices" matrices)
add_unit_test(zeros "matrices" matrices)

