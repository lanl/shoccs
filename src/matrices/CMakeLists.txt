add_library(matrices dense.cpp circulant.cpp inner_block.cpp csr.cpp)

target_include_directories(matrices PUBLIC ..)
target_link_libraries(matrices PUBLIC fields)

add_unit_test(dense "matrices" matrices shoccs-random)
add_unit_test(circulant "matrices" matrices shoccs-random)
add_unit_test(inner_block "matrices" matrices shoccs-random)
add_unit_test(block "matrices" matrices shoccs-random)
add_unit_test(csr "matrices" matrices shoccs-random)
# add_unit_test(Zeros "matrices" matrices)

