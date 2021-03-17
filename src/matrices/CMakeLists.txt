add_library(matrices INTERFACE)

target_include_directories(matrices INTERFACE ..)
target_link_libraries(matrices INTERFACE fields)

add_unit_test(Dense "matrices" matrices)
add_unit_test(Circulant "matrices" matrices)
add_unit_test(InnerBlock "matrices" matrices)
add_unit_test(Block "matrices" matrices)
add_unit_test(CSR "matrices" matrices)
add_unit_test(Zeros "matrices" matrices)

