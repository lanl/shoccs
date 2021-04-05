add_library(matrices Dense.cpp Circulant.cpp InnerBlock.cpp CSR.cpp)

target_include_directories(matrices PUBLIC ..)
target_link_libraries(matrices PUBLIC fields)

add_unit_test(Dense "matrices" matrices shoccs-random)
add_unit_test(Circulant "matrices" matrices shoccs-random)
add_unit_test(InnerBlock "matrices" matrices shoccs-random)
add_unit_test(Block "matrices" matrices shoccs-random)
add_unit_test(CSR "matrices" matrices shoccs-random)
# add_unit_test(Zeros "matrices" matrices)

