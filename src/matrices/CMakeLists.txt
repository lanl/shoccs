add_library(shoccs-matrices 
    dense.cpp circulant.cpp inner_block.cpp csr.cpp unit_stride_visitor.cpp)

target_include_directories(shoccs-matrices PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(shoccs-matrices PUBLIC fields)


add_unit_test(dense "matrices" shoccs-matrices shoccs-random)
add_unit_test(circulant "matrices" shoccs-matrices shoccs-random)
add_unit_test(inner_block "matrices" shoccs-matrices shoccs-random)
add_unit_test(block "matrices" shoccs-matrices shoccs-random)
add_unit_test(csr "matrices" shoccs-matrices shoccs-random)
add_unit_test(unit_stride_visitor "matrices" shoccs-matrices fmt::fmt)

