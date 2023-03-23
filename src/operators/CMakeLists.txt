add_library(shoccs-bcs boundaries.cpp)
target_include_directories(shoccs-bcs PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(shoccs-bcs PUBLIC sol2::sol2 lua shoccs-logging)
add_unit_test(boundaries "bcs" shoccs-bcs)

add_library(shoccs-operators
    gradient.cpp
    laplacian.cpp
    derivative.cpp
    eigenvalue_visitor.cpp)

target_link_libraries(shoccs-operators
    PUBLIC
        shoccs-mesh shoccs-matrices shoccs-logging
    PRIVATE
        lapackpp)
target_include_directories(shoccs-operators PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)



#add_unit_test(directional "operators" operators)
add_unit_test(derivative "operators" shoccs-operators shoccs-random shoccs-stencils fmt::fmt)
#add_unit_test(divergence "operators" operators random)

add_unit_test(gradient "operators" shoccs-operators shoccs-stencils shoccs-bcs)
add_unit_test(laplacian "operators" shoccs-operators shoccs-stencils shoccs-bcs)
add_unit_test(eigenvalue_visitor "operators" shoccs-operators shoccs-stencils shoccs-bcs)
