add_library(shoccs-utils INTERFACE)
target_include_directories(shoccs-utils INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)

add_unit_test(bounded "utils" shoccs-utils)
