add_library(shoccs-utils INTERFACE)
target_include_directories(shoccs-utils INTERFACE ..)

add_unit_test(bounded "utils" shoccs-utils)
