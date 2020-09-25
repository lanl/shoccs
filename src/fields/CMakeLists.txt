add_library(fields INTERFACE)

target_include_directories(fields INTERFACE ..)
target_link_libraries(fields INTERFACE ranges)

add_unit_test(result_field "fields" fields)
add_unit_test(result_field_range "fields" fields)
add_unit_test(scalar_field "fields" fields)
add_unit_test(vector_field "fields" fields)
