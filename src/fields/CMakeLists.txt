add_library(fields INTERFACE)

target_include_directories(fields INTERFACE ..)
target_link_libraries(fields INTERFACE range-v3::range-v3 cppcoro)

add_unit_test(result_field "fields" fields)
add_unit_test(scalar_field "fields" fields)
add_unit_test(vector_field "fields" fields)
add_unit_test(scalar "fields" fields)
add_unit_test(vector "fields" fields)
add_unit_test(r_tuple "field" fields)
add_unit_test(result_r_tuple "field" fields)

add_executable(r_tuple_seg_fault r_tuple_seg_fault.cpp)
target_link_libraries(r_tuple_seg_fault fields)