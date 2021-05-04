add_library(fields INTERFACE)

target_include_directories(fields INTERFACE ..)
target_link_libraries(fields INTERFACE range-v3::range-v3 cppcoro Boost::boost)

add_unit_test(range_concepts "concepts" fields)
add_unit_test(tuple_utils "fields" fields)
add_unit_test(tuple_pipe "field" fields)
add_unit_test(algorithms "fields" fields)
add_unit_test(container_tuple "fields" fields)
add_unit_test(view_tuple "fields" fields)
add_unit_test(tuple "fields" fields)
add_unit_test(tuple_math "fields" fields)
add_unit_test(scalar "fields" fields)
add_unit_test(selector "fields" fields)

add_executable(seg view_tuple_seg.cpp)
target_link_libraries(seg fields)