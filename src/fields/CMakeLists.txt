add_library(fields INTERFACE)

target_include_directories(fields INTERFACE ..)
target_link_libraries(fields INTERFACE range-v3::range-v3 cppcoro Boost::boost)

add_unit_test(range_concepts "concepts" fields)
add_unit_test(TupleUtils "fields" fields)
add_unit_test(TuplePipe "field" fields)
add_unit_test(algorithms "fields" fields)
add_unit_test(ContainerTuple "fields" fields)
add_unit_test(ViewTuple "fields" fields)
add_unit_test(Tuple "fields" fields)
add_unit_test(TupleMath "fields" fields)
add_unit_test(Scalar "fields" fields)
add_unit_test(Vector "fields" fields)

add_executable(ViewTuple_Seg ViewTuple_Seg.cpp)
target_link_libraries(ViewTuple_Seg fields)