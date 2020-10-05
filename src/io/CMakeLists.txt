add_library(io field_io.cpp xdmf.cpp field_data.cpp)
target_link_libraries(io PUBLIC pugixml::pugixml range-v3::range-v3)
target_include_directories(io PUBLIC ..)

add_unit_test(interval "io")
add_unit_test(field_io "io" io)
