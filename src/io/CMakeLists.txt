add_library(shoccs-io FieldIO.cpp xdmf.cpp field_data.cpp)
target_link_libraries(shoccs-io PUBLIC pugixml::pugixml range-v3::range-v3)
target_include_directories(shoccs-io PUBLIC ..)

add_unit_test(interval "shoccs-io")
#add_unit_test(field_io "io" io)
