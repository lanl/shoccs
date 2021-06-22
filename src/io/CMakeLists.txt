add_library(shoccs-io field_io.cpp xdmf.cpp field_data.cpp)
target_link_libraries(shoccs-io PUBLIC pugixml::pugixml fields fmt::fmt sol2::sol2 lua spdlog::spdlog)
target_include_directories(shoccs-io PUBLIC ..)

add_unit_test(interval "shoccs-io" shoccs-io)
add_unit_test(xdmf "shoccs-io" shoccs-io)
add_unit_test(field_io "io" shoccs-io)

add_executable(format-test format_test.cpp)
target_link_libraries(format-test PUBLIC fmt::fmt range-v3::range-v3)
