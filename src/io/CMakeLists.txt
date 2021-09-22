add_library(shoccs-logging logging.cpp)
target_link_libraries(shoccs-logging PUBLIC fmt::fmt spdlog::spdlog)
target_include_directories(shoccs-logging PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
add_unit_test(logging "io" shoccs-logging)


add_library(shoccs-io field_io.cpp xdmf.cpp field_data.cpp)
target_link_libraries(shoccs-io
 PUBLIC pugixml::pugixml fields sol2::sol2 lua shoccs-logging
 PRIVATE shoccs-mesh
)
target_include_directories(shoccs-io PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)

add_unit_test(interval "shoccs-io" shoccs-io)
add_unit_test(xdmf "shoccs-io" shoccs-io)
add_unit_test(field_io "io" shoccs-io)

add_executable(format-test format_test.cpp)
target_link_libraries(format-test PUBLIC fmt::fmt range-v3::range-v3)
