add_library(fields INTERFACE)

target_include_directories(fields INTERFACE ..)
target_link_libraries(fields INTERFACE ranges)

add_unit_test(scalar "fields" fields)

add_executable(custom_fields scalar_field.cpp)
target_link_libraries(custom_fields fields)