add_library(shoccs-mms gauss1d.cpp gauss2d.cpp gauss3d.cpp lua_mms.cpp mms.cpp)
target_include_directories(shoccs-mms PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)

target_link_libraries(shoccs-mms PUBLIC fields sol2::sol2 lua spdlog::spdlog PRIVATE range-v3::range-v3)

add_unit_test(mms "mms" shoccs-mms)
