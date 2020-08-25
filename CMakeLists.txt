cmake_minimum_required(VERSION 3.16)

project(shoccs LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(Lua REQUIRED)
# make lua a target
add_library(lua INTERFACE IMPORTED)
target_include_directories(lua INTERFACE ${LUA_INCLUDE_DIR})
target_link_libraries(lua INTERFACE ${LUA_LIBRARIES})

find_package(sol2 REQUIRED)
find_package(fmt REQUIRED)
find_package(Catch2 3)
if (Catch2_FOUND)
  message(STATUS "Catch2 has been found.  Enabling testing via ctest")
  enable_testing()
endif()

add_library(ranges INTERFACE)
target_include_directories(ranges INTERFACE "/Users/ptb/gitrepos/range-v3/include")

include(GNUInstallDirs)

# helper function for defining serial tests
function(add_unit_test test label)
  add_executable(t-${test} ${test}.t.cpp)
  target_link_libraries(t-${test} Catch2::Catch2WithMain ${ARGN})
  add_test(NAME t-${test} COMMAND t-${test})
  set_tests_properties(t-${test} PROPERTIES LABELS "${label}")
endfunction()

add_subdirectory(src)
add_subdirectory(docs)
