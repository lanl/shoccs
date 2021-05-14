cmake_minimum_required(VERSION 3.16)

project(shoccs LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(SHOCCS_TPL_DIR "Third party library installation directory")
if (SHOCCS_TPL_DIR)
  list(APPEND CMAKE_PREFIX_PATH ${SHOCCS_TPL_DIR})
endif()

find_package(Lua 5.3 REQUIRED)
# make lua a target
add_library(lua INTERFACE IMPORTED)
target_include_directories(lua INTERFACE ${LUA_INCLUDE_DIR})
target_link_libraries(lua INTERFACE ${LUA_LIBRARIES})

find_package(sol2 REQUIRED)
find_package(fmt REQUIRED)
find_package(range-v3 REQUIRED)
find_package(pugixml 1.10 REQUIRED)
find_package(spdlog REQUIRED)
find_package(cxxopts REQUIRED)
find_package(Catch2 3)
if (Catch2_FOUND)
  message(STATUS "Catch2 has been found.  Enabling testing via ctest")
  enable_testing()
endif()
find_package(Boost REQUIRED) # for header only mp11



include(GNUInstallDirs)

# helper function for defining serial tests
function(add_unit_test test label)
  add_executable(t-${test} ${test}.t.cpp)
  target_link_libraries(t-${test} Catch2::Catch2WithMain ${ARGN})
  add_test(NAME t-${test} COMMAND t-${test})
  set_tests_properties(t-${test} PROPERTIES LABELS "${label}")
endfunction()

add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(docs)
