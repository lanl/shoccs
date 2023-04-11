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

find_package(Lua REQUIRED)
# make lua a target
add_library(lua INTERFACE IMPORTED)
target_include_directories(lua INTERFACE ${LUA_INCLUDE_DIR})
target_link_libraries(lua INTERFACE ${LUA_LIBRARIES})

include(CTest)

find_package(sol2 REQUIRED)
find_package(fmt 8 REQUIRED)
find_package(range-v3 REQUIRED)
find_package(pugixml 1.10 REQUIRED)
find_package(spdlog 1.9 REQUIRED)
find_package(cxxopts REQUIRED)
if (BUILD_TESTING)
  find_package(Catch2 3 REQUIRED)
endif()
find_package(Boost REQUIRED) # for header only mp11
find_package(lapackpp REQUIRED)


include(GNUInstallDirs)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()


# helper function for defining serial tests
function(add_unit_test t label)
  if (BUILD_TESTING)
    add_executable(t-${t} ${t}.t.cpp)
    target_link_libraries(t-${t} Catch2::Catch2WithMain ${ARGN})
    add_test(NAME t-${t} COMMAND t-${t})
    set_tests_properties(t-${t} PROPERTIES LABELS "${label}")
  endif()
endfunction()

if(NOT APPLE)
  set(CMAKE_INSTALL_RPATH $ORIGIN)
endif()

add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(docs)


install(
    TARGETS
      fields
      shoccs-logging
      shoccs-io
      shoccs-matrices
      shoccs-mesh
      shoccs-mms
      shoccs-bcs
      shoccs-operators
      shoccs-simulation
      shoccs-system
      shoccs-integrate
      shoccs-stencils
      shoccs-utils
    EXPORT shoccs
)
install(EXPORT shoccs
  NAMESPACE shoccs::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shoccs
  FILE shoccsTargets.cmake
  )

configure_file(${PROJECT_SOURCE_DIR}/config/shoccsConfig.cmake.in
  ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/shoccsConfig.cmake @ONLY)

install(
  FILES ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/shoccsConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shoccs)
