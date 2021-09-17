add_library(shoccs-system
  system.cpp 
  empty_system.cpp 
  scalar_wave.cpp 
  inviscid_vortex.cpp 
  heat.cpp
  hyperbolic_eigenvalues.cpp)
target_include_directories(shoccs-system PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(shoccs-system 
  PUBLIC
    fields 
    shoccs-io 
    shoccs-operators 
    shoccs-bcs 
    shoccs-stencils 
    shoccs-mms
    sol2::sol2 lua spdlog::spdlog fmt::fmt)


add_unit_test(heat "systems" shoccs-system)
add_unit_test(hyperbolic_eigenvalues "systems" shoccs-system)
