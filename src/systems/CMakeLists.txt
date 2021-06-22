add_library(shoccs-system
  system.cpp empty_system.cpp scalar_wave.cpp inviscid_vortex.cpp heat.cpp)
target_include_directories(shoccs-system PUBLIC ..)
target_link_libraries(shoccs-system 
  PUBLIC
    fields 
    shoccs-io 
    shoccs-operators 
    shoccs-bcs 
    shoccs-stencils 
    mms
    sol2::sol2 lua spdlog::spdlog fmt::fmt)

  add_unit_test(heat "systems" shoccs-system)
