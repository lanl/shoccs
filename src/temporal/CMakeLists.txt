add_library(shoccs-integrate
  integrator.cpp empty_integrator.cpp rk4.cpp)
target_include_directories(shoccs-integrate PUBLIC ..)
target_link_libraries(shoccs-integrate 
  PUBLIC
    fields shoccs-system sol2::sol2 lua spdlog::spdlog)
