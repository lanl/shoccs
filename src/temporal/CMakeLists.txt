add_library(shoccs-integrate
  integrator.cpp empty_integrator.cpp rk4.cpp step_controller.cpp)
target_include_directories(shoccs-integrate PUBLIC ..)
target_link_libraries(shoccs-integrate 
  PUBLIC
    fields shoccs-system sol2::sol2 lua spdlog::spdlog)

add_unit_test(step_controller "temporal" shoccs-integrate)
