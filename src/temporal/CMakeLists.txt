add_library(shoccs-integrate
  integrator.cpp empty_integrator.cpp rk4.cpp euler.cpp step_controller.cpp)
target_include_directories(shoccs-integrate PUBLIC ..)
target_link_libraries(shoccs-integrate 
  PUBLIC
    fields shoccs-system sol2::sol2 lua spdlog::spdlog)

add_unit_test(step_controller "temporal" shoccs-integrate)
add_unit_test(rk4 "temporal" shoccs-integrate)
add_unit_test(euler "temporal" shoccs-integrate)
