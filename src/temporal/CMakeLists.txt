add_library(shoccs-integrate
  Integrator.cpp EmptyIntegrator.cpp RK4.cpp)
target_include_directories(shoccs-integrate PUBLIC ..)
target_link_libraries(shoccs-integrate PUBLIC fields shoccs-system)
