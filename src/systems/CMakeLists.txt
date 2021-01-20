add_library(pde-system
  system.cpp vc_scalar_wave.cpp cc_heat_equation.cpp cc_elliptic.cpp euler_vortex.cpp)
target_link_libraries(pde-system PUBLIC ranges pde-mms pde-utils)
