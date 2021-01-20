add_library(pde-integrate
  euler.cpp rk4.cpp)
target_link_libraries(pde-integrate PUBLIC pde-system)
