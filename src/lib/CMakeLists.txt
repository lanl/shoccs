add_library(shoccs-run_sol run_from_sol.cpp)
target_include_directories(shoccs-run_sol PUBLIC ..)
target_link_libraries(shoccs-run_sol PUBLIC lua sol2::sol2 PRIVATE shoccs-simulation)