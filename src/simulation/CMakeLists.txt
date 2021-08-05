add_library(shoccs-simulation simulation_builder.cpp simulation_cycle.cpp)
target_include_directories(shoccs-simulation PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(shoccs-simulation 
    PUBLIC
    lua
    sol2::sol2 
    shoccs-system
    shoccs-integrate)

add_unit_test(simulation_cycle "simulation" shoccs-simulation)
