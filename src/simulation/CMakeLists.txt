add_library(shoccs-simulation SimulationBuilder.cpp SimulationCycle.cpp)
target_include_directories(shoccs-simulation PUBLIC ..)
target_link_libraries(shoccs-simulation 
    PUBLIC
    lua
    sol2::sol2 
    shoccs-system
    shoccs-integrate)
