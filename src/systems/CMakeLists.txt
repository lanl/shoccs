add_library(shoccs-system
  System.cpp EmptySystem.cpp ScalarWave.cpp)
target_include_directories(shoccs-system PUBLIC ..)
target_link_libraries(shoccs-system PUBLIC fields shoccs-io spdlog::spdlog)
