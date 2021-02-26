add_library(shoccs-system
  System.cpp EmptySystem.cpp ScalarWave.cpp InviscidVortex.cpp)
target_include_directories(shoccs-system PUBLIC ..)
target_link_libraries(shoccs-system PUBLIC fields shoccs-io shoccs-operators spdlog::spdlog)
