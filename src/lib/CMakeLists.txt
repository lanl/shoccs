add_library(shoccs-run_sol run_from_sol.cpp)
set_target_properties(shoccs-run_sol PROPERTIES
 PUBLIC_HEADER "shoccs.hpp;../shoccs_config.hpp"
)
target_include_directories(shoccs-run_sol 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/shoccs>)
target_link_libraries(shoccs-run_sol PUBLIC lua sol2::sol2 PRIVATE shoccs-simulation)

set_target_properties(shoccs-run_sol PROPERTIES OUTPUT_NAME "shoccs")
set_target_properties(shoccs-run_sol PROPERTIES EXPORT_NAME shoccs)

install(TARGETS shoccs-run_sol
    EXPORT shoccs
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/shoccs
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)


