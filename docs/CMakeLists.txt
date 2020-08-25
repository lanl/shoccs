# we need Sphinx as well myst_parser to build the docs

find_package(Sphinx)

if(Sphinx_FOUND)
    message(STATUS "Ensure sphinx extension 'myst_parser' is installed.")
    set (Sphinx_source ${CMAKE_CURRENT_SOURCE_DIR})
    set (Sphinx_build ${CMAKE_CURRENT_BINARY_DIR})
    # add docs as a make target
    add_custom_target(docs)
    add_custom_command(
        TARGET docs
        COMMAND ${SPHINX_EXECUTABLE} -b html ${Sphinx_source} ${Sphinx_build}
        WORKING_DIRECTORY ${Sphinx_build}
        COMMENT "Generating documentation with Sphinx")
else()
    message(STATUS "Unable to find suitable 'sphinx-build'.  Will not be able to 'make docs'.")
endif()