#include <cxxopts.hpp>
#include <iostream>
#include <sol/sol.hpp>
#include <string>

#include "lib/run_from_sol.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

int main(int argc, char* argv[])
{

    cxxopts::Options options(
        "shoccs", "Run the Stable High-Order Cut-Cell Solver with a given input");

    // clang-format off
    options.add_options()
        ("input-file", "Main lua input file", cxxopts::value<std::string>())
        ("script", "Supplementary lua options take precedence over file",
             cxxopts::value<std::string>())
        ("check", "Check input file for errors and return",
            cxxopts::value<bool>()->default_value("false"))
        ("help", "Print usage");
    // clang-format on
    options.parse_positional("input-file");

    auto result = options.parse(argc, argv);

    if (result.count("help") || result.arguments().size() == 0) {
        std::cout << options.help() << '\n';
        return 0;
    }

    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);

    // process main input file
    if (result.count("input-file")) {
        lua.script_file(result["input-file"].as<std::string>());
    }

    // run any extra setup associated with user script
    if (result.count("script")) {
        lua.script(result["script"].as<std::string>());
    }

    // We alway check the input but exit early if --check has been specified
    
    if (result.count("check")) {
        return 0;
    }
    // do some registry and activate loggers for now
    spdlog::info("Hello, {}!", "world");
    auto console = spdlog::stdout_color_st("system");

    ccs::simulation_run(lua["simulation"]);


}