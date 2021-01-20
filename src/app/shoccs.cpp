#include <cxxopts.hpp>
#include <iostream>
#include <sol/sol.hpp>
#include <string>

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

    if (result.count("help") || argc == 0) {
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


}