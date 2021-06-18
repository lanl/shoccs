#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "xdmf.hpp"
#include <filesystem>
#include <string>
#include <fstream>

namespace fs = std::filesystem;

using namespace ccs;

TEST_CASE("header")
{

    std::vector<std::string> var_names{"U, V"};
    std::vector<std::string> file_names{"U.00000", "V.0000"};
    auto ix = index_extents{.extents = {3, 4, 5}};
    auto dom = domain_extents{};
    auto tmp = fs::temp_directory_path() / "headertest.xmf";
    auto st = std::fstream{tmp};

    auto ms = xdmf::write(st, ix, dom, 0, 0.0, var_names, file_names);
}
