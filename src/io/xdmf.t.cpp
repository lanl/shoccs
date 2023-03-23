#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "xdmf.hpp"
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

using namespace ccs;

TEST_CASE("header")
{

    using U = std::span<const mesh_object_info>;
    using T = tuple<U, U, U>;
    std::vector<std::string> var_names{"U", "V"};
    std::vector<std::string> file_names{"U.00000", "V.00000"};
    auto ix = index_extents{.extents = {3, 4, 5}};
    auto dom = domain_extents{.min = {0.1, 0.2, 0.3}, .max = {1.1, 1.2, 1.3}};
    auto tmp = fs::temp_directory_path() / "headertest.xmf";
    auto logger = logs{"", true, "field_io"};

    auto writer = xdmf{tmp, ix, dom};
    T t{};
    REQUIRE(rs::size(get<0>(t)) == 0);

    writer.write(0, 0.0, var_names, file_names, T{}, logger);

    file_names[0] = "U.00001";
    file_names[1] = "V.00001";
    writer.write(1, 0.1, var_names, file_names, T{}, logger);
}
