#include "field_io.hpp"
#include <numeric>

#include <catch2/catch_test_macros.hpp>

#include <range/v3/view/iota.hpp>
#include <sol/sol.hpp>

using namespace ccs;

using U = std::span<const mesh_object_info>;
using T = tuple<U, U, U>;

TEST_CASE("field_io - default")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {31},
                domain_bounds = {2}
            }
        }
    )");
    step_controller step{};
    std::vector<std::string> names{};
    field f{};

    auto io_opt = field_io::from_lua(lua["simulation"]);
    REQUIRE(!!io_opt);
    auto& io = *io_opt;

    REQUIRE(!io.write(names, f, step, 0.1, T{}));
}

// For this test, one needs to load the output in paraview
TEST_CASE("field_io - data")
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            mesh = {
                index_extents = {2, 3, 4},
                domain_bounds = {
                    min = {0.1, 0.2, 0.3},
                    max = {0.3, 0.5, 0.7}
                }
            },
            io = {
                write_every_step = 1,
                dir = "io_test"
            }
        }
    )");

    auto io_opt = field_io::from_lua(lua["simulation"]);
    REQUIRE(!!io_opt);
    auto& io = *io_opt;

    step_controller step{};
    std::vector<std::string> names{"U", "V"};
    field f{system_size{2, 0, scalar<integer>{tuple{24}, tuple{0, 0, 0}}}};

    auto&& [u, v] = f.scalars(0, 1);

    u | sel::D = vs::iota(0, 24);
    v | sel::D = vs::iota(24, 48);

    REQUIRE(io.write(names, f, step, 0.0, T{}));
}
