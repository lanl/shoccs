#include "field_io.hpp"
#include "fields/scalar.hpp"

#include <catch2/catch_test_macros.hpp>

#include <ranges>
#include <sol/sol.hpp>

using namespace ccs;

using U = std::span<const mesh_object_info>;
using T = std::array<U, 3>;

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

    auto io_opt = field_io::from_lua(lua["simulation"]);
    REQUIRE(!!io_opt);
    auto& io = *io_opt;

    REQUIRE(!io.write(names, std::vector<scalar_view>{}, step, 0.1, T{}));
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

    // Allocate 2 scalars as plain vectors (d_size=24, no Rx/Ry/Rz).
    std::vector<real> u_d(24), v_d(24);

    scalar_span u{u_d, {}, {}, {}};
    scalar_span v{v_d, {}, {}, {}};

    std::ranges::copy(std::views::iota(0, 24) | std::views::transform([](int i) {
                          return static_cast<real>(i);
                      }),
                      u.D.begin());
    std::ranges::copy(std::views::iota(24, 48) | std::views::transform([](int i) {
                          return static_cast<real>(i);
                      }),
                      v.D.begin());

    std::vector<scalar_view> io_scalars{u, v};
    REQUIRE(io.write(names, io_scalars, step, 0.0, T{}));
}
