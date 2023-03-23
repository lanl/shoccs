#include "logging.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace ccs;

TEST_CASE("logging")
{
    auto l = logs{};

    REQUIRE(!l);

    l = logs{false, "builder"};
    REQUIRE(!l);
    l(spdlog::level::info, "this message shant be seen");

    l = logs{"", true, "builder"};
    REQUIRE(l);

    l(spdlog::level::info, "building!");
}
