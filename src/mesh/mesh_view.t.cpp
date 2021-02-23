#include "mesh_view.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fields/result_field.hpp"
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/zip.hpp>

TEST_CASE("location-2D")
{
    using namespace ccs;

    auto m = mesh::Cartesian{real3{-1, 1}, real3{0, 2}, int3{2, 3, 1}};

    SECTION("X")
    {
        std::vector<real3> c{
            {-1, 1, 0}, {0, 1, 0}, {-1, 1.5, 0}, {0, 1.5, 0}, {-1, 2, 0}, {0, 2, 0}};
        auto g = mesh::location_view<0>(m);
        for (auto&& [x, y] : vs::zip(c, g)) REQUIRE(x == y);
    }

    SECTION("Y")
    {
        std::vector<real3> c{
            {-1, 1, 0}, {-1, 1.5, 0}, {-1, 2, 0}, {0, 1, 0}, {0, 1.5, 0}, {0, 2, 0}};
        auto g = mesh::location_view<1>(m);
        for (auto&& [x, y] : vs::zip(c, g)) REQUIRE(x == y);
    }
}

TEST_CASE("location-3D")
{
    using namespace ccs;

    auto m = mesh::Cartesian{real3{-1, 1, 3}, real3{0, 2, 4}, int3{2, 2, 3}};

    SECTION("X")
    {
        std::vector<real3> c{{-1, 1, 3},
                             {0, 1, 3},
                             {-1, 1, 3.5},
                             {0, 1, 3.5},
                             {-1, 1, 4},
                             {0, 1, 4},
                             {-1, 2, 3},
                             {0, 2, 3},
                             {-1, 2, 3.5},
                             {0, 2, 3.5},
                             {-1, 2, 4},
                             {0, 2, 4}};
        auto g = mesh::location_view<0>(m);
        for (auto&& [x, y] : vs::zip(c, g)) REQUIRE(x == y);
    }

    SECTION("Y")
    {
        std::vector<real3> c{{-1, 1, 3},
                             {-1, 2, 3},
                             {-1, 1, 3.5},
                             {-1, 2, 3.5},
                             {-1, 1, 4},
                             {-1, 2, 4},
                             {0, 1, 3},
                             {0, 2, 3},
                             {0, 1, 3.5},
                             {0, 2, 3.5},
                             {0, 1, 4},
                             {0, 2, 4}};
        auto g = mesh::location_view<1>(m);
        for (auto&& [x, y] : vs::zip(c, g)) REQUIRE(x == y);
    }

    SECTION("Z")
    {
        std::vector<real3> c{{-1, 1, 3},
                             {-1, 1, 3.5},
                             {-1, 1, 4},
                             {-1, 2, 3},
                             {-1, 2, 3.5},
                             {-1, 2, 4},
                             {0, 1, 3},
                             {0, 1, 3.5},
                             {0, 1, 4},
                             {0, 2, 3},
                             {0, 2, 3.5},
                             {0, 2, 4}};
        auto r = location_view<2>(m) | rs::to<std::vector<real3>>();
        REQUIRE(c == r);
    }
}

TEST_CASE("plane")
{
    using namespace ccs;

    auto m = mesh::Cartesian{real3{-1, 1, 3}, real3{0, 2, 4}, int3{2, 2, 3}};

    SECTION("X")
    {
        auto x = mesh::location_view<0>(m, 0) | rs::to<std::vector<real3>>();

        REQUIRE(x == std::vector<real3>{{-1, 1, 3},
                                        {-1, 1, 3.5},
                                        {-1, 1, 4},
                                        {-1, 2, 3},
                                        {-1, 2, 3.5},
                                        {-1, 2, 4}});
        x = mesh::location_view<0>(m, -1) | rs::to<std::vector<real3>>();
        REQUIRE(
            x ==
            std::vector<real3>{
                {0, 1, 3}, {0, 1, 3.5}, {0, 1, 4}, {0, 2, 3}, {0, 2, 3.5}, {0, 2, 4}});
    }

    SECTION("Y")
    {
        auto y = mesh::location_view<1>(m, 0) | rs::to<std::vector<real3>>();
        REQUIRE(
            y ==
            std::vector<real3>{
                {-1, 1, 3}, {-1, 1, 3.5}, {-1, 1, 4}, {0, 1, 3}, {0, 1, 3.5}, {0, 1, 4}});
        y = mesh::location_view<1>(m, -1) | rs::to<std::vector<real3>>();
        REQUIRE(
            y ==
            std::vector<real3>{
                {-1, 2, 3}, {-1, 2, 3.5}, {-1, 2, 4}, {0, 2, 3}, {0, 2, 3.5}, {0, 2, 4}});
    }

    SECTION("Z")
    {
        auto z = mesh::location_view<2>(m, 0) | rs::to<std::vector<real3>>();
        REQUIRE(z == std::vector<real3>{{-1, 1, 3}, {-1, 2, 3}, {0, 1, 3}, {0, 2, 3}});
        z = mesh::location_view<2>(m, -1) | rs::to<std::vector<real3>>();
        REQUIRE(z == std::vector<real3>{{-1, 1, 4}, {-1, 2, 4}, {0, 1, 4}, {0, 2, 4}});
    }
}