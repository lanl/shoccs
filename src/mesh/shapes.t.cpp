#include "shapes.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("sphere")
{
    using namespace ccs;

    auto s = make_sphere(0, std::array{1.0, 1.0, 1.0}, 0.5);

    for (int i = 0; i < 3; i++) {
        real3 direction{};
        real3 origin{1.0, 1.0, 1.0};

        direction[i] = 1;
        origin[i] = 0.0;

        // test miss
        auto hit = s.hit(ray{.origin = origin, .direction = direction}, 0.0, 0.4);
        REQUIRE(!hit);

        // test hit from outside
        hit = s.hit(ray{.origin = origin, .direction = direction}, 0.0, 0.5 + 1e-14);
        REQUIRE(hit);

        REQUIRE(hit->t == Catch::Approx(0.5));
        real3 pos = {1.0, 1.0, 1.0};
        pos[i] = 0.5;
        REQUIRE(hit->position[0] == Catch::Approx(pos[0]));
        REQUIRE(hit->position[1] == Catch::Approx(pos[1]));
        REQUIRE(hit->position[2] == Catch::Approx(pos[2]));
        REQUIRE(hit->ray_outside);
        REQUIRE(hit->shape_id == 0);

        // test hit from inside
        hit = s.hit(ray{.origin = origin, .direction = direction}, hit->t + 1e-14, 100.0);
        REQUIRE(hit);

        REQUIRE(hit->t == Catch::Approx(1.5));
        pos[i] = 1.5;
        REQUIRE(hit->position[0] == Catch::Approx(pos[0]));
        REQUIRE(hit->position[1] == Catch::Approx(pos[1]));
        REQUIRE(hit->position[2] == Catch::Approx(pos[2]));
        REQUIRE(!hit->ray_outside);
        REQUIRE(hit->shape_id == 0);
    }

    auto s2 = make_sphere(0, real3{0.0, 0.0, 0.0}, 1.0);
    {
        auto n = s2.normal(real3{1.0, 0.0, 0.0});
        REQUIRE(n[0] == Catch::Approx(1.0));
    }
    {
        auto n = s2.normal(real3{0.0, 1.0, 0.0});
        REQUIRE(n[1] == Catch::Approx(1.0));
    }
    {
        auto n = s2.normal(real3{0.0, 0.0, -1.0});
        REQUIRE(n[2] == Catch::Approx(-1.0));
    }
}

TEST_CASE("xy_rect")
{
    using namespace ccs;

    real3 c0{0.0, 1.0, 2.0};
    real3 c1{1.0, 2.0, 2.0};

    SECTION("IN")
    {
        auto s = make_xy_rect(0, c0, c1, 1);

        real3 origin{0.5, 1.1, 0.0};
        real3 direction{0, 0, 1};

        auto hit = s.hit(ray{.origin = origin, .direction = direction}, 0.0, 10.0);
        REQUIRE(hit);
        REQUIRE(hit->t == Catch::Approx(2.0));
        real3 pos = {0.5, 1.1, 2.0};
        REQUIRE(hit->position[0] == Catch::Approx(pos[0]));
        REQUIRE(hit->position[1] == Catch::Approx(pos[1]));
        REQUIRE(hit->position[2] == Catch::Approx(pos[2]));
        REQUIRE(!hit->ray_outside);
        REQUIRE(hit->shape_id == 0);
    }

    SECTION("OUT")
    {
        auto s = make_xy_rect(0, c0, c1, -1);

        real3 origin{0.5, 1.1, 0.0};
        real3 direction{0, 0, 1};

        auto hit = s.hit(ray{.origin = origin, .direction = direction}, 0.0, 10.0);
        REQUIRE(hit);
        REQUIRE(hit->t == Catch::Approx(2.0));
        real3 pos = {0.5, 1.1, 2.0};
        REQUIRE(hit->position[0] == Catch::Approx(pos[0]));
        REQUIRE(hit->position[1] == Catch::Approx(pos[1]));
        REQUIRE(hit->position[2] == Catch::Approx(pos[2]));
        REQUIRE(hit->ray_outside);
        REQUIRE(hit->shape_id == 0);
    }
}

TEST_CASE("yz_rect")
{
    using namespace ccs;

    real3 c0{2.0, 0.0, 1.0};
    real3 c1{2.0, 1.0, 3.0};

    SECTION("IN")
    {
        auto s = make_yz_rect(0, c0, c1, 1);

        real3 origin{0.0, 0.5, 1.1};
        real3 direction{1, 0, 0};

        auto hit = s.hit(ray{.origin = origin, .direction = direction}, 0.0, 10.0);
        REQUIRE(hit);
        REQUIRE(hit->t == Catch::Approx(2.0));
        real3 pos = {2.0, 0.5, 1.1};
        REQUIRE(hit->position[0] == Catch::Approx(pos[0]));
        REQUIRE(hit->position[1] == Catch::Approx(pos[1]));
        REQUIRE(hit->position[2] == Catch::Approx(pos[2]));
        REQUIRE(!hit->ray_outside);
        REQUIRE(hit->shape_id == 0);
    }

    SECTION("OUT")
    {
        auto s = make_yz_rect(0, c0, c1, -1);

        real3 origin{0.0, 0.5, 1.1};
        real3 direction{1, 0, 0};

        auto hit = s.hit(ray{.origin = origin, .direction = direction}, 0.0, 10.0);
        REQUIRE(hit);
        REQUIRE(hit->t == Catch::Approx(2.0));
        real3 pos = {2.0, 0.5, 1.1};
        REQUIRE(hit->position[0] == Catch::Approx(pos[0]));
        REQUIRE(hit->position[1] == Catch::Approx(pos[1]));
        REQUIRE(hit->position[2] == Catch::Approx(pos[2]));
        REQUIRE(hit->ray_outside);
        REQUIRE(hit->shape_id == 0);
    }
}
