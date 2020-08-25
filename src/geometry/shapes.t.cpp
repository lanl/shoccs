#include "shapes.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("sphere")
{
    using namespace ccs;

    auto s = make_sphere(42, std::array{1.0, 1.0, 1.0}, 0.5);

    for (int i = 0; i < 3; i++) {
        real3 direction{};
        real3 origin{1.0, 1.0, 1.0};

        direction[i] = 1;
        origin[i] = 0.0;

        // test miss
        auto hit = s.hit(ray{.origin = origin, .direction = direction}, 0.0, 0.4);
        REQUIRE(!hit);

        //test hit from outside
        hit = s.hit(ray{.origin = origin, .direction = direction}, 0.0, 0.5 + 1e-14);
        REQUIRE(hit);

        REQUIRE(hit->t == Catch::Approx(0.5));
        real3 pos = {1.0, 1.0, 1.0};
        pos[i] = 0.5;
        REQUIRE(hit->position[0] == Catch::Approx(pos[0]));
        REQUIRE(hit->position[1] == Catch::Approx(pos[1]));
        REQUIRE(hit->position[2] == Catch::Approx(pos[2]));
        REQUIRE(hit->ray_outside);
        REQUIRE(hit->shape_id == 42);

        //test hit from inside
        hit = s.hit(ray{.origin = origin, .direction = direction}, hit->t+1e-14, 100.0);
        REQUIRE(hit);

        REQUIRE(hit->t == Catch::Approx(1.5));
        pos[i] = 1.5;
        REQUIRE(hit->position[0] == Catch::Approx(pos[0]));
        REQUIRE(hit->position[1] == Catch::Approx(pos[1]));
        REQUIRE(hit->position[2] == Catch::Approx(pos[2]));
        REQUIRE(!hit->ray_outside);
        REQUIRE(hit->shape_id == 42);
    }
}