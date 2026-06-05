#include "handle.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace ccs;

// ---------------------------------------------------------------------------
// Test layout: field_layout<2, 1> with 2 active scalars, 1 active vector.
// Total buffers = 2*4 + 1*12 = 20.
// ---------------------------------------------------------------------------

static constexpr auto layout_2_1 = field_layout<2, 1>{.n_scalars = 2, .n_vectors = 1};

// Handles built via consteval factories (compilation itself proves validity).
static constexpr auto s0 = make_scalar_handle(layout_2_1, 0);
static constexpr auto s1 = make_scalar_handle(layout_2_1, 1);
static constexpr auto v0 = make_vector_handle(layout_2_1, 0);

// ---------------------------------------------------------------------------
// field_layout arithmetic
// ---------------------------------------------------------------------------

TEST_CASE("field_layout arithmetic")
{
    SECTION("layout <1,0>")
    {
        constexpr auto l = field_layout<1, 0>{};
        STATIC_REQUIRE(l.total_buffers == 4);
        STATIC_REQUIRE(l.vector_base == 4);
    }

    SECTION("layout <2,1>")
    {
        STATIC_REQUIRE(layout_2_1.total_buffers == 20);
        STATIC_REQUIRE(layout_2_1.vector_base == 8);
    }

    SECTION("layout <4,4>")
    {
        constexpr auto l = field_layout<4, 4>{};
        STATIC_REQUIRE(l.total_buffers == 4 * 4 + 4 * 12);  // 16 + 48 = 64
        STATIC_REQUIRE(l.vector_base == 16);
    }
}

// ---------------------------------------------------------------------------
// scalar_handle accessors
// ---------------------------------------------------------------------------

TEST_CASE("scalar_handle accessors")
{
    SECTION("scalar 0")
    {
        REQUIRE(s0.D().id == 0);
        REQUIRE(s0.Rx().id == 1);
        REQUIRE(s0.Ry().id == 2);
        REQUIRE(s0.Rz().id == 3);
    }

    SECTION("scalar 1")
    {
        REQUIRE(s1.D().id == 4);
        REQUIRE(s1.Rx().id == 5);
        REQUIRE(s1.Ry().id == 6);
        REQUIRE(s1.Rz().id == 7);
    }

    SECTION("all() returns D, Rx, Ry, Rz in order")
    {
        auto a = s0.all();
        REQUIRE(a[0].id == 0);
        REQUIRE(a[1].id == 1);
        REQUIRE(a[2].id == 2);
        REQUIRE(a[3].id == 3);
    }

    SECTION("R() returns Rx, Ry, Rz")
    {
        auto r = s0.R();
        REQUIRE(r[0].id == 1);
        REQUIRE(r[1].id == 2);
        REQUIRE(r[2].id == 3);
    }
}

// ---------------------------------------------------------------------------
// vector_handle accessors
// ---------------------------------------------------------------------------

TEST_CASE("vector_handle accessors")
{
    // vector_base = 2*4 = 8 for layout<2,1>.
    SECTION("component scalar_handles")
    {
        REQUIRE(v0.x().base == 8);
        REQUIRE(v0.y().base == 12);
        REQUIRE(v0.z().base == 16);
    }

    SECTION("domain buffers")
    {
        REQUIRE(v0.Dx().id == 8);
        REQUIRE(v0.Dy().id == 12);
        REQUIRE(v0.Dz().id == 16);
    }

    SECTION("x boundary buffers")
    {
        REQUIRE(v0.xRx().id == 9);
        REQUIRE(v0.xRy().id == 10);
        REQUIRE(v0.xRz().id == 11);
    }

    SECTION("y boundary buffers")
    {
        REQUIRE(v0.yRx().id == 13);
        REQUIRE(v0.yRy().id == 14);
        REQUIRE(v0.yRz().id == 15);
    }

    SECTION("z boundary buffers")
    {
        REQUIRE(v0.zRx().id == 17);
        REQUIRE(v0.zRy().id == 18);
        REQUIRE(v0.zRz().id == 19);
    }
}

// ---------------------------------------------------------------------------
// make_scalar_handle / make_vector_handle consteval validation
// ---------------------------------------------------------------------------

TEST_CASE("consteval factory validation")
{
    SECTION("make_scalar_handle produces correct base values")
    {
        STATIC_REQUIRE(s0.base == 0);
        STATIC_REQUIRE(s1.base == 4);
    }

    SECTION("make_vector_handle produces correct base value")
    {
        STATIC_REQUIRE(v0.base == 8);
    }
}
