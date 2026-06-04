#include "fields/expr.hpp"
#include "fields/field_registry.hpp"
#include "fields/handle.hpp"

#include <functional>

#include <Kokkos_Core.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace ccs;

// ---------------------------------------------------------------------------
// Custom main: Kokkos must be initialized before any test allocates Views.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

// ---------------------------------------------------------------------------
// 10.1a — Expression node type tests
// ---------------------------------------------------------------------------

TEST_CASE("handle_expr reads from pointer")
{
    real data[] = {10.0, 20.0, 30.0};
    handle_expr e{data};
    REQUIRE(e(0) == 10.0);
    REQUIRE(e(1) == 20.0);
    REQUIRE(e(2) == 30.0);
}

TEST_CASE("scalar_literal_expr returns constant for any index")
{
    scalar_literal_expr e{3.14};
    REQUIRE(e(0) == 3.14);
    REQUIRE(e(1) == 3.14);
    REQUIRE(e(99) == 3.14);
}

TEST_CASE("binary_expr with std::plus")
{
    real a[] = {1.0, 2.0, 3.0};
    real b[] = {10.0, 20.0, 30.0};
    binary_expr expr{std::plus<>{}, handle_expr{a}, handle_expr{b}};
    REQUIRE(expr(0) == 11.0);
    REQUIRE(expr(1) == 22.0);
    REQUIRE(expr(2) == 33.0);
}

TEST_CASE("unary_expr with std::negate")
{
    real a[] = {1.0, -2.0, 3.0};
    unary_expr expr{std::negate<>{}, handle_expr{a}};
    REQUIRE(expr(0) == -1.0);
    REQUIRE(expr(1) == 2.0);
    REQUIRE(expr(2) == -3.0);
}

TEST_CASE("nested expression: (a + b) * c")
{
    real a[] = {1.0, 2.0, 3.0};
    real b[] = {10.0, 20.0, 30.0};
    auto sum = binary_expr{std::plus<>{}, handle_expr{a}, handle_expr{b}};
    auto expr = binary_expr{std::multiplies<>{}, sum, scalar_literal_expr{2.0}};
    REQUIRE(expr(0) == 22.0);
    REQUIRE(expr(1) == 44.0);
    REQUIRE(expr(2) == 66.0);
}

TEST_CASE("contains_ptr detects aliasing")
{
    real a[] = {1.0};
    real b[] = {2.0};

    REQUIRE(contains_ptr(handle_expr{a}, a));
    REQUIRE_FALSE(contains_ptr(handle_expr{a}, b));
    REQUIRE_FALSE(contains_ptr(scalar_literal_expr{1.0}, a));

    auto sum = binary_expr{std::plus<>{}, handle_expr{a}, handle_expr{b}};
    REQUIRE(contains_ptr(sum, a));
    REQUIRE(contains_ptr(sum, b));
    real c[] = {3.0};
    REQUIRE_FALSE(contains_ptr(sum, c));

    auto neg = unary_expr{std::negate<>{}, handle_expr{a}};
    REQUIRE(contains_ptr(neg, a));
    REQUIRE_FALSE(contains_ptr(neg, b));
}

// ---------------------------------------------------------------------------
// 10.2a — assign() tests
// ---------------------------------------------------------------------------

TEST_CASE("assign with binary expression")
{
    constexpr int n = 100;
    Kokkos::View<real*, memory_space> a("a", n);
    Kokkos::View<real*, memory_space> b("b", n);
    Kokkos::View<real*, memory_space> dst("dst", n);

    for (int i = 0; i < n; ++i) {
        a(i) = static_cast<real>(i);
        b(i) = static_cast<real>(i * 10);
    }

    assign(dst.data(), n,
           binary_expr{std::plus<>{}, handle_expr{a.data()}, handle_expr{b.data()}});

    for (int i = 0; i < n; ++i) {
        REQUIRE(dst(i) == static_cast<real>(i + i * 10));
    }
}

TEST_CASE("assign with scalar_literal_expr")
{
    constexpr int n = 50;
    Kokkos::View<real*, memory_space> dst("dst", n);

    assign(dst.data(), n, scalar_literal_expr{7.0});

    for (int i = 0; i < n; ++i) {
        REQUIRE(dst(i) == 7.0);
    }
}

TEST_CASE("assign detects aliasing and stages through temporary")
{
    constexpr int n = 100;
    Kokkos::View<real*, memory_space> a("a", n);
    for (int i = 0; i < n; ++i) a(i) = static_cast<real>(i + 1);

    // dst[i] = dst[i] + dst[i] — aliased, must stage through temporary.
    assign(a.data(), n,
           binary_expr{std::plus<>{}, handle_expr{a.data()}, handle_expr{a.data()}});

    for (int i = 0; i < n; ++i) {
        REQUIRE(a(i) == static_cast<real>(2 * (i + 1)));
    }
}

TEST_CASE("assign without aliasing copies directly")
{
    constexpr int n = 100;
    Kokkos::View<real*, memory_space> src("src", n);
    Kokkos::View<real*, memory_space> dst("dst", n);

    for (int i = 0; i < n; ++i) {
        src(i) = static_cast<real>(i * 3);
    }

    assign(dst.data(), n, handle_expr{src.data()});

    for (int i = 0; i < n; ++i) {
        REQUIRE(dst(i) == static_cast<real>(i * 3));
    }
}

// ---------------------------------------------------------------------------
// Mutating operator tests
// ---------------------------------------------------------------------------

TEST_CASE("plus_assign: dst[i] += src[i]")
{
    constexpr int n = 40;
    Kokkos::View<real*, memory_space> dst_buf("dst", n);
    Kokkos::View<real*, memory_space> src_buf("src", n);
    auto* dst = dst_buf.data();
    auto* src = src_buf.data();
    for (int i = 0; i < n; ++i) {
        dst[i] = static_cast<real>(i);
        src[i] = static_cast<real>(i * 10);
    }

    plus_assign(dst, n, handle_expr{src});

    for (int i = 0; i < n; ++i) {
        REQUIRE(dst[i] == static_cast<real>(i + i * 10));
    }
}

TEST_CASE("minus_assign: dst[i] -= src[i]")
{
    constexpr int n = 40;
    Kokkos::View<real*, memory_space> dst_buf("dst", n);
    Kokkos::View<real*, memory_space> src_buf("src", n);
    auto* dst = dst_buf.data();
    auto* src = src_buf.data();
    for (int i = 0; i < n; ++i) {
        dst[i] = static_cast<real>(i * 10);
        src[i] = static_cast<real>(i);
    }

    minus_assign(dst, n, handle_expr{src});

    for (int i = 0; i < n; ++i) {
        REQUIRE(dst[i] == static_cast<real>(i * 10 - i));
    }
}

TEST_CASE("times_assign: dst[i] *= 2.0")
{
    constexpr int n = 40;
    Kokkos::View<real*, memory_space> dst_buf("dst", n);
    auto* dst = dst_buf.data();
    for (int i = 0; i < n; ++i) dst[i] = static_cast<real>(i + 1);

    times_assign(dst, n, scalar_literal_expr{2.0});

    for (int i = 0; i < n; ++i) {
        REQUIRE(dst[i] == static_cast<real>((i + 1) * 2));
    }
}

TEST_CASE("divide_assign: dst[i] /= 2.0")
{
    constexpr int n = 40;
    Kokkos::View<real*, memory_space> dst_buf("dst", n);
    auto* dst = dst_buf.data();
    for (int i = 0; i < n; ++i) dst[i] = static_cast<real>((i + 1) * 4);

    divide_assign(dst, n, scalar_literal_expr{2.0});

    for (int i = 0; i < n; ++i) {
        REQUIRE(dst[i] == static_cast<real>((i + 1) * 2));
    }
}

TEST_CASE("plus_assign aliasing safety: dst[i] += dst[i]")
{
    constexpr int n = 40;
    Kokkos::View<real*, memory_space> buf("buf", n);
    auto* a = buf.data();
    for (int i = 0; i < n; ++i) a[i] = static_cast<real>(i + 1);

    plus_assign(a, n, handle_expr{a});

    for (int i = 0; i < n; ++i) {
        REQUIRE(a[i] == static_cast<real>(2 * (i + 1)));
    }
}

TEST_CASE("times_assign_scalar: all 4 buffers *= constant")
{
    field_registry<2, 1, 0> reg;
    auto ref0 = reg.allocate_scalar(0, 0, 100, 5, 3, 2);
    constexpr auto sh = scalar_handle{0};

    for (int i = 0; i < 100; ++i) reg.view(ref0, sh.D())(i) = static_cast<real>(i + 1);
    for (int i = 0; i < 5; ++i) reg.view(ref0, sh.Rx())(i) = static_cast<real>(i + 1);
    for (int i = 0; i < 3; ++i) reg.view(ref0, sh.Ry())(i) = static_cast<real>(i + 1);
    for (int i = 0; i < 2; ++i) reg.view(ref0, sh.Rz())(i) = static_cast<real>(i + 1);

    times_assign_scalar(reg, ref0, sh, 3.14);

    for (int i = 0; i < 100; ++i)
        REQUIRE(reg.view(ref0, sh.D())(i) == static_cast<real>((i + 1) * 3.14));
    for (int i = 0; i < 5; ++i)
        REQUIRE(reg.view(ref0, sh.Rx())(i) == static_cast<real>((i + 1) * 3.14));
    for (int i = 0; i < 3; ++i)
        REQUIRE(reg.view(ref0, sh.Ry())(i) == static_cast<real>((i + 1) * 3.14));
    for (int i = 0; i < 2; ++i)
        REQUIRE(reg.view(ref0, sh.Rz())(i) == static_cast<real>((i + 1) * 3.14));
}
