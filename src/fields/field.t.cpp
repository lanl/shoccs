#include "field.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

using namespace ccs;

TEST_CASE("concepts")
{
    REQUIRE(Field<field>);
    REQUIRE(Field<field_view>);
    REQUIRE(Field<field_span>);
    REQUIRE(!Field<scalar_span>);

    REQUIRE(OutputField<field, real>);
    REQUIRE(OutputField<field_span, int>);
    REQUIRE(!OutputField<field_view, real>);

    REQUIRE(OutputField<field, field_view>);
    REQUIRE(!OutputField<field_view, field>);

    {
        using X = std::vector<real>;
        using Y = std::span<real>;
        static_assert(!std::constructible_from<X, Y>);
        static_assert(!std::constructible_from<Y, X>);
        static_assert(ConstructibleFromRange<X, Y>);
        static_assert(ConstructibleFromRange<Y, X>);

        static_assert(sizeof(scalar_real) != sizeof(scalar_span));

        using TP = scalar_real;
        using UP = scalar_span;
        static_assert(std::constructible_from<TP, UP>);
        static_assert(std::constructible_from<UP, TP>);

        using T = std::vector<TP>;
        using U = std::span<TP>;
        static_assert(ConstructibleFromRange<T, U>);
        static_assert(ConstructibleFromRange<U, T>); // fails
        // static_assert(std::constructible_from<T, U>); // fails
        static_assert(
            std::constructible_from<rs::range_value_t<T>, rs::range_value_t<U>>);
        static_assert(
            std::constructible_from<rs::range_value_t<U>, rs::range_value_t<T>>);

        // static_assert(std::constructible_from<U, T>);
        T t{};
        U u{rs::begin(t), rs::end(t)};
    }
}

TEST_CASE("construction")
{
    // default construction
    auto x = field{};

    using I = ssize_t;
    constexpr auto scalar_size = tuple{tuple<I>{10}, tuple<I, I, I>{2, 4, 5}};
    // sized construction
    auto y_size = system_size{2, 0, scalar_size};
    auto y = field{y_size};

    auto&& [u, v] = y.scalars(0, 1);
    REQUIRE(ssize(u) == ssize(v));
    REQUIRE(ssize(u) == scalar_size);
}

TEST_CASE("construction/conversion")
{
    // default construction
    auto x_size = system_size{2, 0, tuple{tuple{10}, tuple{2, 4, 5}}};
    auto x = field{x_size};

    // probe selector type for scalars

    auto&& [a, b] = x.scalars(0, 1);
    a = 1;
    b = 2;
    REQUIRE(a ==
            tuple{tuple{vs::repeat_n(1, 10)},
                  tuple{vs::repeat_n(1, 2), vs::repeat_n(1, 4), vs::repeat_n(1, 5)}});

    {
        field y{x};
        auto&& [c, d] = y.scalars(0, 1);
        REQUIRE(a == c);
        REQUIRE(b == d);
    }

    {
        field_span y{x};
        auto&& [a, b] = x.scalars(0, 1);
        auto&& [c, d] = y.scalars(0, 1);
        REQUIRE(a == c);
        REQUIRE(b == d);
        // ensure that we are not making copies by seeing if the change percolates
        // to original
        c = 3;
        REQUIRE(a == c);

        field z{y};
        auto&& [e, f] = z.scalars(0, 1);
        REQUIRE(e == a);
        REQUIRE(f == b);

        f = 100;
        REQUIRE(b == d);
        REQUIRE(!(f == b));

        field_view v{y};
        auto&& [g, h] = v.scalars(0, 1);
        REQUIRE(g == a);
        REQUIRE(h == b);
    }

    {
        field_view y{x};
        auto&& [a, b] = x.scalars(0, 1);
        auto&& [c, d] = y.scalars(0, 1);
        REQUIRE(a == c);
        REQUIRE(b == d);

        field z{y};
        auto&& [e, f] = z.scalars(0, 1);
        REQUIRE(e == a);
        REQUIRE(f == b);

        f = 100;
        REQUIRE(b == d);
        REQUIRE(!(f == b));
    }
}

TEST_CASE("assignment")
{
    // default construction
    auto x_size = system_size{2, 0, tuple{tuple{10}, tuple{2, 4, 5}}};
    auto x = field{x_size};

    auto integer_scalar = [](integer i) {
        return tuple{tuple{vs::repeat_n(i, 10)},
                     tuple{vs::repeat_n(i, 2), vs::repeat_n(i, 4), vs::repeat_n(i, 5)}};
    };

    x = 1;
    auto&& [a, b] = x.scalars(0, 1);
    REQUIRE(a == integer_scalar(1));
    REQUIRE(a == b);

    field y{x};
    y = 2;

    field_span z{x};
    z = y;
    REQUIRE(a == integer_scalar(2));
    z = -1;
    REQUIRE(b == integer_scalar(-1));

    auto f = [](field_span fs) {
        auto&& [u, v] = fs.scalars(1, 0);
        u = 10;
        v = 11;
    };
    x = f;
    REQUIRE(a == integer_scalar(11));
    REQUIRE(b == integer_scalar(10));
}
