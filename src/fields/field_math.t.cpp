#include "field.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <range/v3/all.hpp>

using namespace ccs;
using T = std::vector<real>;

TEST_CASE("Output")
{
    system_size sz{2, 0, tuple{tuple{5}, tuple{1, 2, 3}}};

    auto x = field{sz};

    auto&& [a, b] = x.scalars(0, 1);

    a = 1;
    b = tuple{tuple{vs::iota(0, 5)},
              tuple{vs::iota(1, 2), vs::iota(2, 4), vs::iota(4, 7)}};

    x += 1;

    REQUIRE(a ==
            tuple{tuple{vs::repeat_n(2, 5)},
                  tuple{vs::repeat_n(2, 1), vs::repeat_n(2, 2), vs::repeat_n(2, 3)}});
    REQUIRE(b == tuple{tuple{vs::iota(1, 6)},
                       tuple{vs::iota(2, 3), vs::iota(3, 5), vs::iota(5, 8)}});

    auto y = field{sz};

    auto&& [c, d] = y.scalars(0, 1);

    c = -2,
    d = tuple{tuple{std::vector{-1, -2, -3, -4, -5}},
              tuple{std::vector{-2}, std::vector{-3, -4}, std::vector{-5, -6, -7}}};
    x += y;

    scalar<T> z{tuple{tuple{5}, tuple{1, 2, 3}}};
    REQUIRE(a == z);
    REQUIRE(b == z);
}

TEST_CASE("real math")
{
    system_size sz{2, 0, tuple{tuple{5}, tuple{1, 2, 3}}};

    auto x = field{sz};

    auto&& [a, b] = x.scalars(0, 1);

    a = 1;
    b = tuple{tuple{vs::iota(0, 5)},
              tuple{vs::iota(1, 2), vs::iota(2, 4), vs::iota(4, 7)}};

    auto y = x + 1;

    auto&& [c, d] = y.scalars(0, 1);
    REQUIRE(c ==
            tuple{tuple{vs::repeat_n(2, 5)},
                  tuple{vs::repeat_n(2, 1), vs::repeat_n(2, 2), vs::repeat_n(2, 3)}});
    REQUIRE(d == tuple{tuple{vs::iota(1, 6)},
                       tuple{vs::iota(2, 3), vs::iota(3, 5), vs::iota(5, 8)}});

    field z{y};

    auto&& [e, f] = z.scalars(0, 1);
    REQUIRE(e == c);
    REQUIRE(f == d);

    auto q = 1 + y + z;
    auto r = 2 * x + 3;
    auto&& [g, h] = q.scalars(0, 1);
    auto&& [i, j] = r.scalars(0, 1);
    REQUIRE(g == i);
    REQUIRE(h == j);
}

TEST_CASE("span math")
{
    system_size sz{2, 0, tuple{tuple{5}, tuple{1, 2, 3}}};

    auto x_ = field{sz};
    field_span x{x_};

    auto&& [a, b] = x.scalars(0, 1);

    a = 1;
    b = tuple{tuple{vs::iota(0, 5)},
              tuple{vs::iota(1, 2), vs::iota(2, 4), vs::iota(4, 7)}};

    auto y = x * x + 1;

    field z{y};

    field xx{1 + z};

    x = 1 + z;

    auto&& [c, d] = xx.scalars(0, 1);

    REQUIRE(a == c);
    REQUIRE(b == d);
}
