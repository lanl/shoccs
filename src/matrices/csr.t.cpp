#include "csr.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "random/random.hpp"
#include <vector>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/algorithm/shuffle.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/generate_n.hpp>

using namespace ccs;
using Catch::Matchers::Approx;
using T = std::vector<real>;

constexpr auto random_vec = [](integer n) {
    return vs::generate_n([]() { return pick(); }, n) | rs::to<T>();
};

TEST_CASE("Identity")
{
    T w{1, 1, 1, 1, 1};
    std::vector<integer> v{0, 1, 2, 3, 4};
    std::vector<integer> u{0, 1, 2, 3, 4, 5};

    const matrix::csr A{w, v, u};

    const T x = random_vec(A.rows());

    T b(x.size());
    A(x, b); // auto res = mat * rhs;

    REQUIRE_THAT(x, Approx(b));
}

TEST_CASE("Random")
{

    T w{6.132558989050928,
        -0.4611523807581932,
        -2.874686661596037,
        9.42084557206411,
        0.2298026797436883,
        6.066446959605997,
        -7.70721485928825,
        -0.9885546582519957,
        -5.302176517574914};

    std::vector<int> v{1, 6, 0, 4, 6, 7, 8, 9, 0};
    std::vector<int> u{0, 2, 3, 4, 4, 4, 4, 8, 8, 8, 9};

    const matrix::csr A{w, v, u};

    const T x{-3.612622416000683,
              -1.1427601879942273,
              0.8552565615441736,
              -4.755547850496647,
              -9.711932690401355,
              -7.232904766266888,
              -5.437050079342718,
              -2.4092054560513105,
              3.557741982344453,
              5.127028269794991};
    T b(x.size());
    A(x, b);

    REQUIRE_THAT(b,
                 Approx(T{-4.500735674823108,
                          10.385157472660014,
                          -91.49461808255228,
                          0.,
                          0.,
                          0.,
                          -48.353395342996556,
                          0.,
                          0.,
                          19.15476174098357}));
}

TEST_CASE("Identity Builder")
{

    constexpr integer vsize = 100;

    using P = matrix::csr::builder::pts;
    auto pts = std::vector<P>(vsize);
    for (integer i = 0; auto&& p : pts) {
        p = P{i, i, 1.0};
        ++i;
    }

    const T x = random_vec(vsize);

    for (int j = 0; j < 10; j++) {
        auto bld = matrix::csr::builder();

        ranges::shuffle(pts);
        for (auto&& [r, c, v] : pts) bld.add_point(r, c, v);

        const auto A = bld.to_csr(vsize);

        T b(x.size());

        A(x, b);

        REQUIRE_THAT(x, Approx(b));
    }
}

TEST_CASE("Random Builder")
{
    using P = matrix::csr::builder::pts;
    auto pts = std::vector<P>{{0, 1, 6.132558989050928},
                              {0, 6, -0.4611523807581932},
                              {1, 0, -2.874686661596037},
                              {2, 4, 9.42084557206411},
                              {6, 6, 0.2298026797436883},
                              {6, 7, 6.066446959605997},
                              {6, 8, -7.70721485928825},
                              {6, 9, -0.9885546582519957},
                              {9, 0, -5.302176517574914}};

    const T x{-3.612622416000683,
              -1.1427601879942273,
              0.8552565615441736,
              -4.755547850496647,
              -9.711932690401355,
              -7.232904766266888,
              -5.437050079342718,
              -2.4092054560513105,
              3.557741982344453,
              5.127028269794991};

    const T exact{-4.500735674823108,
                  10.385157472660014,
                  -91.49461808255228,
                  0.,
                  0.,
                  0.,
                  -48.353395342996556,
                  0.,
                  0.,
                  19.15476174098357};

    for (int j = 0; j < 10; j++) {
        auto builder = matrix::csr::builder();

        ranges::shuffle(pts);
        for (auto&& [r, c, v] : pts) builder.add_point(r, c, v);

        const auto A = builder.to_csr(10);
        T b(x.size());

        A(x, b);

        REQUIRE_THAT(b, Approx(exact));
    }
}
