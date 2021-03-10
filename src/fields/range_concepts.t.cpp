#include "ContainerTuple.hpp"
#include "types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

#include <vector>

namespace ccs
{
template <typename T>
concept any_output_range = rs::range<T>&& rs::output_range<T, rs::range_value_t<T>>;
}

TEST_CASE("Output Ranges")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    REQUIRE(rs::output_range<std::vector<real>&, real>);
    REQUIRE(rs::output_range<std::vector<real>&, int>);
    REQUIRE(traits::AnyOutputRange<std::vector<real>&>);
    REQUIRE(traits::OutputRange<std::vector<real>&, real>);

    REQUIRE(!rs::output_range<const std::vector<real>&, real>);
    REQUIRE(!traits::AnyOutputRange<const std::vector<real>&>);

    REQUIRE(traits::AnyOutputRange<std::span<real>>);
    REQUIRE(
        rs::output_range<std::span<real>, rs::range_value_t<decltype(vs::iota(0, 10))>>);
    REQUIRE(!traits::AnyOutputRange<std::span<const real>>);
    REQUIRE(traits::OutputRange<std::span<real>, std::span<const real>>);
}

TEST_CASE("OutputTuple")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    using T = std::vector<real>;

    REQUIRE(traits::OutputTuple<std::tuple<T, T>, real>);
    REQUIRE(traits::OutputTuple<std::tuple<std::tuple<T>, std::tuple<T, T, T>>, real>);
    REQUIRE(traits::OutputTuple<std::tuple<T, T>, int>);
    REQUIRE(traits::OutputTuple<std::tuple<T&, T&>, T>);
    REQUIRE(!traits::OutputTuple<std::tuple<const T&, const T&>, T>);

    REQUIRE(traits::OutputTuple<std::tuple<std::span<real>>, T>);
    REQUIRE(
        !traits::OutputTuple<std::tuple<std::span<const real>>, std::span<const real>>);

    REQUIRE(traits::OutputTuple<std::tuple<std::span<real>, std::span<real>>,
                                std::tuple<const T&, const T&>>);
    REQUIRE(!traits::OutputTuple<std::tuple<const T&, const T&>,
                                 std::tuple<std::span<real>, std::span<real>>>);
}

TEST_CASE("Modify Containers from Views")
{
    using namespace ccs;

    auto x = std::vector{1, 2, 3};
    auto y = vs::all(x);

    REQUIRE(any_output_range<decltype(y)>);

    for (auto&& i : y) i *= 2;

    REQUIRE(rs::equal(x, std::vector{2, 4, 6}));

    {
        constexpr auto f = [](auto&& i) { return i; };
        auto a = x | vs::transform(f);
        auto b = y | vs::transform(f);
        static_assert(std::same_as<decltype(a), decltype(b)>);
    }
}

TEST_CASE("TupleLike")
{
    using namespace ccs::field::tuple;
    REQUIRE(traits::TupleLike<std::tuple<int, int>>);
    REQUIRE(traits::TupleLike<ContainerTuple<std::vector<int>>>);
}

TEST_CASE("NotTupleRanges")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    REQUIRE(traits::NonTupleRange<std::vector<real>>);
    REQUIRE(traits::NonTupleRange<std::vector<real>&>);
    REQUIRE(traits::NonTupleRange<const std::vector<real>&>);
    REQUIRE(traits::NonTupleRange<std::span<real>>);
    REQUIRE(traits::NonTupleRange<std::span<const real>>);
}

TEST_CASE("From")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    using T = std::vector<int>;
    using I = decltype(vs::iota(0, 10));
    using Z = decltype(vs::zip_with(std::plus{}, vs::iota(0, 10), vs::iota(1, 11)));

    REQUIRE(traits::is_constructible_from_range<std::span<const int>, T>::value);
    REQUIRE(traits::is_constructible_from_range<T, I>::value);
    REQUIRE(traits::is_constructible_from<std::span<const int>, const T&>::value);
    REQUIRE(traits::is_constructible_from<T, I>::value);
    REQUIRE(traits::is_constructible_from<T, Z>::value);

    REQUIRE(traits::TupleFromTuple<std::tuple<T>, std::tuple<I>>);
    REQUIRE(traits::TupleFromTuple<std::tuple<T, T>, std::tuple<Z, I>>);
    REQUIRE(traits::TupleFromTuple<std::tuple<std::tuple<T>, std::tuple<T, T>>,
                                   std::tuple<std::tuple<Z>, std::tuple<I, Z>>>);
}

TEST_CASE("tuple shape")
{
    using namespace ccs::field::tuple::traits;

    REQUIRE(SimilarTuples<std::tuple<int>, std::tuple<double>>);
    REQUIRE(SimilarTuples<std::tuple<std::tuple<int>>, std::tuple<std::tuple<void*>>>);
    REQUIRE(!SimilarTuples<std::tuple<std::tuple<int>>, std::tuple<void*>>);
    REQUIRE(
        SimilarTuples<std::tuple<std::tuple<int>, std::tuple<int, int, int>>,
                      std::tuple<std::tuple<void*>, std::tuple<void*, char, double>>>);
    REQUIRE(
        !SimilarTuples<std::tuple<std::tuple<int, int, int>, std::tuple<void*>>,
                       std::tuple<std::tuple<void*>, std::tuple<void*, char, double>>>);
}
