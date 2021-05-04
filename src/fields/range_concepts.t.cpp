#include "container_tuple.hpp"
#include "types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <range/v3/all.hpp>

#include <cstdlib>
#include <vector>

#include <iostream>

namespace ccs
{
template <typename T>
concept any_output_range = rs::range<T> && rs::output_range<T, rs::range_value_t<T>>;
}

using namespace ccs;

TEST_CASE("Output Ranges")
{
    REQUIRE(rs::output_range<std::vector<real>&, real>);
    REQUIRE(rs::output_range<std::vector<real>&, int>);
    REQUIRE(OutputRange<std::vector<real>&>);
    REQUIRE(OutputRange<std::vector<real>&, real>);

    REQUIRE(!rs::output_range<const std::vector<real>&, real>);
    REQUIRE(!OutputRange<const std::vector<real>&>);

    REQUIRE(OutputRange<std::span<real>>);
    REQUIRE(
        rs::output_range<std::span<real>, rs::range_value_t<decltype(vs::iota(0, 10))>>);
    REQUIRE(!OutputRange<std::span<const real>>);
    REQUIRE(OutputRange<std::span<real>, std::span<const real>>);
}

TEST_CASE("OutputTuple")
{
    using T = std::vector<real>;

    REQUIRE(OutputTuple<std::tuple<T, T>, real>);
    REQUIRE(OutputTuple<std::tuple<std::tuple<T>, std::tuple<T, T, T>>, real>);
    REQUIRE(OutputTuple<std::tuple<T, T>, int>);
    REQUIRE(OutputTuple<std::tuple<T&, T&>, T>);
    REQUIRE(!OutputTuple<std::tuple<const T&, const T&>, T>);

    REQUIRE(OutputTuple<std::tuple<std::span<real>>, T>);
    REQUIRE(!OutputTuple<std::tuple<std::span<const real>>, std::span<const real>>);

    REQUIRE(OutputTuple<std::tuple<std::span<real>, std::span<real>>,
                        std::tuple<const T&, const T&>>);
    REQUIRE(!OutputTuple<std::tuple<const T&, const T&>,
                         std::tuple<std::span<real>, std::span<real>>>);
}

TEST_CASE("Modify Containers from Views")
{
    auto x = std::vector{1, 2, 3};
    auto y = vs::all(x);

    REQUIRE(OutputRange<decltype(y)>);

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
    REQUIRE(TupleLike<std::tuple<int, int>>);
    REQUIRE(TupleLike<container_tuple<std::vector<int>>>);

    // take_exactly results in a custom range tuple.  Ensure we do not treat it as one of
    // ours.
    auto x = std::vector<int>(50);
    REQUIRE(!TupleLike<decltype(x | vs::take_exactly(5))>);
}

TEST_CASE("NotTupleRanges")
{
    REQUIRE(NonTupleRange<std::vector<real>>);
    REQUIRE(NonTupleRange<std::vector<real>&>);
    REQUIRE(NonTupleRange<const std::vector<real>&>);
    REQUIRE(NonTupleRange<std::span<real>>);
    REQUIRE(NonTupleRange<std::span<const real>>);
}

TEST_CASE("From")
{
    using T = std::vector<int>;
    using I = decltype(vs::iota(0, 10));
    using Z = decltype(vs::zip_with(std::plus{}, vs::iota(0, 10), vs::iota(1, 11)));

    REQUIRE(is_constructible_from_range<std::span<const int>, T>::value);
    REQUIRE(is_constructible_from_range<T, I>::value);
    REQUIRE(is_constructible_from<std::span<const int>, const T&>::value);
    REQUIRE(is_constructible_from<T, I>::value);
    REQUIRE(is_constructible_from<T, Z>::value);

    REQUIRE(TupleFromTuple<std::tuple<T>, std::tuple<I>>);
    REQUIRE(TupleFromTuple<std::tuple<T, T>, std::tuple<Z, I>>);
    REQUIRE(TupleFromTuple<std::tuple<std::tuple<T>, std::tuple<T, T>>,
                           std::tuple<std::tuple<Z>, std::tuple<I, Z>>>);
}

TEST_CASE("tuple shape")
{
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

TEST_CASE("levels")
{
    REQUIRE(tuple_levels_v<std::tuple<int>> == 1);
    REQUIRE(tuple_levels_v<std::tuple<int, double, float, char, void*>> == 1);
    REQUIRE(tuple_levels_v<std::tuple<std::tuple<int>>> == 2);
    REQUIRE(tuple_levels_v<std::tuple<std::tuple<int>,
                                      std::tuple<float, double, char>,
                                      std::tuple<void*>>> == 2);
}

TEST_CASE("view closures")
{
    using I = decltype(vs::transform([](auto&& i) { return i; }));
    REQUIRE(ViewClosure<I>);
    REQUIRE(ViewClosures<I>);
    REQUIRE(ViewClosures<std::tuple<I, I>>);
}

TEST_CASE("list index")
{
    using L = list_index<4, 5, 6>;
    REQUIRE(ListIndex<L>);
    static_assert(index_v<L, 0> == 4);
    static_assert(index_v<L, 1> == 5);
    static_assert(index_v<L, 2> == 6);

    using ListOfL = mp_list<list_index<0, 1, 2>, list_index<1>>;
    REQUIRE(ListIndices<ListOfL>);
}

TEST_CASE("viewable ranges")
{
    static_assert(std::same_as<viewable_range_by_value<std::span<int>>, std::span<int>>);
    static_assert(std::same_as<viewable_range_by_value<std::span<int>&>, std::span<int>>);
    static_assert(std::same_as<viewable_range_by_value<std::span<const int>>,
                               std::span<const int>>);
    static_assert(std::same_as<viewable_range_by_value<std::span<const int>&>,
                               std::span<const int>>);
    static_assert(
        std::same_as<viewable_range_by_value<std::vector<int>&>, std::vector<int>&>);
    static_assert(std::same_as<viewable_range_by_value<const std::vector<int>&>,
                               const std::vector<int>&>);
}

TEST_CASE("NumericTuple")
{
    REQUIRE(NumericTuple<real3>);
    REQUIRE(NumericTuple<std::tuple<real, int>>);
    REQUIRE(NumericTuple<std::tuple<real&, const int&>>);
    REQUIRE(!NumericTuple<std::tuple<std::vector<int>>>);
    using T = rs::common_tuple<const int&, const int&, const int&>;
    REQUIRE(TupleLike<T>);
    REQUIRE(NumericTuple<rs::common_tuple<const int&, const int&, const int&>>);
}
