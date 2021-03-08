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
    // REQUIRE(traits::OutputTuple<std::tuple<std::tuple<T>, std::tuple<T, T, T>>, real>);
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

TEST_CASE("FromRange")
{
    using namespace ccs;
    using namespace ccs::field::tuple;

    REQUIRE(traits::FromRange<std::vector<int>, std::span<int>>);
    REQUIRE(traits::FromRange<std::span<int>, std::vector<int>>);
    REQUIRE(!traits::FromRange<real, std::vector<int>>);
    REQUIRE(traits::FromRange<decltype(vs::iota(0, 10)), std::vector<int>>);
}

TEST_CASE("FromTuple")
{
    using namespace ccs;
    using namespace field::tuple::traits;

    static_assert(std::constructible_from<int, int>);
    static_assert(FromTuple<std::tuple<int>, std::tuple<std::vector<real>>>);
    static_assert(!FromTuple<std::tuple<void*>, std::tuple<std::vector<real>>>);
    using T = std::vector<real>;
    using U = decltype(vs::iota(0, 10));
    static_assert(FromRange<U, T>);
    static_assert(FromTuple<std::tuple<U, U>, std::tuple<T, T>>);
    static_assert(FromTuple<std::tuple<const U&>&, std::tuple<T>>);
}
