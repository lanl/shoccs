#include "field.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

using namespace ccs;

TEST_CASE("concepts") {
    REQUIRE(Field<field>);
    REQUIRE(Field<field_view>);
    REQUIRE(Field<field_span>);
    REQUIRE(!Field<scalar_span>);

    REQUIRE(OutputField<field, real>);
    REQUIRE(OutputField<field_span, int>);
    REQUIRE(!OutputField<field_view, real>);

    REQUIRE(OutputField<field, field_view>);
    REQUIRE(!OutputField<field_view, field>);
}

TEST_CASE("construction")
{
    // default construction
    auto x = field{};

    // sized construction
    auto y_size = system_size{2, 0, tuple{tuple{10}, tuple{2, 4, 5}}};
    auto y = field{y_size};

    auto&& [u, v] = y.scalars(0, 1);
    REQUIRE(rs::size(u | sel::D) == 10u);
    REQUIRE(rs::size(v | sel::D) == 10u);
    REQUIRE(rs::size(u | sel::Rx) == 2u);
    REQUIRE(rs::size(v | sel::Ry) == 4u);
    REQUIRE(rs::size(u | sel::Rz) == 5u);

   
}
