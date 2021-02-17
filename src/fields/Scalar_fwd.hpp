#pragma once

#include <concepts>
#include <type_traits>

#include "Tuple_fwd.hpp"

namespace ccs::mesh
{
struct Location;
}

namespace ccs::field::tuple
{

template <traits::OneTuple U, traits::ThreeTuple V>
class Scalar;

template <typename U, typename V>
Scalar(U&&, V&&) -> Scalar<U, V>;

template <typename U, typename V>
Scalar(const mesh::Location*, U&&, V&&) -> Scalar<U, V>;

namespace traits
{
template <typename>
struct is_Scalar : std::false_type {
};

template <OneTuple U, ThreeTuple V>
struct is_Scalar<Scalar<U, V>> : std::true_type {
};

template <typename T>
concept ScalarType = is_Scalar<std::remove_cvref_t<T>>::value;

} // namespace traits

// Specialize from_view for ScalarType
template <traits::ScalarType U>
struct from_view<U> {
    static constexpr auto create = [](auto&& u, auto&&... args) {
        return Scalar{u.location(), FWD(args)...};
    };
};

// Combination views
template <traits::ScalarType U, traits::ScalarType V>
struct from_view<U, V> {
    static constexpr auto create = [](auto&& u, auto&&, auto&&... args) {
        return Scalar{u.location(), FWD(args)...};
    };
};

} // namespace ccs::field::tuple

namespace ccs::field
{
using tuple::Scalar;

template <typename T>
using SimpleScalar = Scalar<Tuple<T>, Tuple<T, T, T>>;
} // namespace ccs::field
