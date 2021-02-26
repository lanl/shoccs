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
template <traits::ThreeTuple U, traits::ThreeTuple V>
class Vector;

template <typename U, typename V>
Vector(U&&, V&&) -> Vector<U, V>;

template <typename U, typename V>
Vector(const mesh::Location*, U&&, V&&) -> Vector<U, V>;

namespace traits
{
template <typename>
struct is_Vector : std::false_type {
};

template <ThreeTuple U, ThreeTuple V>
struct is_Vector<Vector<U, V>> : std::true_type {
};

template <typename T>
concept VectorType = is_Vector<std::remove_cvref_t<T>>::value;

} // namespace traits

// Specialize from_view for VectorType
template <traits::VectorType U>
struct from_view<U> {
    static constexpr auto create = [](auto&& u, auto&&... args) {
        return Vector{u.location(), FWD(args)...};
    };
};

// Combination views
template <traits::VectorType U, traits::VectorType V>
struct from_view<U, V> {
    static constexpr auto create = [](auto&& u, auto&&, auto&&... args) {
        return Vector{u.location(), FWD(args)...};
    };
};

} // namespace ccs::field::tuple

namespace ccs::field
{
using tuple::Vector;

template <typename T>
using SimpleVector = Vector<Tuple<T, T, T>, Tuple<T, T, T>>;

using VectorView_Mutable = SimpleVector<std::span<real>>;
using VectorView_Const = SimpleVector<std::span<const real>>;
} // namespace ccs::field