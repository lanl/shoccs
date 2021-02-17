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

} // namespace ccs::field::tuple

namespace ccs::field
{
using tuple::Vector;

template <typename T>
using SimpleVector = Vector<Tuple<T, T, T>, Tuple<T, T, T>>;
} // namespace ccs::field