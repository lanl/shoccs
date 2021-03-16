#pragma once

#include <concepts>
#include <type_traits>

#include "Scalar_fwd.hpp"

namespace ccs::field::tuple
{

namespace detail
{
template <traits::ScalarType X, traits::ScalarType Y, traits::ScalarType Z>
using Vector = Tuple<X, Y, Z>;
}

namespace traits
{
template <typename>
struct is_vector : std::false_type {
};

template <typename X, typename Y, typename Z>
struct is_vector<tuple::detail::Vector<X, Y, Z>> : std::true_type {
};

template <typename T>
concept VectorType = is_vector<std::remove_cvref_t<T>>::value;

} // namespace traits

} // namespace ccs::field::tuple

namespace ccs::field
{

template <typename T>
using Vector = tuple::detail::Vector<Scalar<T>, Scalar<T>, Scalar<T>>;

using VectorView_Mutable = Vector<std::span<real>>;
using VectorView_Const = Vector<std::span<const real>>;
} // namespace ccs::field