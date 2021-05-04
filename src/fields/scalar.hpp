#pragma once

#include "tuple.hpp"

namespace ccs
{
namespace detail
{

template <OneTuple U, ThreeTuple V>
using scalar = tuple<U, V>;
}

template <typename>
struct is_scalar : std::false_type {
};

template <OneTuple U, ThreeTuple V>
struct is_scalar<detail::scalar<U, V>> : std::true_type {
};

template <typename T>
concept Scalar = is_scalar<std::remove_cvref_t<T>>::value;

template <typename T>
using scalar = detail::scalar<tuple<T>, tuple<T, T, T>>;

using scalar_span = scalar<std::span<real>>;
using scalar_view = scalar<std::span<const real>>;

} // namespace ccs
