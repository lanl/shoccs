#pragma once

#include "scalar.hpp"
#include "tuple.hpp"

namespace ccs
{

namespace detail
{
template <Scalar X, Scalar Y, Scalar Z>
using vector = tuple<X, Y, Z>;
}

template <typename>
struct is_vector : std::false_type {
};

template <typename X, typename Y, typename Z>
struct is_vector<detail::vector<X, Y, Z>> : std::true_type {
};

template <typename T>
concept Vector = is_vector<std::remove_cvref_t<T>>::value;

template <typename T>
using vector = detail::vector<scalar<T>, scalar<T>, scalar<T>>;

using vector_span = vector<std::span<real>>;
using vector_view = vector<std::span<const real>>;

} // namespace ccs
