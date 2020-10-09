#pragma once

#include "types.hpp"

#include "scalar_field_fwd.hpp"
#include "vector_field_fwd.hpp"

namespace ccs
{
template <typename T, int I>
struct scalar;

namespace detail
{
template <typename S, typename U>
struct scalar_proxy;

namespace traits
{
template <typename = void>
struct is_scalar_proxy : std::false_type {
};
template <typename T, typename U>
struct is_scalar_proxy<scalar_proxy<T, U>> : std::true_type {
};
} // namespace traits
} // namespace detail

namespace traits
{
template <typename = void>
struct is_scalar : std::false_type {
};
template <typename T, int I>
struct is_scalar<scalar<T, I>> : std::true_type {
};
} // namespace traits

template <typename U>
concept Scalar = traits::is_scalar<std::remove_cvref_t<U>>::value ||
                 detail::traits::is_scalar_proxy<std::remove_cvref_t<U>>::value;
} // namespace ccs