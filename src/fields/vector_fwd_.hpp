#pragma once

#include "types.hpp"

#include "scalar_field_fwd.hpp"
#include "vector_field_fwd.hpp"

namespace ccs
{
template <typename T>
struct vector;

namespace detail
{
template <typename S, typename U, typename V>
struct vector_proxy;

namespace traits
{
template <typename = void>
struct is_vector_proxy : std::false_type {
};
template <typename T, typename U, typename V>
struct is_vector_proxy<vector_proxy<T, U, V>> : std::true_type {
};
} // namespace traits
} // namespace detail

namespace traits
{
template <typename = void>
struct is_vector : std::false_type {
};
template <typename T>
struct is_vector<vector<T>> : std::true_type {
};
} // namespace traits

template <typename U>
concept Vector = traits::is_vector<std::remove_cvref_t<U>>::value ||
                 detail::traits::is_vector_proxy<std::remove_cvref_t<U>>::value;
} // namespace ccs