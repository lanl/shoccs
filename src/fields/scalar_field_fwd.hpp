#pragma once

#include "types.hpp"
#include <concepts>

#include <range/v3/range/concepts.hpp>

namespace ccs
{

template <typename T = real, int = 2>
class scalar_field;

template <rs::random_access_range R, int I>
struct scalar_range;

namespace traits
{
// define some traits and concepts to constrain our universal references and ranges
template <typename = void>
struct is_scalar_field : std::false_type {
};
template <typename T, int I>
struct is_scalar_field<scalar_field<T, I>> : std::true_type {
};

template <typename U>
constexpr bool is_scalar_field_v = is_scalar_field<std::remove_cvref_t<U>>::value;

template <typename = void>
struct is_scalar_range : std::false_type {
};
template <typename T, int I>
struct is_scalar_range<scalar_range<T, I>> : std::true_type {
};

template <typename U>
constexpr bool is_scalar_range_v = is_scalar_range<std::remove_cvref_t<U>>::value;

template <typename U>
constexpr bool is_range_or_field_v = is_scalar_field_v<U> || is_scalar_range_v<U>;

template <typename>
struct scalar_dim_;

template <typename T, int I>
struct scalar_dim_<scalar_range<T, I>> {
    static constexpr int dim = I;
};

template <typename T, int I>
struct scalar_dim_<scalar_field<T, I>> {
    static constexpr int dim = I;
};

template <typename T>
requires is_range_or_field_v<std::remove_cvref_t<T>> constexpr int scalar_dim =
    scalar_dim_<std::remove_cvref_t<T>>::dim;

template <typename T, typename U>
requires is_range_or_field_v<T>&& is_range_or_field_v<U> constexpr bool is_same_dim_v =
    scalar_dim<T> == scalar_dim<U>;

} // namespace traits

template <typename T>
concept Scalar_Field = traits::is_range_or_field_v<T>;

template <typename T, typename U>
concept Compatible_Fields = Scalar_Field<T>&& Scalar_Field<U>&& traits::is_same_dim_v<T, U>;

template <typename T, typename U>
concept Transposable_Fields = Scalar_Field<T>&& Scalar_Field<U> && (!traits::is_same_dim_v<T, U>);

} // namespace ccs