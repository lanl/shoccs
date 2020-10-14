#pragma once

#include "types.hpp"
#include <concepts>

namespace ccs
{

template <typename T = real>
class vector_field;

template <typename X, typename Y, typename Z>
struct vector_range;

// A way to encapsulate arguments to various components 
template <typename... Args>
struct v_arg {
    std::tuple<Args...> args;
};
template<typename...Args>
v_arg(Args&&...) -> v_arg<Args...>;


namespace traits
{
// define some traits and concepts to constrain our universal references and ranges
template <typename = void>
struct is_vector_field : std::false_type {
};
template <typename T>
struct is_vector_field<vector_field<T>> : std::true_type {
};

template <typename U>
constexpr bool is_vector_field_v = is_vector_field<std::remove_cvref_t<U>>::value;

template <typename = void>
struct is_vector_range : std::false_type {
};
template <typename X, typename Y, typename Z>
struct is_vector_range<vector_range<X, Y, Z>> : std::true_type {
};

template <typename U>
constexpr bool is_vector_range_v = is_vector_range<std::remove_cvref_t<U>>::value;

template <typename U>
constexpr bool is_vrange_or_vfield_v = is_vector_field_v<U> || is_vector_range_v<U>;

} // namespace traits

template <typename T>
concept Vector_Field = traits::is_vrange_or_vfield_v<T>;
} // namespace ccs