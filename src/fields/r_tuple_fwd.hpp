#pragma once

#include "types.hpp"
#include <concepts>

namespace ccs
{
template <typename...>
struct r_tuple;

namespace traits
{
template <typename>
struct is_r_tuple : std::false_type {
};

template <typename... Args>
struct is_r_tuple<r_tuple<Args...>> : std::true_type {
};

template <typename T>
concept R_Tuple = is_r_tuple<std::remove_cvref_t<T>>::value;
} // namespace traits
} // namespace ccs