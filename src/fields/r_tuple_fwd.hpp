#pragma once

#include "types.hpp"
#include <concepts>
#include <range/v3/range/concepts.hpp>

namespace ccs
{

// Concept for being able to apply vs::all.  Need to exclude int3 as
// an "Allable" range to trigger proper tuple construction calls
template <typename T>
concept All = rs::range<T&>&& rs::viewable_range<T> &&
              (!std::same_as<int3, std::remove_cvref_t<T>>);

// have a more limited definition of input_range so that
// int3 extents are not considered an input_range
namespace traits
{
template <typename T>
concept range = rs::input_range<T> && (!std::same_as<int3, std::remove_cvref_t<T>>);
}

// Forward decls for main types
template <typename...>
struct r_tuple;

template <typename, int>
class directional_field;

namespace detail
{

// some operations (such as the construction of nested r_tuples)
// will require an extra argument to chose the right templated
// function (ie prefering a constructor over a converting constructor)
struct tag {
};

// forward decls for component types
template <typename...>
struct container_tuple;

template <typename...>
struct view_tuple;

// traits and concepts for component types
namespace traits
{
template <typename>
struct is_container_tuple : std::false_type {
};

template <typename... Args>
struct is_container_tuple<container_tuple<Args...>> : std::true_type {
};

template <typename T>
concept Container_Tuple = is_container_tuple<std::remove_cvref_t<T>>::value;

template <typename T, typename U>
concept Other_Container_Tuple = Container_Tuple<T>&& Container_Tuple<U> &&
                                (!std::same_as<U, std::remove_cvref_t<T>>);

template <typename>
struct is_view_tuple : std::false_type {
};

template <typename... Args>
struct is_view_tuple<view_tuple<Args...>> : std::true_type {
};

template <typename T>
concept View_Tuple = is_view_tuple<std::remove_cvref_t<T>>::value;

} // namespace traits
} // namespace detail

// traits and concepts for main types
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

template <typename T>
concept Non_Tuple_Input_Range = rs::input_range<T> && (!R_Tuple<T>);

template <typename T>
concept Owning_R_Tuple = R_Tuple<T>&& requires(T t)
{
    t.template get<0>();
};

template <typename>
struct is_directional_field : std::false_type {
};

template <typename T, int I>
struct is_directional_field<directional_field<T, I>> : std::true_type {
};

template <typename T>
concept Directional_Field = is_directional_field<std::remove_cvref_t<T>>::value;

// utilities for pulling out direction
template <typename>
struct direction {
};

template <typename T, int I>
struct direction<directional_field<T, I>> {
    static constexpr auto dim = I;
};

template <Directional_Field D>
constexpr auto direction_v = direction<std::remove_cvref_t<D>>::dim;

template <typename U, typename V>
concept Same_Direction = Directional_Field<U>&& Directional_Field<V> &&
                         (direction_v<U> == direction_v<V>);

template <typename...>
struct from_view {
};

// single argument views
template <R_Tuple U>
struct from_view<U> {
    static constexpr auto create = [](auto&&, auto&&... args) {
        return r_tuple{::ccs::detail::tag{}, FWD(args)...};
    };
};

template <Directional_Field U>
struct from_view<U> {
    static constexpr auto create = [](auto&& u, auto&&... args) {
        return directional_field{lit<direction_v<U>>{}, FWD(args)..., u.extents()};
    };
};

// Combination views
template <R_Tuple U, R_Tuple V>
struct from_view<U, V> {
    static constexpr auto create = [](auto&&, auto&&, auto&&... args) {
        return r_tuple{::ccs::detail::tag{}, FWD(args)...};
    };
};

template <Directional_Field U, Same_Direction<U> V>
struct from_view<U, V> {
    static constexpr auto create = [](auto&& u, auto&&, auto&&... args) {
        return directional_field{lit<direction_v<U>>{}, FWD(args)..., u.extents()};
    };
};

// combine directional fields and 1-tuples
template <Directional_Field U, R_Tuple V>
struct from_view<U, V> {
    static constexpr auto create = [](auto&& u, auto&&, auto&&... args) {
        return directional_field{lit<direction_v<U>>{}, FWD(args)..., u.extents()};
    };
};

template <R_Tuple V, Directional_Field U>
struct from_view<V, U> {
    static constexpr auto create = [](auto&&, auto&& u, auto&&... args) {
        return directional_field{lit<direction_v<U>>{}, FWD(args)..., u.extents()};
    };
};

template <typename... T>
static constexpr auto create_from_view = from_view<std::remove_cvref_t<T>...>::create;

template <typename... T>
concept From_View = requires
{
    from_view<std::remove_cvref_t<T>...>::create;
};

} // namespace traits
} // namespace ccs

// need to specialize this bool inorder for r_tuples to have the correct behavior.
// This is somewhat tricky and is hopefully tested by all the "concepts" tests in
// r_tuple.t.cpp
namespace ranges
{
template <typename... Args>
inline constexpr bool enable_view<ccs::r_tuple<Args...>> = false;

template <::ccs::All T>
inline constexpr bool enable_view<ccs::r_tuple<T>> = true;

template <typename T, int I>
inline constexpr bool enable_view<ccs::directional_field<T, I>> = false;
} // namespace ranges
