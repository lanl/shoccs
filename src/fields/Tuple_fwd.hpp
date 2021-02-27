#pragma once

#include "types.hpp"
#include <concepts>
#include <range/v3/range/concepts.hpp>

namespace ccs::field::tuple
{

using std::get;

template <int I, typename R>
constexpr auto view(R&&);

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
concept Range = rs::input_range<T> && (!std::same_as<int3, std::remove_cvref_t<T>>);

template <typename T>
concept OutputRange = rs::range<T>&& rs::output_range<T, rs::range_value_t<T>>;
} // namespace traits

// Forward decls for main types
template <typename...>
struct Tuple;

// some operations (such as the construction of nested r_tuples)
// will require an extra argument to chose the right templated
// function (ie prefering a constructor over a converting constructor)
struct tag {
};

// forward decls for component types
template <typename...>
struct ContainerTuple;

template <typename...>
struct view_tuple;

template <typename...>
struct ViewBaseTuple;

template <typename...>
struct ViewTuple;

// traits and concepts for component types
namespace traits
{
template <typename>
struct is_ContainerTuple : std::false_type {
};

template <typename... Args>
struct is_ContainerTuple<ContainerTuple<Args...>> : std::true_type {
};

template <typename T>
concept ContainerTupleType = is_ContainerTuple<std::remove_cvref_t<T>>::value;

template <typename T, typename U>
concept OtherContainerTuple = ContainerTupleType<T>&& ContainerTupleType<U> &&
                              (!std::same_as<U, std::remove_cvref_t<T>>);

template <typename>
struct is_ViewBaseTuple : std::false_type {
};

template <typename... Args>
struct is_ViewBaseTuple<ViewBaseTuple<Args...>> : std::true_type {
};

template <typename T>
concept ViewBaseTupleType = is_ViewBaseTuple<std::remove_cvref_t<T>>::value;

template <typename>
struct is_view_tuple : std::false_type {
};

template <typename... Args>
struct is_view_tuple<view_tuple<Args...>> : std::true_type {
};

template <typename T>
concept View_Tuple = is_view_tuple<std::remove_cvref_t<T>>::value;

template <typename>
struct is_ViewTuple : std::false_type {
};

template <typename... Args>
struct is_ViewTuple<ViewTuple<Args...>> : std::true_type {
};

template <typename T>
concept ViewTupleType = is_ViewTuple<std::remove_cvref_t<T>>::value;

template <typename>
struct is_Tuple : std::false_type {
};

template <typename... Args>
struct is_Tuple<Tuple<Args...>> : std::true_type {
};

template <typename T>
concept TupleType = is_Tuple<std::remove_cvref_t<T>>::value;

template <typename T>
concept Non_Tuple_Input_Range = rs::input_range<T> && (!TupleType<T>);

template <typename T>
concept NonTupleRange = rs::input_range<T> &&
                        (!ViewBaseTupleType<T>)&&(!ViewTupleType<T>)&&(!TupleType<T>);

template <typename T>
concept Owning_Tuple = TupleType<T>&& requires(T t)
{
    t.template get<0>();
};

template <typename T>
concept TupleLike = requires(T t)
{
    get<0>(t);
};
} // namespace traits

template <typename...>
struct from_view {
};

// single argument views
template <traits::TupleType U>
struct from_view<U> {
    static constexpr auto create = [](auto&&, auto&&... args) {
        return Tuple{tag{}, FWD(args)...};
    };
};

// Combination views
template <traits::TupleType U, traits::TupleType V>
struct from_view<U, V> {
    static constexpr auto create = [](auto&&, auto&&, auto&&... args) {
        return Tuple{tag{}, FWD(args)...};
    };
};

template <typename... T>
static constexpr auto create_from_view = from_view<std::remove_cvref_t<T>...>::create;

template <typename... T>
concept From_View = requires
{
    from_view<std::remove_cvref_t<T>...>::create;
};

} // namespace ccs::field::tuple

namespace ccs::field
{
using tuple::Tuple;
}

// specialize tuple_size
namespace std
{
template <typename... Args>
struct tuple_size<ccs::field::Tuple<Args...>>
    : std::integral_constant<size_t, sizeof...(Args)> {
};
} // namespace std

// add concepts for commonly used tuple sizes
namespace ccs::field::tuple::traits
{
template <typename T, auto N>
concept NTuple = TupleType<T>&& std::tuple_size_v<std::remove_cvref_t<T>> == N;

template <typename T>
concept OneTuple = NTuple<T, 1u>;

template <typename T>
concept ThreeTuple = NTuple<T, 3u>;

} // namespace ccs::field::tuple::traits

// need to specialize this bool inorder for r_tuples to have the correct behavior.
// This is somewhat tricky and is hopefully tested by all the "concepts" tests in
// r_tuple.t.cpp
namespace ranges
{
template <typename... Args>
inline constexpr bool enable_view<ccs::field::Tuple<Args...>> = false;

template <ccs::field::tuple::All T>
inline constexpr bool enable_view<ccs::field::Tuple<T>> = true;

// template <typename T, int I>
// inline constexpr bool enable_view<ccs::directional_field<T, I>> = false;
} // namespace ranges
