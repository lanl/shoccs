#pragma once

#include "types.hpp"
#include <concepts>
#include <range/v3/range/concepts.hpp>

namespace ccs
{
template <typename...>
struct r_tuple;

template <typename, int>
class directional_field;

namespace detail
{
template <typename...>
struct container_tuple;

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
} // namespace traits

} // namespace detail

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
concept Owning_R_Tuple = R_Tuple<T> && requires(T t) { t.template get<0>(); };

template <typename>
struct is_directional_field : std::false_type {
};

template <typename T, int I>
struct is_directional_field<directional_field<T, I>> : std::true_type {
};

template <typename T>
concept Directional_Field = is_directional_field<std::remove_cvref_t<T>>::value;

template <typename...>
struct from_view {
};

template <template <typename...> typename U, typename... Args>
struct from_view<U<Args...>> {
    static constexpr auto create = [](auto&&, auto&&... args) { return U{FWD(args)...}; };
};

template <template <typename...> typename U,
          typename... UArgs,
          template <typename...>
          typename V,
          typename... VArgs>
struct from_view<U<UArgs...>, V<VArgs...>> {
    static constexpr auto create = [](auto&&, auto&&, auto&&... args) {
        return U{FWD(args)...};
    };
};

template <template <typename, int> typename U, typename UArgs, int I>
struct from_view<U<UArgs, I>> {
    static constexpr auto create = [](auto&& u, auto&&... args) {
        return U{lit<I>{}, FWD(args)..., u.extents()};
    };
};

template <template <typename, int> typename U,
          typename UArgs,
          int I,
          template <typename, int>
          typename V,
          typename VArgs>
struct from_view<U<UArgs, I>, V<VArgs, I>> {
    static constexpr auto create = [](auto&& u, auto&&, auto&&... args) {
        return U{lit<I>{}, FWD(args)..., u.extents()};
    };
};

// combine directional fields and 1-tuples
template <template <typename, int> typename U,
          typename UArgs,
          int I,
          template <typename>
          typename V,
          typename VArgs>
struct from_view<U<UArgs, I>, V<VArgs>> {
    static constexpr auto create = [](auto&& u, auto&&, auto&&... args) {
        return U{lit<I>{}, FWD(args)..., u.extents()};
    };
};

template <template <typename, int> typename U,
          typename UArgs,
          int I,
          template <typename>
          typename V,
          typename VArgs>
struct from_view<V<VArgs>, U<UArgs, I>> {
    static constexpr auto create = [](auto&&, auto&& u, auto&&... args) {
        return U{lit<I>{}, FWD(args)..., u.extents()};
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

// need to specialize this bool inorder for r_tuples to have the correct behavior
namespace ranges
{
template <typename... Args>
inline constexpr bool enable_view<ccs::r_tuple<Args...>> = false;

template <typename T, int I>
inline constexpr bool enable_view<ccs::directional_field<T, I>> = false;
} // namespace ranges