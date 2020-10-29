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

template <typename...>
struct from_view {
};

template <template <typename...> typename U, typename... Args>
struct from_view<U<Args...>> {
    static constexpr auto create = [](auto&&, auto&&... args) {
        return U{FWD(args)...};
    };
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