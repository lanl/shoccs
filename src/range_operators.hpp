#pragma once

#include "types.hpp"

#include <concepts>
#include <functional>

#include <range/v3/range/concepts.hpp>
#include <range/v3/view/zip_with.hpp>
#include <range/v3/view/repeat_n.hpp>

namespace ccs
{

template <typename T>
concept numeric = std::integral<T> || std::floating_point<T>;

#if 0
template <ranges::random_access_range X, ranges::random_access_range Y>
constexpr auto operator+(X&& x, Y&& y) noexcept
{
    return ranges::views::zip_with(std::plus{}, x, y);
}

template <numeric X, ranges::random_access_range Y>
constexpr auto operator+(X x, Y&& y) noexcept
{
    return ranges::views::zip_with(std::plus{}, ranges::views::repeat_n(x, y.size()), y);
}

template <ranges::random_access_range Y, numeric X>
constexpr auto operator+(Y&& y, X x) noexcept
{
    return ranges::views::zip_with(std::plus{}, y, ranges::views::repeat_n(x, y.size()));
}

template <ranges::random_access_range X, ranges::random_access_range Y>
constexpr auto operator*(X&& x, Y&& y) noexcept
{
    return ranges::views::zip_with(std::multiplies{}, x, y);
}

template <numeric X, ranges::random_access_range Y>
constexpr auto operator*(X x, Y&& y) noexcept
{
    return ranges::views::zip_with(
        std::multiplies{}, ranges::views::repeat_n(x, y.size()), y);
}

template <ranges::random_access_range Y, numeric X>
constexpr auto operator*(Y&& y, X x) noexcept
{
    return ranges::views::zip_with(
        std::multiplies{}, y, ranges::views::repeat_n(x, y.size()));
}

template <ranges::random_access_range X, ranges::random_access_range Y>
constexpr auto operator-(X&& x, Y&& y) noexcept
{
    return ranges::views::zip_with(std::minus{}, x, y);
}

template <numeric X, ranges::random_access_range Y>
constexpr auto operator-(X x, Y&& y) noexcept
{
    return ranges::views::zip_with(std::minus{}, ranges::views::repeat_n(x, y.size()), y);
}

template <ranges::random_access_range Y, numeric X>
constexpr auto operator-(Y&& y, X x) noexcept
{
    return ranges::views::zip_with(std::minus{}, y, ranges::views::repeat_n(x, y.size()));
}
#endif

} // namespace ccs