#pragma once

#include "types.hpp"
#include <functional>
#include <range/v3/view/all.hpp>
#include <range/v3/view/empty.hpp>

namespace ccs
{

namespace field::tuple
{

template <int I, typename R>
constexpr decltype(auto) view(R&&);
}

namespace selector
{
template <typename T>
using all_t = decltype(vs::all(std::declval<T>()));

template <typename U>
struct Selection : all_t<U> {
private:
    using View = all_t<U>;

public:
    constexpr Selection(U&& u) : View(vs::all(FWD(u))) {}

    template <typename T>
    Selection& operator=(T&&)
    {
        return *this;
    }
};

template <typename U>
Selection(U&&) -> Selection<U>;

namespace detail
{
template <typename Func>
struct SelectorFunc {

    Func fn;

    constexpr SelectorFunc(Func fn) : fn{MOVE(fn)} {}

    template <typename U>
    constexpr auto operator()(U&& u) const
    {
        return Selection{fn(FWD(u))};
    }

    template <typename U>
    friend constexpr auto operator|(U&& u, SelectorFunc selector)
    {
        return selector(FWD(u));
    }
};
} // namespace detail

inline constexpr auto D = detail::SelectorFunc{
    [](auto&& x) { return field::tuple::view<0>(FWD(x).template get<0>()); }};
inline constexpr auto Rx = detail::SelectorFunc{
    [](auto&& x) { return field::tuple::view<0>(FWD(x).template get<1>()); }};
inline constexpr auto Ry = detail::SelectorFunc{
    [](auto&& x) { return field::tuple::view<1>(FWD(x).template get<1>()); }};
inline constexpr auto Rz = detail::SelectorFunc{
    [](auto&& x) { return field::tuple::view<2>(FWD(x).template get<1>()); }};
inline constexpr auto Rxyz = detail::SelectorFunc{std::identity{}};
} // namespace selector

} // namespace ccs