#pragma once

#include "types.hpp"
#include <functional>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/empty.hpp>

#include "Tuple_fwd.hpp"

namespace ccs::selector
{
template <field::tuple::All T>
struct Selection;
}

namespace ranges
{
template <typename T>
inline constexpr bool enable_view<ccs::selector::Selection<T>> = true;

//template <ccs::field::tuple::All T>
//inline constexpr bool enable_view<ccs::selector::Selection<T>> = true;

} // namespace ranges

namespace ccs
{

namespace field::tuple
{

template <int I, typename R>
constexpr decltype(auto) view(R&&);
}

namespace selector
{
using field::tuple::All;

template <typename T>
using all_t = decltype(vs::all(std::declval<T>()));

template <All U>
struct Selection : all_t<U> {
private:
    using View = all_t<U>;

public:

    Selection() = default; // need default constructible for viewable_range

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

// namespace ranges
// {
// template <typename T>
// inline constexpr bool enable_view<ccs::selector::Selection<T>> = false;

// template <ccs::field::tuple::All T>
// inline constexpr bool enable_view<ccs::selector::Selection<T>> = true;

//} // namespace ranges