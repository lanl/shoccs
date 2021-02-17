#pragma once

#include "types.hpp"
#include <functional>
#include <range/v3/algorithm/copy.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/empty.hpp>

#include "Scalar_fwd.hpp"
#include "Tuple_fwd.hpp"

namespace ccs::selector
{

namespace detail
{
template <typename L, typename R>
constexpr auto makeSelection(L l, R r);
}

template <typename L, field::tuple::traits::TupleType R>
struct Selection : R {
    L location;

    Selection() = default; // default construction needed for semi-regular concept

    constexpr Selection(L l, R r) : R{MOVE(r)}, location{MOVE(l)} {}

    template <Numeric N>
    Selection& operator=(N n)
    {
        R::operator=(n);
        return *this;
    }

    template <typename Fn>
    requires(!Numeric<Fn> && requires(Selection s, Fn f) { s | f; }) Selection&
    operator=(Fn f)
    {
        auto rng = *this | f;
        R::operator=(rng);
        return *this;
    }

    template <typename Fn>
    requires(!Numeric<Fn> && field::tuple::traits::ThreeTuple<R>) friend constexpr auto
    operator|(Selection& selection, Fn f)
    {
        return field::Tuple{
            detail::makeSelection(field::tuple::view<0>(selection.location),
                                  field::Tuple{field::tuple::view<0>(selection)}) |
                f,
            detail::makeSelection(field::tuple::view<1>(selection.location),
                                  field::Tuple{field::tuple::view<1>(selection)}) |
                f,
            detail::makeSelection(field::tuple::view<2>(selection.location),
                                  field::Tuple{field::tuple::view<2>(selection)}) |
                f};
    }
};

#if 0
template <typename R>
Selection(R&&) -> Selection<std::remove_reference_t<R>>;

template <typename R>
Selection(R) -> Selection<R>;
#endif
namespace detail
{
template <typename L, typename R>
constexpr auto makeSelection(L l, R r)
{
    return Selection<L, R>{MOVE(l), MOVE(r)};
}

template <typename LocationFn, typename TupleFn>
struct SelectorFunc {

    LocationFn location_fn;
    TupleFn tuple_fn;

    constexpr SelectorFunc(LocationFn l, TupleFn t)
        : location_fn{MOVE(l)}, tuple_fn{MOVE(t)}
    {
    }

    template <field::tuple::traits::ScalarType U>
    constexpr auto operator()(U&& u) const
    {
        return Selection{location_fn(FWD(u)), tuple_fn(FWD(u))};
    }

    template <field::tuple::traits::ScalarType U>
    friend constexpr auto operator|(U&& u, SelectorFunc selector)
    {
        return selector(FWD(u));
    }
};
} // namespace detail

inline constexpr auto D = detail::SelectorFunc{
    [](auto&& x) {
        auto l = x.location();
        return vs::cartesian_product(l->x, l->y, l->z);
    },
    [](auto&& x) {
        return field::Tuple{field::tuple::view<0>(FWD(x).template get<0>())};
    }};

inline constexpr auto Rx = detail::SelectorFunc{
    [](auto&& x) {
        auto l = x.location();
        return vs::all(l->rx);
    },
    [](auto&& x) {
        return field::Tuple{field::tuple::view<0>(FWD(x).template get<1>())};
    }};

inline constexpr auto Ry = detail::SelectorFunc{
    [](auto&& x) {
        auto l = x.location();
        return vs::all(l->ry);
    },
    [](auto&& x) {
        return field::Tuple{field::tuple::view<1>(FWD(x).template get<1>())};
    }};

inline constexpr auto Rz = detail::SelectorFunc{
    [](auto&& x) {
        auto l = x.location();
        return vs::all(l->rz);
    },
    [](auto&& x) {
        return field::Tuple{field::tuple::view<2>(FWD(x).template get<1>())};
    }};

inline constexpr auto Rxyz = detail::SelectorFunc{
    [](auto&& x) {
        auto l = x.location();
        return field::Tuple{l->rx, l->ry, l->rz};
    },
    [](auto&& x) {
        // Note that simply re
        return field::Tuple{field::tuple::view<0>(FWD(x).template get<1>()),
                            field::tuple::view<1>(FWD(x).template get<1>()),
                            field::tuple::view<2>(FWD(x).template get<1>())};
    }};
} // namespace ccs::selector
