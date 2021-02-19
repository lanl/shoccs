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
#include "Vector_fwd.hpp"

#include "Selector_fwd.hpp"

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

    template <typename ViewFn>
    requires(field::tuple::traits::ThreeTuple<R>&& requires(Selection s,
                                                            vs::view_closure<ViewFn> f) {
        f(detail::makeSelection(std::get<0>(s.location.view()),
                                field::Tuple{std::get<0>(s.view())}));
    }) friend constexpr auto
    operator|(Selection& selection, vs::view_closure<ViewFn> f)
    {
        // For some reason, spans trigger errors with the pipe syntax but things appear to
        // work with direct function calls.
        // return field::Tuple{
        //     detail::makeSelection(field::tuple::view<0>(selection.location),
        //                           field::Tuple{field::tuple::view<0>(selection)}) |
        //         f,
        //     detail::makeSelection(field::tuple::view<1>(selection.location),
        //                           field::Tuple{field::tuple::view<1>(selection)}) |
        //         f,
        //     detail::makeSelection(field::tuple::view<2>(selection.location),
        //                           field::Tuple{field::tuple::view<2>(selection)}) |
        //         f};
        return field::Tuple{
            f(detail::makeSelection(field::tuple::view<0>(selection.location),
                                    field::Tuple{field::tuple::view<0>(selection)})),
            f(detail::makeSelection(field::tuple::view<1>(selection.location),
                                    field::Tuple{field::tuple::view<1>(selection)})),
            f(detail::makeSelection(field::tuple::view<2>(selection.location),
                                    field::Tuple{field::tuple::view<2>(selection)}))};
    }

    template <field::tuple::traits::ThreeTuple TupleFn>
    requires(field::tuple::traits::ThreeTuple<R>&& requires(Selection s, TupleFn f) {
        f.template get<0>()(detail::makeSelection(std::get<0>(s.location.view()),
                                                  field::Tuple{std::get<0>(s.view())}));
    }) friend constexpr auto
    operator|(Selection& selection, TupleFn&& f)
    {
        // For some reason, spans trigger errors with the pipe syntax but things appear to
        // work with direct function calls.
        // return field::Tuple{
        //     detail::makeSelection(field::tuple::view<0>(selection.location),
        //                           field::Tuple{field::tuple::view<0>(selection)}) |
        //         f,
        //     detail::makeSelection(field::tuple::view<1>(selection.location),
        //                           field::Tuple{field::tuple::view<1>(selection)}) |
        //         f,
        //     detail::makeSelection(field::tuple::view<2>(selection.location),
        //                           field::Tuple{field::tuple::view<2>(selection)}) |
        //         f};
        return field::Tuple{f.template get<0>()(detail::makeSelection(
                                field::tuple::view<0>(selection.location),
                                field::Tuple{field::tuple::view<0>(selection)})),
                            f.template get<1>()(detail::makeSelection(
                                field::tuple::view<1>(selection.location),
                                field::Tuple{field::tuple::view<1>(selection)})),
                            f.template get<2>()(detail::makeSelection(
                                field::tuple::view<2>(selection.location),
                                field::Tuple{field::tuple::view<2>(selection)}))};
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

    template <field::tuple::traits::VectorType U>
    constexpr auto operator()(U&& u) const
    {
        return Selection{location_fn(FWD(u)), tuple_fn(FWD(u))};
    }

    template <field::tuple::traits::ScalarType U>
    friend constexpr auto operator|(U&& u, SelectorFunc selector)
    {
        return selector(FWD(u));
    }

    template <field::tuple::traits::VectorType U>
    friend constexpr auto operator|(U&& u, SelectorFunc selector)
    {
        return selector(FWD(u));
    }
};
} // namespace detail

inline constexpr auto Dx = detail::SelectorFunc{
    [](field::tuple::traits::VectorType auto&& x) {
        auto l = x.location();
        return vs::cartesian_product(l->x, l->y, l->z);
    },
    [](field::tuple::traits::VectorType auto&& x) {
        return field::Tuple{field::tuple::view<0>(FWD(x).template get<0>())};
    }};

inline constexpr auto Dy = detail::SelectorFunc{
    [](field::tuple::traits::VectorType auto&& x) {
        auto l = x.location();
        return vs::cartesian_product(l->x, l->y, l->z);
    },
    [](field::tuple::traits::VectorType auto&& x) {
        return field::Tuple{field::tuple::view<1>(FWD(x).template get<0>())};
    }};

inline constexpr auto Dz = detail::SelectorFunc{
    [](field::tuple::traits::VectorType auto&& x) {
        auto l = x.location();
        return vs::cartesian_product(l->x, l->y, l->z);
    },
    [](field::tuple::traits::VectorType auto&& x) {
        return field::Tuple{field::tuple::view<2>(FWD(x).template get<0>())};
    }};

inline constexpr auto D = detail::SelectorFunc{
    []<typename T>(T&& x) {
        auto l = x.location();
        if constexpr (field::tuple::traits::ScalarType<T>)
            return vs::cartesian_product(l->x, l->y, l->z);
        else
            return field::Tuple{vs::cartesian_product(l->x, l->y, l->z),
                                vs::cartesian_product(l->x, l->y, l->z),
                                vs::cartesian_product(l->x, l->y, l->z)};
    },
    []<typename T>(T&& x) {
        if constexpr (field::tuple::traits::ScalarType<T>)
            return field::Tuple{field::tuple::view<0>(FWD(x).template get<0>())};
        else
            return field::Tuple{field::tuple::view<0>(FWD(x).template get<0>()),
                                field::tuple::view<1>(FWD(x).template get<0>()),
                                field::tuple::view<2>(FWD(x).template get<0>())};
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
