#pragma once

#include "Tuple.hpp"

#include "Scalar_fwd.hpp"
#include "Vector_fwd.hpp"

#include "Selector_fwd.hpp"

namespace ccs::field::tuple
{

// traits for mapping scalar and vector types to appropriate selection mp_list
namespace traits
{
namespace detail
{
template <typename>
struct selector_func_index_impl;

template <ScalarType T>
struct selector_func_index_impl<T> {
    using type = mp_size_t<0>;
};

template <VectorType T>
struct selector_func_index_impl<T> {
    using type = mp_size_t<1>;
};
} // namespace detail

template <typename T>
using selector_func_index = detail::selector_func_index_impl<T>::type;
} // namespace traits

namespace detail
{
template <traits::ListIndex I, typename R>
constexpr auto makeSelection(R r);
}

template <traits::ListIndex I, traits::TupleType R>
struct Selection : R {
    using Index = I;

    Selection() = default; // default construction needed for semi-regular concept

    constexpr Selection(R r) : R{MOVE(r)} {}

    Selection(const Selection&) = default;
    Selection(Selection&&) = default;

    Selection& operator=(const Selection&) = default;
    Selection& operator=(Selection&&) = default;

    template <typename T>
        requires(!traits::ViewClosures<T>) && std::is_assignable_v<R&, T> Selection&
                                              operator=(T&& t)
    {
        R::operator=(FWD(t));
        return *this;
    }
    // template <traits::SelectionType S>
    // Selection& operator=(S&& s)
    // {
    //     R::operator=(FWD(s).as_Tuple());
    //     return *this;
    // }

    template <traits::ViewClosures F>
    Selection& operator=(F f)
    {
        auto rng = *this | f;
        R::operator=(rng);
        return *this;
    }
};

namespace detail
{
template <traits::ListIndex L, typename R>
constexpr auto makeSelection(R r)
{
    return Selection<L, R>{MOVE(r)};
}

template <typename... Lists>
struct SelectorFunc {
    using Indices = traits::mp_list<Lists...>;

    template <traits::TupleLike U>
    constexpr auto operator()(U&& u) const
    {

        // for now just grab the first element of the list
        using List = traits::mp_at<Indices, traits::selector_func_index<U>>;
        static_assert(!traits::mp_empty<List>::value,
                      "Selection operation not permitted");

        // Build a selection using the tuples indexed in the list
        return []<auto... Is>(std::index_sequence<Is...>, auto&& u)
        {
            if constexpr (sizeof...(Is) == 1)
                return detail::makeSelection<traits::mp_at_c<List, 0>>(
                    Tuple{get<traits::mp_at_c<List, 0>>(FWD(u))});
            else
                return Tuple{detail::makeSelection<traits::mp_at_c<List, Is>>(
                    Tuple{get<traits::mp_at_c<List, Is>>(FWD(u))})...};
        }
        (sequence<List>, FWD(u));
    }

    template <traits::TupleLike U>
    friend constexpr auto operator|(U&& u, SelectorFunc selector)
    {
        return selector(FWD(u));
    }
};
} // namespace detail

template <typename... L>
constexpr auto Selector = detail::SelectorFunc<L...>{};

using namespace ccs::selector;
using traits::mp_list;

// inline constexpr auto Dx = Selector<0, 0>;
// inline constexpr auto Dy = Selector<1, 0>;
// inline constexpr auto Dz = Selector<2, 0>;
inline constexpr auto Rx = Selector<mp_list<scalar::Rx>, mp_list<>>;
inline constexpr auto Ry = Selector<mp_list<scalar::Ry>, mp_list<>>;
inline constexpr auto Rz = Selector<mp_list<scalar::Rz>, mp_list<>>;
inline constexpr auto D = Selector<mp_list<scalar::D>, mp_list<>>;
inline constexpr auto R =
    Selector<mp_list<scalar::Rx, scalar::Ry, scalar::Rz>, mp_list<>>;

} // namespace ccs::field::tuple

namespace ccs::selector
{
using field::tuple::D;
// using field::tuple::Dx;
// using field::tuple::Dy;
// using field::tuple::Dz;
using field::tuple::R;
using field::tuple::Rx;
using field::tuple::Ry;
using field::tuple::Rz;
} // namespace ccs::selector
