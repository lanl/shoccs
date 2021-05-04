#pragma once

#include "tuple.hpp"

#include "scalar.hpp"
#include "vector.hpp"

#include "selector_fwd.hpp"

#include "index_extents.hpp"

namespace ccs
{

// traits for mapping scalar and vector types to appropriate selection mp_list
namespace detail
{
template <typename>
struct selection_fn_index_impl;

template <Scalar T>
struct selection_fn_index_impl<T> {
    using type = mp_size_t<0>;
};

template <Vector T>
struct selection_fn_index_impl<T> {
    using type = mp_size_t<1>;
};
} // namespace detail

template <typename T>
using selection_fn_index = detail::selection_fn_index_impl<T>::type;

namespace detail
{
template <ListIndex I, typename R>
constexpr auto make_selection(R r);
}

template <ListIndex I, Tuple R>
struct selection : R {
    using index = I;

    selection() = default; // default construction needed for semi-regular concept

    constexpr selection(R r) : R{MOVE(r)} {}

    selection(const selection&) = default;
    selection(selection&&) = default;

    selection& operator=(const selection&) = default;
    selection& operator=(selection&&) = default;

    template <typename T>
        requires(!ViewClosures<T>)
    &&std::is_assignable_v<R&, T> selection& operator=(T&& t)
    {
        R::operator=(FWD(t));
        return *this;
    }
    // template <Selection S>
    // selection& operator=(S&& s)
    // {
    //     R::operator=(FWD(s).as_Tuple());
    //     return *this;
    // }

    template <ViewClosures F>
    selection& operator=(F f)
    {
        auto rng = *this | f;
        R::operator=(rng);
        return *this;
    }
};

namespace detail
{
template <ListIndex L, typename R>
constexpr auto make_selection(R r)
{
    return selection<L, R>{MOVE(r)};
}

template <typename... Lists>
struct selection_fn {
    using Indices = mp_list<Lists...>;

    template <TupleLike U>
    constexpr auto operator()(U&& u) const
    {

        // for now just grab the first element of the list
        using List = mp_at<Indices, selection_fn_index<U>>;
        static_assert(!mp_empty<List>::value, "selection operation not permitted");

        // Build a selection using the tuples indexed in the list
        return []<auto... Is>(std::index_sequence<Is...>, auto&& u)
        {
            if constexpr (sizeof...(Is) == 1)
                return detail::make_selection<mp_at_c<List, 0>>(
                    tuple{get<mp_at_c<List, 0>>(FWD(u))});
            else
                return tuple{detail::make_selection<mp_at_c<List, Is>>(
                    tuple{get<mp_at_c<List, Is>>(FWD(u))})...};
        }
        (sequence<List>, FWD(u));
    }

    template <TupleLike U>
    friend constexpr auto operator|(U&& u, selection_fn fn)
    {
        return fn(FWD(u));
    }
};

// First parameter indicate the direction of the plane {0, 1, 2}
template <auto I>
struct plane_selection_fn {
    // using Indices = mp_list<Lists...>;

    template <TupleLike U>
    constexpr auto operator()(U&& u) const
    {

        // // for now just grab the first element of the list
        // using List = mp_at<Indices, selection_fn_index<U>>;
        // static_assert(!mp_empty<List>::value, "selection operation not permitted");

        // // Build a selection using the tuples indexed in the list
        // return []<auto... Is>(std::index_sequence<Is...>, auto&& u)
        // {
        //     if constexpr (sizeof...(Is) == 1)
        //         return detail::make_selection<mp_at_c<List, 0>>(
        //             tuple{get<mp_at_c<List, 0>>(FWD(u))});
        //     else
        //         return tuple{detail::make_selection<mp_at_c<List, Is>>(
        //             tuple{get<mp_at_c<List, Is>>(FWD(u))})...};
        // }
        // (sequence<List>, FWD(u));
        return u;
    }

    template <TupleLike U>
    constexpr auto operator()(U&& u, int coord, index_extents extents)
    {
        return u;
    }

    template <TupleLike U>
    friend constexpr auto operator|(U&& u, plane_selection_fn fn)
    {
        return fn(FWD(u));
    }
};
} // namespace detail

template <typename... L>
constexpr auto selection_fn = detail::selection_fn<L...>{};

template <auto I>
constexpr auto plane_selection_fn = detail::plane_selection_fn<I>{};

namespace sel
{

inline constexpr auto Dx = selection_fn<mp_list<>, mp_list<vi::Dx>>;
inline constexpr auto Dy = selection_fn<mp_list<>, mp_list<vi::Dy>>;
inline constexpr auto Dz = selection_fn<mp_list<>, mp_list<vi::Dz>>;
inline constexpr auto Rx =
    selection_fn<mp_list<si::Rx>, mp_list<vi::xRx, vi::yRx, vi::zRx>>;
inline constexpr auto Ry =
    selection_fn<mp_list<si::Ry>, mp_list<vi::xRy, vi::yRy, vi::zRy>>;
inline constexpr auto Rz =
    selection_fn<mp_list<si::Rz>, mp_list<vi::xRz, vi::yRz, vi::zRz>>;
inline constexpr auto D = selection_fn<mp_list<si::D>, mp_list<vi::Dx, vi::Dy, vi::Dz>>;
inline constexpr auto R = selection_fn<mp_list<si::Rx, si::Ry, si::Rz>,
                                       mp_list<vi::xRx,
                                               vi::yRx,
                                               vi::zRx,
                                               vi::xRy,
                                               vi::yRy,
                                               vi::zRy,
                                               vi::xRz,
                                               vi::yRz,
                                               vi::zRz>>;
inline constexpr auto xRx = selection_fn<mp_list<>, mp_list<vi::xRx>>;
inline constexpr auto xRy = selection_fn<mp_list<>, mp_list<vi::xRy>>;
inline constexpr auto xRz = selection_fn<mp_list<>, mp_list<vi::xRz>>;
inline constexpr auto yRx = selection_fn<mp_list<>, mp_list<vi::yRx>>;
inline constexpr auto yRy = selection_fn<mp_list<>, mp_list<vi::yRy>>;
inline constexpr auto yRz = selection_fn<mp_list<>, mp_list<vi::yRz>>;
inline constexpr auto zRx = selection_fn<mp_list<>, mp_list<vi::zRx>>;
inline constexpr auto zRy = selection_fn<mp_list<>, mp_list<vi::zRy>>;
inline constexpr auto zRz = selection_fn<mp_list<>, mp_list<vi::zRz>>;

inline constexpr auto x_plane = plane_selection_fn<0>;
inline constexpr auto y_plane = plane_selection_fn<1>;
inline constexpr auto z_plane = plane_selection_fn<2>;

// inline constexpr auto xmin = x_plane(0);
// inline constexpr auto xmax = x_plane(-1);
// inline constexpr auto ymin = y_plane(0);
// inline constexpr auto ymax = y_plane(-1);
// inline constexpr auto zmin = z_plane(0);
// inline constexpr auto zmax = z_plane(-1);

} // namespace sel

} // namespace ccs
