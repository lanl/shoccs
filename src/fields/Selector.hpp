#pragma once

#include "Tuple.hpp"

#if 0
#include "Scalar_fwd.hpp"
#include "Vector_fwd.hpp"
#endif
#include "Selector_fwd.hpp"

namespace ccs::field::tuple
{

namespace detail
{
template <auto... I, typename R>
constexpr auto makeSelection(R r);
}

template <traits::TupleType R, auto... I>
struct Selection : R {
    using Idx = traits::mp_list<traits::mp_size_t<I>...>;

    Selection() = default; // default construction needed for semi-regular concept

    constexpr Selection(R r, traits::mp_size_t<I>...) : R{MOVE(r)} {}

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

    constexpr auto to_Tuple() const
    {
        return []<auto... Rs>(std::index_sequence<Rs...>, auto&& t)
        {
            return Tuple{detail::makeSelection<Rs, I...>(Tuple{get<Rs>(FWD(t))})...};
        }
        (TupleIndex<R>, this->as_Tuple());
    }
};

namespace detail
{
template <auto... I, typename R>
constexpr auto makeSelection(R r)
{
    return Selection{MOVE(r), traits::mp_size_t<I>()...};
}

template <auto... I>
struct SelectorFunc {

    template <traits::TupleLike U>
    constexpr auto operator()(U&& u) const
    {
        using G = decltype(get<I...>(FWD(u)));

        if constexpr (!traits::TupleLike<G>) {
            return Selection{Tuple{get<I...>(FWD(u))}, traits::mp_size_t<I>()...};
        } else {
            // to get the right reference semantics we need to extract all the components
            // at this level and stuff it into the return tuple
            return []<auto... Gs>(std::index_sequence<Gs...>, auto&& v)
            {
                return Selection{Tuple{get<Gs>(FWD(v))...}, traits::mp_size_t<I>()...};
            }
            (TupleIndex<G>, get<I...>(FWD(u)));
        }
    }

    template <traits::TupleLike U>
    friend constexpr auto operator|(U&& u, SelectorFunc selector)
    {
        return selector(FWD(u));
    }
};
} // namespace detail

template <auto... I>
constexpr auto Selector = detail::SelectorFunc<I...>{};

inline constexpr auto Dx = Selector<0, 0>;
inline constexpr auto Dy = Selector<1, 0>;
inline constexpr auto Dz = Selector<2, 0>;
inline constexpr auto Rx = Selector<0, 1>;
inline constexpr auto Ry = Selector<1, 1>;
inline constexpr auto Rz = Selector<2, 1>;
inline constexpr auto D = Selector<0>;
inline constexpr auto R = Selector<1>;

} // namespace ccs::field::tuple

namespace ccs::selector
{
using field::tuple::D;
using field::tuple::Dx;
using field::tuple::Dy;
using field::tuple::Dz;
using field::tuple::R;
using field::tuple::Rx;
using field::tuple::Ry;
using field::tuple::Rz;
} // namespace ccs::selector
