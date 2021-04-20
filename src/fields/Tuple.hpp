#pragma once

#include "Tuple_fwd.hpp"

#include "TupleMath.hpp"
#include "indexing.hpp"

#include "ContainerTuple.hpp"
#include "TupleUtils.hpp"
#include "ViewTuple.hpp"

namespace ccs::field::tuple
{

// r_tuple for viewable ref components
template <typename... Args>
struct Tuple : ContainerTuple<Args...>, ViewTuple<Args&...> {
    using Container = ContainerTuple<Args...>;
    using View = ViewTuple<Args&...>;
    using Type = Tuple<Args...>;

    Tuple() = default;

    Tuple(Args&&... args) : Container{FWD(args)...}, View{*this} {}

    template <typename... T>
        requires(sizeof...(Args) > 0) && std::constructible_from<Container, T...> Tuple(
                                             T&&... args)
        : Container{FWD(args)...},
    View{*this}
    {
    }

    Tuple(tag, Args&&... args) : Container{FWD(args)...}, View{*this} {}

    template <typename... T>
        requires(sizeof...(Args) > 0) && std::constructible_from<Container, T...> Tuple(
                                             tag, T&&... args)
        : Container{FWD(args)...},
    View{*this}
    {
    }

    template <typename T>
    requires std::is_assignable_v<Container&, T> Tuple& operator=(T&& t)
    {
        Container::operator=(FWD(t));
        // need to adjust the views incase the container has resized
        View::operator=(*this);
        return *this;
    }

    // need to define custom copy and move construction/assignment here since the
    // component-wise approach is not correct
    Tuple(const Tuple& r) : Container{r}, View{*this} {}
    Tuple& operator=(const Tuple& r)
    {
        Container::operator=(r);
        View::operator=(*this);
        return *this;
    }

    Tuple(Tuple&& r) noexcept : Container{MOVE(r)}, View{*this} {}

    Tuple& operator=(Tuple&& r) noexcept
    {
        Container::operator=(MOVE(r));
        View::operator=(*this);
        return *this;
    }

    template <traits::NonTupleRange T>
    friend bool operator==(const Tuple& x,
                           const T& y) requires traits::NestedTupleLike<Tuple>
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return ((get<Is>(x) == y) && ...);
        }
        (TupleIndex<Tuple>);
    }

    template <traits::SimilarTuples<Tuple> T>
    friend bool operator==(const Tuple& x,
                           const T& y) requires traits::NestedTupleLike<Tuple>
    {
        return [&]<auto... Is>(std::index_sequence<Is...>)
        {
            return ((get<Is>(x) == get<Is>(y)) && ...);
        }
        (TupleIndex<Tuple>);
    }

    Tuple& as_Tuple() & { return *this; }
    const Tuple& as_Tuple() const& { return *this; }
    Tuple&& as_Tuple() && { return MOVE(*this); }
};

//
// Non Owning Tuple
//
template <All... Args>
struct Tuple<Args...> : ViewTuple<Args...> {
    using View = ViewTuple<Args...>;
    using Type = Tuple<Args...>;

    explicit Tuple(Args&&... args) : View{FWD(args)...} {};

    explicit Tuple(tag, Args&&... args) : View{FWD(args)...} {};

    Tuple() = default;

    template <typename... T>
    requires(std::constructible_from<View, T...>) Tuple(T&&... t) : View{FWD(t)...}
    {
    }

    template <typename T>
    requires std::is_assignable_v<View&, T> Tuple& operator=(T&& t)
    {
        View::operator=(FWD(t));
        return *this;
    }

    Tuple& as_Tuple() & { return *this; }
    const Tuple& as_Tuple() const& { return *this; }
    Tuple&& as_Tuple() && { return MOVE(*this); }
};

template <typename... Args>
Tuple(Args&&...) -> Tuple<traits::viewable_range_by_value<Args>...>;
// Tuple(Args&&...) -> Tuple<Args...>;

// need to caputre view closures by value to meet range-v3 concepts
template <typename... ViewFn>
Tuple(vs::view_closure<ViewFn>&...) -> Tuple<vs::view_closure<ViewFn>...>;
template <typename... ViewFn>
Tuple(const vs::view_closure<ViewFn>&...) -> Tuple<vs::view_closure<ViewFn>...>;

template <typename... Args>
Tuple(tag, Args&&...) -> Tuple<traits::viewable_range_by_value<Args>...>;

template <std::size_t I, traits::TupleType C>
constexpr decltype(auto) get(C&& c)
{
    if constexpr (traits::OwningTuple<C>)
        return get<I>(FWD(c).as_Container());
    else {
        using B = typename std::remove_cvref_t<C>::View;
        return get<I>(static_cast<boost::copy_cv_ref_t<B, C&&>>(FWD(c)));

        // return get<I>(FWD(c).as_ViewTuple());
    }
}

template <std::size_t I, std::size_t J, std::size_t... Rest, traits::TupleType C>
constexpr decltype(auto) get(C&& c)
{
    return get<J, Rest...>(get<I>(FWD(c)));
}

template <traits::ListIndex L, traits::TupleType C>
constexpr decltype(auto) get(C&& c)
{
    // this really should only require one lambda but I can't do it without triggering a
    // parse error
    return []<auto... Ls>(std::index_sequence<Ls...>, auto&& c)->decltype(auto)
    {
        return []<auto... Is>(auto&& c, traits::mp_size_t<Is>...)->decltype(auto)
        {
            return get<Is...>(FWD(c));
        }
        (FWD(c), traits::mp_at_c<L, Ls>{}...);
    }
    (std::make_index_sequence<traits::mp_size<L>::value>(), FWD(c));
}

} // namespace ccs::field::tuple

// specialize tuple_size
namespace std
{
template <typename... Args>
struct tuple_size<ccs::field::Tuple<Args...>>
    : std::integral_constant<size_t, sizeof...(Args)> {
};

template <size_t I, typename... Args>
struct tuple_element<I, ccs::field::tuple::Tuple<Args...>>
    : tuple_element<I, ccs::field::tuple::ContainerTuple<Args...>> {
};

template <size_t I, ccs::field::tuple::All... Args>
struct tuple_element<I, ccs::field::tuple::Tuple<Args...>>
    : tuple_element<I, ccs::field::tuple::ViewTuple<Args...>> {
};
} // namespace std

namespace ccs::field::tuple
{

namespace traits
{
template <typename T, auto N>
concept NTuple = TupleType<T>&& std::tuple_size_v<std::remove_cvref_t<T>> == N;

template <typename T>
concept OneTuple = NTuple<T, 1u>;

template <typename T>
concept ThreeTuple = NTuple<T, 3u>;
} // namespace traits

} // namespace ccs::field::tuple
