#pragma once

#include "Tuple_fwd.hpp"

#include "TupleMath.hpp"
#include "indexing.hpp"

#include "ContainerTuple.hpp"
#include "TupleUtils.hpp"
#include "ViewTuple.hpp"

namespace ccs::field::tuple
{

namespace detail
{
template <typename... Args>
auto makeTuple(Args&&...);
}

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

    // template <traits::Non_Tuple_Input_Range... R>
    // Tuple& operator=(R&&... r)
    // {
    //     Container::operator=(FWD(r)...);
    //     View::operator=(this->container());
    //     return *this;
    // }

    // template <typename ViewFn>
    // requires(sizeof...(Args) > 1 &&
    //          requires(Tuple t, vs::view_closure<ViewFn> f) {
    //              std::get<0>(t.view()) | f;
    //          }) friend constexpr auto
    // operator|(Tuple& t, vs::view_closure<ViewFn> f)
    // {
    //     return [f]<auto... Is>(std::index_sequence<Is...>, auto& t)
    //     {
    //         return detail::makeTuple((view<Is>(t) | f)...);
    //     }
    //     (std::make_index_sequence<sizeof...(Args)>{}, t);
    // }

    // // allow for composing a tuple of ranges with a tuple of functions
    // template <traits::TupleType TupleFn>
    // requires(sizeof...(Args) > 1 &&
    //          requires(Tuple t, TupleFn f) {
    //              std::get<0>(t.view()) | f.template get<0>();
    //          }) friend constexpr auto
    // operator|(Tuple& t, TupleFn&& f)
    // {
    //     return [&t]<auto... Is>(std::index_sequence<Is...>, auto&& fn)
    //     {
    //         return detail::makeTuple((view<Is>(t) | fn.template get<Is>())...);
    //     }
    //     (std::make_index_sequence<sizeof...(Args)>{}, FWD(f));
    // }

    // template <typename ViewFn>
    // requires requires(vs::view_closure<ViewFn> f, Tuple t)
    // {
    //     f | t.template get<0>();
    // }
    // friend constexpr auto operator|(vs::view_closure<ViewFn> f, Tuple t)
    // {
    //     return [f]<auto... Is>(std::index_sequence<Is...>, auto&& t)
    //     {
    //         return detail::makeTuple((f | t.template get<Is>())...);
    //     }
    //     (std::make_index_sequence<sizeof...(Args)>{}, MOVE(t));
    // }
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

    // template <typename Fn>
    // requires(sizeof...(Args) > 1 &&
    //          requires(Tuple t, Fn f) { std::get<0>(t.view()) | f; }) friend constexpr
    //          auto
    // operator|(Tuple& t, Fn f)
    // {
    //     return [f]<auto... Is>(std::index_sequence<Is...>, auto& t)
    //     {
    //         return detail::makeTuple((view<Is>(t) | f)...);
    //     }
    //     (std::make_index_sequence<sizeof...(Args)>{}, t);
    // }
    // template <traits::TupleType TupleFn>
    // requires(sizeof...(Args) > 1 &&
    //          requires(Tuple t, TupleFn fn) {
    //              std::get<0>(t.view()) | fn.template get<0>();
    //          }) friend constexpr auto
    // operator|(Tuple& t, TupleFn&& f)
    // {
    //     return [&t]<auto... Is>(std::index_sequence<Is...>, auto&& fn)
    //     {
    //         return detail::makeTuple((view<Is>(t) | fn.template get<Is>())...);
    //     }
    //     (std::make_index_sequence<sizeof...(Args)>{}, FWD(f));
    // }
};

template <typename... Args>
Tuple(Args&&...) -> Tuple<Args...>;

template <typename... Args>
Tuple(tag, Args&&...) -> Tuple<Args...>;

template <std::size_t I, traits::TupleType C>
constexpr decltype(auto) get(C&& c)
{
    if constexpr (traits::OwningTuple<C>)
        return get<I>(FWD(c).as_Container());
    else
        return get<I>(FWD(c).as_ViewTuple());
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

namespace detail
{
template <typename... Args>
auto makeTuple(Args&&... args)
{
    return Tuple<Args...>(tag{}, FWD(args)...);
}
} // namespace detail

} // namespace ccs::field::tuple
