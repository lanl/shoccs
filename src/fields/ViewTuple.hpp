#pragma once

#include <range/v3/view/all.hpp>

#include "TupleMath.hpp"
#include "TuplePipe.hpp"
#include "TupleUtils.hpp"

#include <iostream>

namespace ccs::field::tuple
{
// this base class stores the view of (or pointer to) the data in a tuple.  It should be
// inherited before AsView to ensure the tuple is initialized correctly.
template <typename... Args>
struct ViewBaseTuple {
private:
    using Type = ViewBaseTuple<Args...>;

public:
    std::tuple<std::add_pointer_t<std::remove_reference_t<Args>>...> v;

    ViewBaseTuple() = default;
    ViewBaseTuple(Args&&... args) : v{FWD(args)...} {};

    ViewBaseTuple(const ViewBaseTuple&) = default;
    ViewBaseTuple(ViewBaseTuple&&) = default;

    ViewBaseTuple& operator=(const ViewBaseTuple&) = default;
    ViewBaseTuple& operator=(ViewBaseTuple&&) = default;

    template <typename... C>
    ViewBaseTuple(ContainerTuple<C...>& x)
        : v{tuple_map([](auto&& a) { return std::addressof(FWD(a)); }, x.c)}
    {
    }

    template <typename... C>
    ViewBaseTuple& operator=(ContainerTuple<C...>& x)
    {
        v = tuple_map([](auto&& a) { return std::addressof(FWD(a)); }, x.c);
        return *this;
    }

    Type& as_ViewBaseTuple() & { return *this; }
    const Type& as_ViewBaseTuple() const& { return *this; }
    Type&& as_ViewBaseTuple() && { return MOVE(*this); }
};

template <All... Args>
struct ViewBaseTuple<Args...> {
private:
    using Type = ViewBaseTuple<Args...>;
    static constexpr bool Output = (traits::AnyOutputRange<Args> && ...);

public:
    std::tuple<vs::all_t<Args>...> v;

    ViewBaseTuple() = default;
    ViewBaseTuple(const ViewBaseTuple&) = default;
    ViewBaseTuple(ViewBaseTuple&&) = default;
    // defaults are triggered when Output is false
    ViewBaseTuple& operator=(const ViewBaseTuple&) = default;
    ViewBaseTuple& operator=(ViewBaseTuple&&) = default;

    explicit ViewBaseTuple(Args&&... args) requires(sizeof...(Args) > 0)
        : v{vs::all(FWD(args))...} {};

    template <traits::NonTupleRange... Ranges>
    requires(std::constructible_from<Args, Ranges>&&...) explicit ViewBaseTuple(
        Ranges&&... args)
        : v{vs::all(Args{FWD(args)})...}
    {
    }

    template <traits::TupleLike T>
    explicit ViewBaseTuple(T&& t) : v{tuple_map(vs::all, FWD(t))}
    {
    }

    ViewBaseTuple& operator=(const ViewBaseTuple& x) requires Output
    {
        resize_and_copy(*this, x);
        return *this;
    }

    ViewBaseTuple& operator=(ViewBaseTuple&& x) requires Output
    {
        resize_and_copy(*this, MOVE(x));
        return *this;
    }

    // When assigning from a ContainerTuple, make sure we don't copy anything and instead
    // just take news views
    template <traits::OwningTuple C>
    ViewBaseTuple& operator=(C&& c)
    {
        v = tuple_map(vs::all, FWD(c));
        return *this;
    }

    // This overload is only reached when called on a Tuple that does not
    // have a corresponding container.  Thus, this call results in a copy operation
    // rather than simply resetting the view to the container
    template <typename T>
        requires(!traits::OwningTuple<T>) &&
        traits::OutputTuple<ViewBaseTuple, T> ViewBaseTuple& operator=(T&& t)
    {
        // calling here
        resize_and_copy(*this, FWD(t));
        return *this;
    }

    Type& as_ViewBaseTuple() & { return *this; }
    const Type& as_ViewBaseTuple() const& { return *this; }
    Type&& as_ViewBaseTuple() && { return MOVE(*this); }
};

template <typename... Args>
ViewBaseTuple(Args&&...) -> ViewBaseTuple<Args...>;

template <std::size_t I, traits::ViewBaseTupleType V>
constexpr decltype(auto) get(V&& v) noexcept
{
    return std::get<I>(FWD(v).v);
}

} // namespace ccs::field::tuple

// specialize tuple_size
namespace std
{
template <typename... Args>
struct tuple_size<ccs::field::tuple::ViewBaseTuple<Args...>>
    : std::integral_constant<size_t, sizeof...(Args)> {
};

template <size_t I, typename... Args>
struct tuple_element<I, ccs::field::tuple::ViewBaseTuple<Args...>>
    : tuple_element<I, decltype(declval<ccs::field::tuple::ViewBaseTuple<Args...>>().v)> {
};
} // namespace std

namespace ccs::field::tuple
{

// this base class allows for treating tuples of one parameters directly as ranges
template <typename... Args>
struct AsView {

    AsView() = default;

    AsView(const AsView&) = default;
    AsView(AsView&&) = default;
    AsView& operator=(const AsView&) = default;
    AsView& operator=(AsView&&) = default;

    template <typename... T>
    AsView(T&&...)
    {
    }

    template <typename... T>
    AsView& operator=(T&&...)
    {
        return *this;
    }
};

template <All A>
struct AsView<A> : vs::all_t<A> {
private:
    using View = vs::all_t<A>;

public:
    AsView() = default;
    AsView(const AsView&) = default;
    AsView(AsView&&) = default;
    AsView& operator=(const AsView&) = default;
    AsView& operator=(AsView&&) = default;

    AsView(A&& a) : View(vs::all(FWD(a))) {}

    template <traits::TupleLike T>
    AsView(T&& t) : View(get<0>(t))
    {
    }

    AsView& operator=(A&& a)
    {
        View::operator=(vs::all(FWD(a)));
        return *this;
    }

    template <traits::TupleLike T>
    AsView& operator=(T&& t)
    {
        View::operator=(get<0>(t));
        return *this;
    }
};

template <typename... Args>
AsView(Args&&...) -> AsView<Args...>;

template <std::size_t I, traits::ViewTupleType V>
constexpr decltype(auto) get(V&& v) noexcept;

// r_tuple's inherit from this base class which combines AsView/base_view_tuple
// into a workable unified abstraction
template <typename... Args>
struct ViewTuple : ViewBaseTuple<Args...>,
                   AsView<Args...>,
                   field::tuple::lazy::ViewMath<ViewTuple<Args...>>,
                   field::tuple::lazy::ViewPipe<ViewTuple<Args...>> {
private:
    using Base_Tup = ViewBaseTuple<Args...>;
    using As_View = AsView<Args...>;
    using Type = ViewTuple<Args...>;
    static constexpr bool Output = (traits::AnyOutputRange<Args> && ...);

    friend class ViewMathAccess;
    friend class ViewPipeAccess;

public:
    explicit ViewTuple() = default;
    explicit ViewTuple(Args&&... args) requires(sizeof...(Args) > 0)
        : Base_Tup{FWD(args)...}, As_View{*this}
    {
    }

    // Forward all construction to ViewBaseTuple
    template <typename... T>
    requires std::constructible_from<Base_Tup, T...> explicit ViewTuple(T&&... t)
        : Base_Tup{FWD(t)...}, As_View{*this}
    {
    }

    // Foward all assignment to ViewBaseTuple
    template <typename T>
    requires std::is_assignable_v<Base_Tup&, T> ViewTuple& operator=(T&& t)
    {
        Base_Tup::operator=(FWD(t));
        As_View::operator=(*this);
        return *this;
    }

    ViewTuple& as_ViewTuple() & { return *this; }
    const ViewTuple& as_ViewTuple() const& { return *this; }
    ViewTuple&& as_ViewTuple() && { return MOVE(*this); }
};

template <typename... Args>
ViewTuple(Args&&...) -> ViewTuple<Args...>;

template <std::size_t I, traits::ViewTupleType V>
constexpr decltype(auto) get(V&& v) noexcept
{
    return get<I>(FWD(v).as_ViewBaseTuple());
}

} // namespace ccs::field::tuple

// specialize tuple_size
namespace std
{
template <typename... Args>
struct tuple_size<ccs::field::tuple::ViewTuple<Args...>>
    : std::integral_constant<size_t, sizeof...(Args)> {
};

template <size_t I, typename... Args>
struct tuple_element<I, ccs::field::tuple::ViewTuple<Args...>>
    : tuple_element<I, ccs::field::tuple::ViewBaseTuple<Args...>> {
};
} // namespace std