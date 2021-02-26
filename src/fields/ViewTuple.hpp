#pragma once

#include <range/v3/view/all.hpp>

#include "TupleMath.hpp"
#include "TupleUtils.hpp"

#include <iostream>

namespace ccs::field::tuple
{
// Several utilities for working with container/view tuples
// template <typename T>
// using all_t = decltype(vs::all(std::declval<T>()));

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

    auto view() { return v; }
    auto view() const { return v; }
};

template <All... Args>
struct ViewBaseTuple<Args...> {
private:
    using Type = ViewBaseTuple<Args...>;

public:
    std::tuple<vs::all_t<Args>...> v;

    ViewBaseTuple() = default;
    // ViewBaseTuple(Args&&...args) : v{vs::all(FWD(args))...} {};

    template <traits::Non_Tuple_Input_Range... Ranges>
    ViewBaseTuple(Ranges&&... args) : v{vs::all(FWD(args))...}
    {
    }

    template <traits::TupleType T>
    ViewBaseTuple(T&& t) : v{tuple_map(vs::all, t.view())}
    {
    }

    template <typename... C>
    ViewBaseTuple(ContainerTuple<C...>& x) : v{tuple_map(vs::all, x.c)}
    {
    }

    template <typename... C>
    ViewBaseTuple& operator=(ContainerTuple<C...>& x)
    {
        v = tuple_map(vs::all, x.c);
        return *this;
    }

    // This overload is only reached when called on a Tuple that does not
    // have a corresponding container.  Thus, this call results in a copy operation
    // rather than simply resetting the view to the container
    template <All... Ranges>
    ViewBaseTuple& operator=(Ranges&&... args)
    {
        [this]<auto... Is>(std::index_sequence<Is...>, auto&&... r)
        {
            (resize_and_copy(std::get<Is>(v), r), ...);
        }
        (std::make_index_sequence<sizeof...(Args)>{}, FWD(args)...);
        // v = std::tuple{vs::all(args)...};
        return *this;
    }

    template <traits::TupleType R>
    requires(sizeof...(Args) == std::tuple_size_v<std::remove_cvref_t<R>>) ViewBaseTuple&
    operator=(R&& r)
    {
        [this]<auto... Is>(std::index_sequence<Is...>, auto&& r)
        {
            (resize_and_copy(std::get<Is>(v), std::get<Is>(FWD(r).view())), ...);
        }
        (std::make_index_sequence<sizeof...(Args)>{}, FWD(r));
        return *this;
    }

    template <Numeric T>
    ViewBaseTuple& operator=(T t)
    {
        [ this, t ]<auto... Is>(std::index_sequence<Is...>)
        {
            constexpr bool direct = requires(T t) { ((std::get<Is>(v) = t), ...); };
            if constexpr (direct)
                ((std::get<Is>(v) = t), ...);
            else
                (rs::fill(std::get<Is>(v), t), ...);
        }
        (std::make_index_sequence<sizeof...(Args)>{});
        return *this;
    }

    auto view() { return v; }
    auto view() const { return v; }
};

template <typename... Args>
ViewBaseTuple(Args&&...) -> ViewBaseTuple<Args...>;

template <std::size_t I, traits::ViewBaseTupleType V>
auto get(V&& v)
{
    std::cout << "type VBT: " << debug::type(FWD(v)) << std::endl;
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
    AsView(A&& a) : View(vs::all(FWD(a))) {}

    //template <typename R>
    AsView(std::tuple<vs::all_t<A>> v) : View(std::get<0>(v))
    {
    }

    AsView& operator=(A&& a)
    {
        View::operator=(vs::all(FWD(a)));
        return *this;
    }

    template <All R>
    AsView& operator=(std::tuple<R> v)
    {
        View::operator=(vs::all(std::get<0>(v)));
        return *this;
    }
};

template <typename... Args>
AsView(Args&&...) -> AsView<Args...>;

// r_tuple's inherit from this base class which combines AsView/base_view_tuple
// into a workable unified abstraction
template <typename... Args>
struct ViewTuple : ViewBaseTuple<Args...>,
                   AsView<Args...>,
                   field::tuple::lazy::ViewMath<ViewTuple<Args...>> {
private:
    using Base_Tup = ViewBaseTuple<Args...>;
    using As_View = AsView<Args...>;
    using Type = ViewTuple<Args...>;

    friend class ViewAccess;

public:
    static constexpr int N = sizeof...(Args);

    ViewTuple() = default;
    ViewTuple(Args&&... args) : Base_Tup{FWD(args)...}, As_View{Base_Tup::view()} {}

    template <traits::Non_Tuple_Input_Range... Ranges>
    ViewTuple(Ranges&&... r) : Base_Tup{FWD(r)...}, As_View{Base_Tup::view()}
    {
    }

    template <typename... C>
    ViewTuple(ContainerTuple<C...>& x) : Base_Tup{x}, As_View{Base_Tup::view()}
    {
    }

    template <traits::TupleType R>
    ViewTuple(R&& r) : Base_Tup{FWD(r)}, As_View{Base_Tup::view()}
    {
    }

    template <typename... C>
    ViewTuple& operator=(ContainerTuple<C...>& x)
    {
        Base_Tup::operator=(x);
        As_View::operator=(this->view());
        return *this;
    }

    template <Numeric T>
    ViewTuple& operator=(T t)
    {
        Base_Tup::operator=(t);
        return *this;
    }

    template <traits::TupleType R>
    ViewTuple& operator=(R&& r)
    {
        Base_Tup::operator=(FWD(r));
        return *this;
    }

    template <traits::Non_Tuple_Input_Range... R>
    ViewTuple& operator=(R&&... r)
    {
        Base_Tup::operator=(FWD(r)...);
        return *this;
    }

    ViewTuple& as_ViewTuple() { return *this; }
    const ViewTuple& as_ViewTuple() const { return *this; }

    Base_Tup& as_ViewBaseTuple() & { return static_cast<Base_Tup&>(*this); }
    const Base_Tup& as_ViewBaseTuple() const&
    {
        return static_cast<const Base_Tup&>(*this);
    }
    Base_Tup&& as_ViewBaseTuple() && { return static_cast<Base_Tup&&>(*this); }
};

template <std::size_t I, traits::ViewTupleType V>
decltype(auto) get(V&& v)
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