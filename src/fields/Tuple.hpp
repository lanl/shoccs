#pragma once

#include "Tuple_fwd.hpp"

#include "TupleMath.hpp"
#include "indexing.hpp"

#include <tuple>

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/common.hpp>

namespace ccs::field::tuple
{

template <int I, typename R>
constexpr decltype(auto) view(R&& r)
{
    constexpr int Idx = I < r.N ? I : r.N - 1;
    return std::get<Idx>(FWD(r).view());
}

// Several utilities for working with container/view tuples
template <typename T>
using all_t = decltype(vs::all(std::declval<T>()));

// Map a function `f` over the elements in tuple `t`.  Returns a new tuple.
template <typename F, typename T>
requires requires(F f, T t)
{
    f(std::get<0>(t));
}
constexpr auto tuple_map(F&& f, T&& t)
{
    return [&f, &t ]<auto... Is>(std::index_sequence<Is...>)
    {
        return std::tuple{f(std::get<Is>(t))...};
    }
    (std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>{});
}

// Given a container and and input range, attempt to resize the container.
// Either way, copy the input range into the container - may not be safe.
template <typename C, traits::range R>
void resize_and_copy(C& container, R&& r)
{
    constexpr bool can_resize_member = requires(C c, R r) { c.resize(r.size()); };
    constexpr bool can_resize_rs = requires(C c, R r) { c.resize(rs::size(r)); };
    if constexpr (can_resize_member)
        container.resize(r.size());
    else if constexpr (can_resize_rs)
        container.resize(rs::size(r));

    rs::copy(FWD(r), rs::begin(container));
}

// Construct and return a new container from an input container.  Preference
// is given to delegating to direct construction.  Falls back to construction
// via input ranges.
template <typename... Args, typename T>
auto container_from_container(const T& t)
{
    return [&t]<auto... Is>(std::index_sequence<Is...>)
    {
        // prefer to delegate to direct construction rather than building
        // from ranges.
        // Use () instead of {} to allow constructing vectors from from sizes
        constexpr bool direct = requires(T t)
        {
            std::tuple<Args...>{Args(t.template get<Is>())...};
        };

        if constexpr (direct)
            return std::tuple<Args...>{Args(t.template get<Is>())...};
        else
            return std::tuple<Args...>{
                Args{rs::begin(t.template get<Is>()), rs::end(t.template get<Is>())}...};
    }
    (std::make_index_sequence<sizeof...(Args)>{});
}

// Construct a container from a view by converting the view to a
// common range.  As above, prefer to delegate construction directly
// if possible.  This is helpful for constructing owning r_tuples with sizes.
template <typename... Args, typename T>
auto container_from_view(const T& t)
{
    return [&t]<auto... Is>(std::index_sequence<Is...>)
    {
        constexpr bool direct = requires(T t)
        {
            std::tuple<Args...>{Args(view<Is>(t))...};
        };

        if constexpr (direct) {
            return std::tuple<Args...>{Args(view<Is>(t))...};
        } else {
            auto x = std::tuple{vs::common(view<Is>(t))...};
            return std::tuple<Args...>{
                Args{rs::begin(std::get<Is>(x)), rs::end(std::get<Is>(x))}...};
        }
    }
    (std::make_index_sequence<sizeof...(Args)>{});
}

// Base clase for r_tuples which own the containers associated with the data
// i.e vectors and spans
template <typename... Args>
struct container_tuple : field::tuple::lazy::ContainerMath<container_tuple<Args...>> {
private:
    friend class ContainerAccess;
    using Type = container_tuple<Args...>;

public:
    static constexpr auto container_size = sizeof...(Args);

    std::tuple<Args...> c;

    container_tuple() = default;
    container_tuple(Args&&... args) : c{FWD(args)...} {}

    template <typename... T>
    requires(std::constructible_from<Args, T>&&...) container_tuple(T&&... args)
        : c(FWD(args)...)
    {
    }

    // allow for constructing and assigning from input_ranges
    template <traits::range... Ranges>
    container_tuple(Ranges&&... r) : c{Args{rs::begin(r), rs::end(r)}...}
    {
    }

    template <traits::range... Ranges>
    container_tuple& operator=(Ranges&&... r)
    {
        static_assert(sizeof...(Args) == sizeof...(Ranges));

        [this]<auto... Is>(std::index_sequence<Is...>, auto&&... r)
        {
            (resize_and_copy(std::get<Is>(c), FWD(r)), ...);
        }
        (std::make_index_sequence<sizeof...(Ranges)>{}, FWD(r)...);

        return *this;
    }

    // allow for constructing and assigning from container tuples of different types
    template <traits::Other_Container_Tuple<Type> T>
    container_tuple(const T& t) : c{container_from_container<Args...>(t)}
    {
        static_assert(container_size == t.container_size);
    }

    template <traits::Other_Container_Tuple<Type> T>
    container_tuple& operator=(const T& t)
    {
        static_assert(container_size == t.container_size);

        [ this, &t ]<auto... Is>(std::index_sequence<Is...>)
        {
            constexpr bool direct = requires(T t)
            {
                ((std::get<Is>(c) = t.template get<Is>()), ...);
            };
            if constexpr (direct)
                ((std::get<Is>(c) = t.template get<Is>()), ...);
            else
                (resize_and_copy(std::get<Is>(c), t.template get<Is>()), ...);
        }
        (std::make_index_sequence<container_size>{});

        return *this;
    }

    // allow for constructing and assigning from r_tuples
    template <traits::View_Tuple T>
    requires requires(T t)
    {
        container_from_view<Args...>(t);
    }
    container_tuple(T&& t) : c{container_from_view<Args...>(FWD(t))} {}

    template <traits::View_Tuple T>
    container_tuple& operator=(const T& t)
    {
        [ this, &t ]<auto... Is>(std::index_sequence<Is...>)
        {
            (resize_and_copy(std::get<Is>(c), view<Is>(t)), ...);
        }
        (std::make_index_sequence<container_size>{});

        return *this;
    }

    template <int I>
    const auto& get() const
    {
        return std::get<I>(c);
    }

    template <int I>
    auto& get()
    {
        return std::get<I>(c);
    }

    container_tuple& container() { return *this; }
    const container_tuple& container() const { return *this; }
};

template <typename... Args>
container_tuple(Args&&...) -> container_tuple<std::remove_reference_t<Args>...>;

// this base class stores the view of (or pointer to) the data in a tuple.  It should be
// inherited before as_view to ensure the tuple is initialized correctly.
template <typename... Args>
struct view_base_tuple {
private:
    using Type = view_base_tuple<Args...>;

public:
    std::tuple<std::add_pointer_t<std::remove_reference_t<Args>>...> v;

    view_base_tuple() = default;
    view_base_tuple(Args&&... args) : v{FWD(args)...} {};

    template <typename... C>
    view_base_tuple(container_tuple<C...>& x)
        : v{tuple_map([](auto&& a) { return std::addressof(FWD(a)); }, x.c)}
    {
    }

    template <typename... C>
    view_base_tuple& operator=(container_tuple<C...>& x)
    {
        v = tuple_map([](auto&& a) { return std::addressof(FWD(a)); }, x.c);
        return *this;
    }

    auto& view() { return v; }
    const auto& view() const { return v; }
};

template <All... Args>
struct view_base_tuple<Args...> {
private:
    using Type = view_base_tuple<Args...>;

public:
    std::tuple<all_t<Args>...> v;

    view_base_tuple() = default;
    // view_base_tuple(Args&&...args) : v{vs::all(FWD(args))...} {};

    template <All... Ranges>
    view_base_tuple(Ranges&&... args) : v{vs::all(FWD(args))...}
    {
    }

    template <typename... C>
    view_base_tuple(container_tuple<C...>& x) : v{tuple_map(vs::all, x.c)}
    {
    }

    template <All... Ranges>
    view_base_tuple& operator=(Ranges&&... args)
    {
        v = std::tuple{vs::all(args)...};
        return *this;
    }

    template <typename... C>
    view_base_tuple& operator=(container_tuple<C...>& x)
    {
        v = tuple_map(vs::all, x.c);
        return *this;
    }

    auto& view() { return v; }
    const auto& view() const { return v; }
};

template <typename... Args>
view_base_tuple(Args&&...) -> view_base_tuple<Args...>;

// this base class allows for treating tuples of one parameters directly as ranges
template <typename... Args>
struct as_view {

    as_view() = default;

    template <typename... T>
    as_view(T&&...)
    {
    }

    template <typename... T>
    as_view& operator=(T&&...)
    {
        return *this;
    }
};

template <All A>
struct as_view<A> : all_t<A> {
private:
    using View = all_t<A>;

public:
    as_view() = default;
    as_view(A&& a) : View(vs::all(FWD(a))) {}

    template <All R>
    as_view(std::tuple<R>& v) : View(vs::all(std::get<0>(v)))
    {
    }

    as_view& operator=(A&& a)
    {
        View::operator=(vs::all(FWD(a)));
        return *this;
    }

    template <All R>
    as_view& operator=(std::tuple<R>& v)
    {
        View::operator=(vs::all(std::get<0>(v)));
        return *this;
    }
};

template <typename... Args>
as_view(Args&&...) -> as_view<Args...>;

// r_tuple's inherit from this base class which combines as_view/base_view_tuple
// into a workable unified abstraction
template <typename... Args>
struct view_tuple : view_base_tuple<Args...>,
                    as_view<Args...>,
                    field::tuple::lazy::ViewMath<view_tuple<Args...>> {
private:
    using Base_Tup = view_base_tuple<Args...>;
    using As_View = as_view<Args...>;
    using Type = view_tuple<Args...>;

    friend class ViewAccess;

public:
    static constexpr int N = sizeof...(Args);

    view_tuple() = default;
    view_tuple(Args&&... args) : Base_Tup{FWD(args)...}, As_View{Base_Tup::view()} {}

    template <traits::range... Ranges>
    view_tuple(Ranges&&... r) : Base_Tup{FWD(r)...}, As_View{Base_Tup::view()}
    {
    }

    template <typename... C>
    view_tuple(container_tuple<C...>& x) : Base_Tup{x}, As_View{Base_Tup::view()}
    {
    }

    template <typename... C>
    view_tuple& operator=(container_tuple<C...>& x)
    {
        Base_Tup::operator=(x);
        As_View::operator=(this->view());
        return *this;
    }

    view_tuple& as_view_tuple() { return *this; }
    const view_tuple& as_view_tuple() const { return *this; }
};

// r_tuple for viewable ref components
template <typename... Args>
struct Tuple : container_tuple<Args...>, view_tuple<Args&...> {
    using Container = container_tuple<Args...>;
    using View = view_tuple<Args&...>;
    using Type = Tuple<Args...>;

    Tuple() = default;

    Tuple(Args&&... args) : Container{FWD(args)...}, View{this->container()} {}

    template <typename... T>
    requires(std::constructible_from<Args, T>&&...) Tuple(T&&... args)
        : Container{FWD(args)...}, View{this->container()}
    {
    }

    Tuple(tag, Args&&... args) : Container{FWD(args)...}, View{this->container()} {}

    template <typename... T>
    requires(std::constructible_from<Args, T>&&...) Tuple(tag, T&&... args)
        : Container{FWD(args)...}, View{this->container()}
    {
    }

    template <traits::range... R>
    Tuple(R&&... r) : Container{FWD(r)...}, View{this->container()}
    {
    }

    template <traits::R_Tuple R>
    Tuple(R&& r) : Container{FWD(r).as_view_tuple()}, View{this->container()}
    {
    }

    template <traits::Owning_R_Tuple R>
    Tuple(R&& r) : Container{FWD(r).container()}, View{this->container()}
    {
    }

    template <traits::R_Tuple R>
    Tuple& operator=(R&& r)
    {
        Container::operator=(FWD(r).as_view_tuple());
        View::operator=(this->container());
        return *this;
    }

    template <traits::Owning_R_Tuple R>
    Tuple& operator=(R&& r)
    {
        Container::operator=(FWD(r).container());
        View::operator=(this->container());
        return *this;
    }

    // need to define custom copy and move construction/assignment here
    Tuple(const Tuple& r) : Container{r.container()}, View{this->container()} {}
    Tuple& operator=(const Tuple& r)
    {
        Container::operator=(r.container());
        View::operator=(this->container());
        return *this;
    }

    Tuple(Tuple&& r) noexcept : Container{MOVE(r.container())}, View{this->container()} {}

    Tuple& operator=(Tuple&& r) noexcept
    {
        Container::operator=(MOVE(r.container()));
        View::operator=(this->container());
        return *this;
    }

    template <traits::Non_Tuple_Input_Range... R>
    Tuple& operator=(R&&... r)
    {
        Container::operator=(FWD(r)...);
        View::operator=(this->container());
        return *this;
    }
};

template <All... Args>
struct Tuple<Args...> : view_tuple<Args...> {
    using View = view_tuple<Args...>;
    using Type = Tuple<Args...>;

    Tuple(Args&&... args) : View{FWD(args)...} {};

    Tuple(tag, Args&&... args) : View{FWD(args)...} {};

    Tuple() = default;
};

template <typename... Args>
Tuple(Args&&...) -> Tuple<Args...>;

template <typename... Args>
Tuple(tag, Args&&...) -> Tuple<Args...>;

} // namespace ccs::field::tuple
