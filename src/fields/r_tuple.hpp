#pragma once

#include "indexing.hpp"
#include "lazy_tuple_math.hpp"
#include "r_tuple_fwd.hpp"
#include <tuple>

#include <range/v3/view/all.hpp>

namespace ccs
{
namespace detail
{
template <typename T>
using all_t = decltype(vs::all(std::declval<T>()));

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
} // namespace detail

template <typename T>
concept All = rs::range<T&>&& rs::viewable_range<T>;

namespace detail
{
// Base clase for r_tuples which own the containers associated with the data
// i.e vectors and spans
template <typename... Args>
struct container_tuple : lazy::container_math_crtp<container_tuple<Args...>> {
private:
    friend class container_math_access;
    using Type = container_tuple<Args...>;

public:
    std::tuple<Args...> c;

    container_tuple(Args&&... args) : c{FWD(args)...} {}

    template <rs::input_range... Ranges>
    requires(std::constructible_from<Args,
                                     decltype(rs::begin(std::declval<Ranges>())),
                                     decltype(rs::begin(std::declval<Ranges>()))>&&...)
        container_tuple(Ranges&&... r)
        : c{Args{rs::begin(r), rs::end(r)}...}
    {
    }

    template <rs::input_range... Ranges>
    requires(std::constructible_from<Args,
                                     decltype(rs::begin(std::declval<Ranges>())),
                                     decltype(rs::begin(std::declval<Ranges>()))>&&...)
        container_tuple&
        operator=(Ranges&&... r)
    {
        c = std::tuple{Args{rs::begin(r), rs::end(r)}...};
        return *this;
    }

    container_tuple& container() { return *this; }
    const container_tuple& container() const { return *this; }
};

template <typename... Args>
container_tuple(Args&&...) -> container_tuple<std::remove_reference_t<Args>...>;

// this base class encapusulates the tuple aspect.  It should be inherited before
// as_view to ensure the tuple is initialized correctly
template <All... Args>
struct view_base_tuple {
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
template <All... Args>
struct as_view {
private:
    using Type = as_view<Args...>;

public:
    as_view() = default;

    template <typename... T>
    requires(!(std::same_as<Type, std::remove_cvref_t<T>> && ...)) as_view(T&&...)
    {
    }

    template <typename... T>
    requires(!(std::same_as<Type, std::remove_cvref_t<T>> && ...)) as_view&
    operator=(T&&...)
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
        static_cast<View&>(*this) = vs::all(FWD(a));
        return *this;
    }

    template <All R>
    as_view& operator=(std::tuple<R>& v)
    {
        static_cast<View&>(*this) = vs::all(std::get<0>(v));
        return *this;
    }
};

template <typename... Args>
as_view(Args&&...) -> as_view<Args...>;

// r_tuple's inherit from this base class which combines as_view/base_view_tuple
// into a workable unified abstraction
template <All... Args>
struct view_tuple : view_base_tuple<Args...>,
                    as_view<Args...>,
                    lazy::view_math_crtp<view_tuple<Args...>> {
private:
    using Base_Tup = view_base_tuple<Args...>;
    using As_View = as_view<Args...>;
    using Type = view_tuple<Args...>;

    friend class view_math_access;

public:
    static constexpr int N = sizeof...(Args);

    view_tuple(Args&&... args) : Base_Tup{FWD(args)...}, As_View{this->view()} {}

    template <rs::input_range... Ranges>
    view_tuple(Ranges&&... r) : Base_Tup{FWD(r)...}, As_View{this->view()}
    {
    }

    template <typename... C>
    view_tuple(container_tuple<C...>& x) : Base_Tup{x}, As_View{this->view()}
    {
    }

    template <typename... C>
    view_tuple& operator=(container_tuple<C...>& x)
    {
        static_cast<Base_Tup&>(*this) = x;
        static_cast<As_View&>(*this) = this->view();
        return *this;
    }
};

} // namespace detail

template <typename... Args>
struct r_tuple : detail::container_tuple<Args...>, detail::view_tuple<Args&...> {
    using Container = detail::container_tuple<Args...>;
    using View = detail::view_tuple<Args&...>;
    using Type = r_tuple<Args...>;

    r_tuple(Args&&... args) : Container{FWD(args)...}, View{this->container()} {}

    template <rs::input_range... R>
    requires(!(std::same_as<Type, std::remove_cvref_t<R>> && ...)) r_tuple(R&&... r)
        : Container{FWD(r)...}, View{this->container()}
    {
    }

    // need to define custom copy and move construction/assignment here
    r_tuple(const r_tuple& r) : Container{r.container()}, View{this->container()} {}
    r_tuple& operator=(const r_tuple& r)
    {
        this->container() = r.container();
        static_cast<View&>(*this) = this->container();
        return *this;
    }

    r_tuple(r_tuple&& r) : Container{MOVE(r.container())}, View{this->container()} {}

    r_tuple& operator=(r_tuple&& r)
    {
        this->container() = MOVE(r.container());
        static_cast<View&>(*this) = this->container();
        return *this;
    }

    template <rs::input_range... R>
    requires(!(std::same_as<Type, std::remove_cvref_t<R>> && ...)) r_tuple&
    operator=(R&&... r)
    {
        this->container().operator=(FWD(r)...);
        static_cast<View&>(*this) = this->container();
        return *this;
    }
};

template <All... Args>
struct r_tuple<Args...> : detail::view_tuple<Args...> {
    using View = detail::view_tuple<Args...>;
    using Type = r_tuple<Args...>;

    r_tuple(Args&&... args) : View{FWD(args)...} {};
};

template <typename... Args>
r_tuple(Args&&...) -> r_tuple<Args...>;

template <int I, typename R>
constexpr decltype(auto) view(R&& r)
{
    constexpr int Idx = I < r.N ? I : r.N - 1;
    return std::get<Idx>(FWD(r).view());
}

template <typename R, int I>
class directional_field : public r_tuple<R>, public index::bounds<I>
{
    using Type = directional_field<R, I>;
    using View = r_tuple<R>;
    using Bounds = index::bounds<I>;

public:
    directional_field() = default;

    template <typename T>
    requires std::constructible_from<View, T> directional_field(T&& t, const int3& bounds)
        : View{FWD(t)}, Bounds{bounds}
    {
    }

    template <typename T>
    requires std::constructible_from<View, T>
    directional_field(lit<I>, T&& t, const int3& bounds) : View{FWD(t)}, Bounds{bounds}
    {
    }
};

template <typename R, int I>
directional_field(lit<I>, R&&, const int3&) -> directional_field<R, I>;

template <typename T, int I>
using owning_field = directional_field<std::vector<T>, I>;

using x_field = owning_field<real, 0>;
using y_field = owning_field<real, 1>;
using z_field = owning_field<real, 2>;

template <typename R, int I, typename... Args>
class directional_composite
    : public r_tuple<r_tuple<directional_field<R, I>>, r_tuple<Args...>>
{
    using Type = directional_composite<R, I, Args...>;
    using Domain = r_tuple<directional_field<R, I>>;
    using Object = r_tuple<Args...>;

public:
    // directional_composite() = default;
};

template <typename R, int I, typename... Args>
directional_composite(R&&, Args&&...) -> directional_composite<R, I, Args...>;

} // namespace ccs