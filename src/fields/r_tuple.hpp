#pragma once

#include "indexing.hpp"
#include "lazy_tuple_math.hpp"
#include "r_tuple_fwd.hpp"
#include <tuple>

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/common.hpp>

#include <iostream>

namespace ccs
{

template <int I, typename R>
constexpr decltype(auto) view(R&& r)
{
    constexpr int Idx = I < r.N ? I : r.N - 1;
    return std::get<Idx>(FWD(r).view());
}

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

template <typename C, rs::input_range R>
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
} // namespace detail

template <typename T>
concept All = rs::range<T&>&& rs::viewable_range<T>;

namespace detail
{

using ::ccs::traits::Non_Tuple_Input_Range;

template <typename... Args, typename T>
auto container_from_container(const T& t)
{
    return [&t]<auto... Is>(std::index_sequence<Is...>)
    {
        // prefer to delegate to direct construction rather than building
        // from ranges.
        constexpr bool direct = requires(T t)
        {
            std::tuple<Args...>{Args{t.template get<Is>()}...};
        };

        if constexpr (direct)
            return std::tuple<Args...>{Args{t.template get<Is>()}...};
        else
            return std::tuple<Args...>{
                Args{rs::begin(t.template get<Is>()), rs::end(t.template get<Is>())}...};
    }
    (std::make_index_sequence<sizeof...(Args)>{});
}

template <typename... Args, typename T>
auto container_from_rtuple(const T& t)
{
    return [&t]<auto... Is>(std::index_sequence<Is...>)
    {
        auto x = std::tuple{vs::common(view<Is>(t))...};
        return std::tuple<Args...>{
            Args{rs::begin(std::get<Is>(x)), rs::end(std::get<Is>(x))}...};
    }
    (std::make_index_sequence<sizeof...(Args)>{});
}

// Base clase for r_tuples which own the containers associated with the data
// i.e vectors and spans
template <typename... Args>
struct container_tuple : lazy::container_math_crtp<container_tuple<Args...>> {
private:
    friend class container_math_access;
    using Type = container_tuple<Args...>;

public:
    static constexpr auto container_size = sizeof...(Args);

    std::tuple<Args...> c;

    container_tuple() = default;
    container_tuple(Args&&... args) : c{FWD(args)...} {}

    // allow for constructing and assigning from input_ranges
    template <rs::input_range... Ranges>
    container_tuple(Ranges&&... r) : c{Args{rs::begin(r), rs::end(r)}...}
    {
    }

    template <rs::input_range... Ranges>
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
    template <::ccs::traits::R_Tuple T>
    requires requires(T t)
    {
        container_from_rtuple<Args...>(t);
    }
    container_tuple(T&& t) : c{container_from_rtuple<Args...>(t)} {}

    template <::ccs::traits::R_Tuple T>
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

    view_tuple() = default;
    view_tuple(Args&&... args) : Base_Tup{FWD(args)...}, As_View{Base_Tup::view()} {}

    template <rs::input_range... Ranges>
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
};

} // namespace detail

template <typename... Args>
struct r_tuple : detail::container_tuple<Args...>, detail::view_tuple<Args&...> {
    using Container = detail::container_tuple<Args...>;
    using View = detail::view_tuple<Args&...>;
    using Type = r_tuple<Args...>;

    r_tuple() = default;

    r_tuple(Args&&... args) : Container{FWD(args)...}, View{this->container()} {}

    template <rs::input_range... R>
    r_tuple(R&&... r) : Container{FWD(r)...}, View{this->container()}
    {
    }

    template <traits::R_Tuple R>
    r_tuple(R&& r) : Container{FWD(r)}, View{this->container()}
    {
    }

    template <traits::Owning_R_Tuple R>
    r_tuple(R&& r) : Container{FWD(r).container()}, View{this->container()}
    {
    }

    template <traits::R_Tuple R>
    r_tuple& operator=(R&& r)
    {
        Container::operator=(FWD(r));
        View::operator=(this->container());
        return *this;
    }

    template <traits::Owning_R_Tuple R>
    r_tuple& operator=(R&& r)
    {
        Container::operator=(FWD(r).container());
        View::operator=(this->container());
        return *this;
    }

    // need to define custom copy and move construction/assignment here
    r_tuple(const r_tuple& r) : Container{r.container()}, View{this->container()} {}
    r_tuple& operator=(const r_tuple& r)
    {
        Container::operator=(r.container());
        View::operator=(this->container());
        return *this;
    }

    r_tuple(r_tuple&& r) noexcept
        : Container{MOVE(r.container())}, View{this->container()}
    {
    }

    r_tuple& operator=(r_tuple&& r) noexcept
    {
        Container::operator=(MOVE(r.container()));
        View::operator=(this->container());
        return *this;
    }

    template <traits::Non_Tuple_Input_Range... R>
    r_tuple& operator=(R&&... r)
    {
        Container::operator=(FWD(r)...);
        View::operator=(this->container());
        return *this;
    }
};

template <All... Args>
struct r_tuple<Args...> : detail::view_tuple<Args...> {
    using View = detail::view_tuple<Args...>;
    using Type = r_tuple<Args...>;

    r_tuple(Args&&... args) : View{FWD(args)...} {};

    r_tuple() = default;
};

template <typename... Args>
r_tuple(Args&&...) -> r_tuple<Args...>;

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

    template <traits::Directional_Field T>
    requires(!std::same_as<Type, std::remove_cvref_t<T>>) directional_field(T&& t)
        : View{FWD(t).as_r_tuple()}, Bounds{FWD(t).extents()}
    {
    }

    template <traits::Directional_Field T>
    requires(!std::same_as<Type, std::remove_cvref_t<T>>) directional_field&
    operator=(T&& t)
    {
        View::operator=(FWD(t).as_r_tuple());
        Bounds::operator=(FWD(t).extents());
        return *this;
    }

    r_tuple<R>& as_r_tuple() { return *this; }
    const r_tuple<R>& as_r_tuple() const { return *this; }
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

template <typename U, typename V, typename W, typename... Args>
class tuple_composite : public r_tuple<r_tuple<directional_field<U, 0>,
                                               directional_field<V, 1>,
                                               directional_field<W, 2>>,
                                       r_tuple<Args...>>
{
    using Type = tuple_composite<U, V, W, Args...>;
    using Domain = r_tuple<directional_field<U, 0>,
                           directional_field<V, 1>,
                           directional_field<W, 2>>;
    using Object = r_tuple<Args...>;

public:
    // directional_composite() = default;
};

template <typename U, typename V, typename W, typename... Args>
tuple_composite(U&&, V&&, W&&, Args&&...) -> tuple_composite<U, V, W, Args...>;

} // namespace ccs