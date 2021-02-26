#pragma once

#include "Tuple_fwd.hpp"

#include "TupleMath.hpp"
#include "indexing.hpp"

#include <tuple>

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/copy_n.hpp>
#include <range/v3/algorithm/fill.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/common.hpp>
#include <range/v3/view/view.hpp>

namespace ccs::field::tuple
{

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
requires requires(C container, R r)
{
    rs::copy(r, rs::begin(container));
}
void resize_and_copy(C& container, R&& r)
{
    constexpr bool can_resize_member = requires(C c, R r) { c.resize(r.size()); };
    constexpr bool can_resize_rs = requires(C c, R r) { c.resize(rs::size(r)); };
    constexpr bool compare_sizes = requires(C c, R r) { rs::size(c) < rs::size(r); };
    if constexpr (can_resize_member) {
        container.resize(r.size());
        rs::copy(FWD(r), rs::begin(container));
    } else if constexpr (can_resize_rs) {
        container.resize(rs::size(r));
        rs::copy(FWD(r), rs::begin(container));
    } else if constexpr (compare_sizes) {
        auto min_sz = std::min(rs::size(container), rs::size(r));
        // note that copy_n takes an input iterator rather than a range
        rs::copy_n(rs::begin(r), min_sz, rs::begin(container));
    } else {
        rs::copy(FWD(r), rs::begin(container));
    }
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
        constexpr bool fromRange = requires(T t)
        {
            std::tuple<Args...>{
                Args{rs::begin(t.template get<Is>()), rs::end(t.template get<Is>())}...};
        };

        if constexpr (direct)
            return std::tuple<Args...>{Args(t.template get<Is>())...};
        else if constexpr (fromRange)
            return std::tuple<Args...>{
                Args{rs::begin(t.template get<Is>()), rs::end(t.template get<Is>())}...};
        else
            static_assert(true, "Cannot construct container");
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
    requires(std::constructible_from<Args, T>&&...) explicit container_tuple(T&&... args)
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

    template <Numeric T>
    container_tuple& operator=(T t)
    {
        [ this, t ]<auto... Is>(std::index_sequence<Is...>)
        {
            constexpr bool direct = requires(T t) { ((std::get<Is>(c) = t), ...); };
            if constexpr (direct)
                ((std::get<Is>(c) = t), ...);
            else
                (rs::fill(std::get<Is>(c), t), ...);
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

    auto view() { return v; }
    auto view() const { return v; }
};

template <All... Args>
struct view_base_tuple<Args...> {
private:
    using Type = view_base_tuple<Args...>;

public:
    std::tuple<all_t<Args>...> v;

    view_base_tuple() = default;
    // view_base_tuple(Args&&...args) : v{vs::all(FWD(args))...} {};

    template <traits::Non_Tuple_Input_Range... Ranges>
    view_base_tuple(Ranges&&... args) : v{vs::all(FWD(args))...}
    {
    }

    template <traits::TupleType T>
    view_base_tuple(T&& t) : v{tuple_map(vs::all, t.view())}
    {
    }

    template <typename... C>
    view_base_tuple(container_tuple<C...>& x) : v{tuple_map(vs::all, x.c)}
    {
    }

    template <typename... C>
    view_base_tuple& operator=(container_tuple<C...>& x)
    {
        v = tuple_map(vs::all, x.c);
        return *this;
    }

    // This overload is only reached when called on a Tuple that does not
    // have a corresponding container.  Thus, this call results in a copy operation
    // rather than simply resetting the view to the container
    template <All... Ranges>
    view_base_tuple& operator=(Ranges&&... args)
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
    requires(sizeof...(Args) ==
             std::tuple_size_v<std::remove_cvref_t<R>>) view_base_tuple&
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
    view_base_tuple& operator=(T t)
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
    as_view(std::tuple<R> v) : View(vs::all(std::get<0>(v)))
    {
    }

    as_view& operator=(A&& a)
    {
        View::operator=(vs::all(FWD(a)));
        return *this;
    }

    template <All R>
    as_view& operator=(std::tuple<R> v)
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

    template <traits::Non_Tuple_Input_Range... Ranges>
    view_tuple(Ranges&&... r) : Base_Tup{FWD(r)...}, As_View{Base_Tup::view()}
    {
    }

    template <typename... C>
    view_tuple(container_tuple<C...>& x) : Base_Tup{x}, As_View{Base_Tup::view()}
    {
    }

    template <traits::TupleType R>
    view_tuple(R&& r) : Base_Tup{FWD(r)}, As_View{Base_Tup::view()}
    {
    }

    template <typename... C>
    view_tuple& operator=(container_tuple<C...>& x)
    {
        Base_Tup::operator=(x);
        As_View::operator=(this->view());
        return *this;
    }

    template <Numeric T>
    view_tuple& operator=(T t)
    {
        Base_Tup::operator=(t);
        return *this;
    }

    template <traits::TupleType R>
    view_tuple& operator=(R&& r)
    {
        Base_Tup::operator=(FWD(r));
        return *this;
    }

    template <traits::Non_Tuple_Input_Range... R>
    view_tuple& operator=(R&&... r)
    {
        Base_Tup::operator=(FWD(r)...);
        return *this;
    }

    view_tuple& as_view_tuple() { return *this; }
    const view_tuple& as_view_tuple() const { return *this; }
};

namespace detail
{
template <typename... Args>
auto makeTuple(Args&&...);
}

// r_tuple for viewable ref components
template <typename... Args>
struct Tuple : container_tuple<Args...>, view_tuple<Args&...> {
    using Container = container_tuple<Args...>;
    using View = view_tuple<Args&...>;
    using Type = Tuple<Args...>;

    Tuple() = default;

    Tuple(Args&&... args) : Container{FWD(args)...}, View{this->container()} {}

    template <typename... T>
    requires((sizeof...(T) == sizeof...(Args)) &&
             (std::constructible_from<Args, T> && ...)) explicit Tuple(T&&... args)
        : Container{FWD(args)...}, View{this->container()}
    {
    }

    Tuple(tag, Args&&... args) : Container{FWD(args)...}, View{this->container()} {}

    template <typename... T>
    requires((sizeof...(T) == sizeof...(Args)) &&
             (std::constructible_from<Args, T> && ...)) explicit Tuple(tag, T&&... args)
        : Container{FWD(args)...}, View{this->container()}
    {
    }

    template <traits::range... R>
    Tuple(R&&... r) : Container{FWD(r)...}, View{this->container()}
    {
    }

    template <traits::TupleType R>
    Tuple(R&& r) : Container{FWD(r).as_view_tuple()}, View{this->container()}
    {
    }

    template <traits::Owning_Tuple R>
    Tuple(R&& r) : Container{FWD(r).container()}, View{this->container()}
    {
    }

    template <traits::TupleType R>
    Tuple& operator=(R&& r)
    {
        Container::operator=(FWD(r).as_view_tuple());
        View::operator=(this->container());
        return *this;
    }

    template <traits::Owning_Tuple R>
    Tuple& operator=(R&& r)
    {
        Container::operator=(FWD(r).container());
        View::operator=(this->container());
        return *this;
    }

    template <Numeric T>
    Tuple& operator=(T t)
    {
        Container::operator=(t);
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

    template <typename ViewFn>
    requires(sizeof...(Args) > 1 &&
             requires(Tuple t, vs::view_closure<ViewFn> f) {
                 std::get<0>(t.view()) | f;
             }) friend constexpr auto
    operator|(Tuple& t, vs::view_closure<ViewFn> f)
    {
        return [f]<auto... Is>(std::index_sequence<Is...>, auto& t)
        {
            return detail::makeTuple((view<Is>(t) | f)...);
        }
        (std::make_index_sequence<sizeof...(Args)>{}, t);
    }

    // allow for composing a tuple of ranges with a tuple of functions
    template <traits::TupleType TupleFn>
    requires(sizeof...(Args) > 1 &&
             requires(Tuple t, TupleFn f) {
                 std::get<0>(t.view()) | f.template get<0>();
             }) friend constexpr auto
    operator|(Tuple& t, TupleFn&& f)
    {
        return [&t]<auto... Is>(std::index_sequence<Is...>, auto&& fn)
        {
            return detail::makeTuple((view<Is>(t) | fn.template get<Is>())...);
        }
        (std::make_index_sequence<sizeof...(Args)>{}, FWD(f));
    }

    template <typename ViewFn>
    requires requires(vs::view_closure<ViewFn> f, Tuple t)
    {
        f | t.template get<0>();
    }
    friend constexpr auto operator|(vs::view_closure<ViewFn> f, Tuple t)
    {
        return [f]<auto... Is>(std::index_sequence<Is...>, auto&& t)
        {
            return detail::makeTuple((f | t.template get<Is>())...);
        }
        (std::make_index_sequence<sizeof...(Args)>{}, MOVE(t));
    }
};

//
// Non Owning Tuple
//
template <All... Args>
struct Tuple<Args...> : view_tuple<Args...> {
    using View = view_tuple<Args...>;
    using Type = Tuple<Args...>;

    Tuple(Args&&... args) : View{FWD(args)...} {};

    Tuple(tag, Args&&... args) : View{FWD(args)...} {};

    Tuple() = default;

    template <traits::TupleType T>
    requires(sizeof...(Args) == std::tuple_size_v<std::remove_cvref_t<T>>) Tuple(T&& t)
        : View{FWD(t)}
    {
    }

    template <Numeric T>
    Tuple& operator=(T t)
    {
        View::operator=(t);
        return *this;
    }

    template <traits::Non_Tuple_Input_Range... R>
    Tuple& operator=(R&&... r)
    {
        View::operator=(FWD(r)...);
        return *this;
    }

    template <traits::TupleType R>
    requires(sizeof...(Args) == std::tuple_size_v<std::remove_cvref_t<R>>) Tuple&
    operator=(R&& r)
    {
        View::operator=(FWD(r));
        return *this;
    }

    template <typename Fn>
    requires(sizeof...(Args) > 1 &&
             requires(Tuple t, Fn f) { std::get<0>(t.view()) | f; }) friend constexpr auto
    operator|(Tuple& t, Fn f)
    {
        return [f]<auto... Is>(std::index_sequence<Is...>, auto& t)
        {
            return detail::makeTuple((view<Is>(t) | f)...);
        }
        (std::make_index_sequence<sizeof...(Args)>{}, t);
    }
    template <traits::TupleType TupleFn>
    requires(sizeof...(Args) > 1 &&
             requires(Tuple t, TupleFn fn) {
                 std::get<0>(t.view()) | fn.template get<0>();
             }) friend constexpr auto
    operator|(Tuple& t, TupleFn&& f)
    {
        return [&t]<auto... Is>(std::index_sequence<Is...>, auto&& fn)
        {
            return detail::makeTuple((view<Is>(t) | fn.template get<Is>())...);
        }
        (std::make_index_sequence<sizeof...(Args)>{}, FWD(f));
    }
};

template <typename... Args>
Tuple(Args&&...) -> Tuple<Args...>;

template <typename... Args>
Tuple(tag, Args&&...) -> Tuple<Args...>;

template <int I, typename R>
constexpr auto view(R&& r)
{
    constexpr auto sz = std::tuple_size_v<std::remove_cvref_t<decltype(r.view())>>;
    constexpr auto Idx = I < sz ? I : sz - 1;
    return std::get<Idx>(FWD(r).view());
}

namespace detail
{
template <typename... Args>
auto makeTuple(Args&&... args)
{
    return Tuple<Args...>(tag{}, FWD(args)...);
}
} // namespace detail

} // namespace ccs::field::tuple
