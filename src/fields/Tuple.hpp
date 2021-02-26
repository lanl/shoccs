#pragma once

#include "Tuple_fwd.hpp"

#include "TupleMath.hpp"
#include "indexing.hpp"

#include "TupleUtils.hpp"
#include "ContainerTuple.hpp"
#include "ViewTuple.hpp"

namespace ccs::field::tuple
{

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
    view_base_tuple(ContainerTuple<C...>& x)
        : v{tuple_map([](auto&& a) { return std::addressof(FWD(a)); }, x.c)}
    {
    }

    template <typename... C>
    view_base_tuple& operator=(ContainerTuple<C...>& x)
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
    view_base_tuple(ContainerTuple<C...>& x) : v{tuple_map(vs::all, x.c)}
    {
    }

    template <typename... C>
    view_base_tuple& operator=(ContainerTuple<C...>& x)
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
    view_tuple(ContainerTuple<C...>& x) : Base_Tup{x}, As_View{Base_Tup::view()}
    {
    }

    template <traits::TupleType R>
    view_tuple(R&& r) : Base_Tup{FWD(r)}, As_View{Base_Tup::view()}
    {
    }

    template <typename... C>
    view_tuple& operator=(ContainerTuple<C...>& x)
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
struct Tuple : ContainerTuple<Args...>, view_tuple<Args&...> {
    using Container = ContainerTuple<Args...>;
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

    template <traits::Range... R>
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
