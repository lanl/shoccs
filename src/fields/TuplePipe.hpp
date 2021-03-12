#pragma once

#include "TupleUtils.hpp"

namespace ccs::field::tuple::lazy
{
template <typename T>
struct ViewPipe;

class ViewPipeAccess
{
    template <typename T>
    friend class ViewPipe;
};

template <typename>
struct ViewPipe {
};

template <template <typename...> typename T, typename... Ts>
requires(sizeof...(Ts) > 1) struct ViewPipe<T<Ts...>> {
private:
    using Type = T<Ts...>;

    template <traits::TupleLike U, traits::PipeableOver<U> F>
    requires std::derived_from<std::remove_cvref_t<U>, Type> friend constexpr auto
    operator|(U&& u, F f)
    {
        return transform([f](auto&& e) { return FWD(e) | f; }, FWD(u));
    }

    template <traits::TupleLike U, traits::TuplePipeableOver<U> F>
    requires std::derived_from<std::remove_cvref_t<U>, Type> friend constexpr auto
    operator|(U&& u, F&& f)
    {
        return transform([](auto&& e, auto fn) { return FWD(e) | fn; }, FWD(u), FWD(f));
    }

    template <typename ViewFn, traits::TupleLike U>
    requires std::derived_from<std::remove_cvref_t<U>, Type> friend constexpr auto
    operator|(vs::view_closure<ViewFn> f, U&& u)
    {
        return transform([f](auto&& e) { return f | FWD(e); }, FWD(u));
    }

    // provide hooks for some non-tuple like entities if they define certain memeber
    // properties
    //

    //
    // as_tuple is a defined on Tuples so this method will always pull out the tuple base
    // class
    //
    // template <typename U, typename F>
    //     requires std::derived_from<std::remove_cvref_t<U>, Type> &&
    //     (!traits::TupleLike<U>)&&requires(U u, F f)
    // {
    //     u.as_Tuple() | f;
    // }
    // friend constexpr auto operator|(U&& u, F&& f) { return FWD(u).as_Tuple() | FWD(f);
    // }

    //
    // to_tuple allows for a class to customize its tuple representation if as_tuple is
    // not sufficient
    //
    template <typename U, typename F>
        requires std::derived_from<std::remove_cvref_t<U>, Type> &&
        (!traits::TupleLike<U>)&&requires(U u, F f)
    {
        u.to_Tuple() | f;
    }
    friend constexpr auto operator|(U&& u, F&& f) { return FWD(u).to_Tuple() | FWD(f); }
};

} // namespace ccs::field::tuple::lazy