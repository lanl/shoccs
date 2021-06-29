#pragma once

#include "tuple_utils.hpp"

namespace ccs::detail
{
template <typename T>
struct tuple_pipe;

class tuple_pipe_access
{
    template <typename T>
    friend class tuple_pipe;
};

template <typename>
struct tuple_pipe {
};

template <template <typename...> typename T, typename... Ts>
    requires(sizeof...(Ts) > 1)
struct tuple_pipe<T<Ts...>> {
private:
    using Type = T<Ts...>;

    template <TupleLike U, PipeableOver<U> F>
        requires(std::derived_from<std::remove_cvref_t<U>, Type> && !TupleLike<F>)
    friend constexpr auto operator|(U&& u, F f)
    {
        return transform([f](auto&& e) { return FWD(e) | f; }, FWD(u));
    }

    template <TupleLike U, TuplePipeableOver<U> F>
        requires std::derived_from<std::remove_cvref_t<U>, Type>
    friend constexpr auto operator|(U&& u, F&& f)
    {
        // Assume that if both are nested then we will preserve the structure
        // otherwise join the result
        if constexpr (NestedTuple<U> && NestedTuple<F>)
            return transform(
                [](auto&& e, auto fn) { return FWD(e) | fn; }, FWD(u), FWD(f));
        else {
            return join(
                transform([](auto&& e, auto fn) { return FWD(e) | fn; }, FWD(u), FWD(f)));
        }
    }

    template <typename ViewFn, TupleLike U>
        requires std::derived_from<std::remove_cvref_t<U>, Type>
    friend constexpr auto operator|(vs::view_closure<ViewFn> f, U&& u)
    {
        return transform([f](auto&& e) { return f | FWD(e); }, FWD(u));
    }

    // Some of the view_closures are meant to be applied to the whole structure
    // rather than recursively on the components
    template <TupleLike U, typename ViewFn>
        requires(std::derived_from<std::remove_cvref_t<U>, Type> &&
                 !PipeableOver<vs::view_closure<ViewFn>, U>)
    friend constexpr auto operator|(U&& u, vs::view_closure<ViewFn> f)
    {
        return f(FWD(u));
    }

    template <TupleLike U, TupleLike F>
        requires(std::derived_from<std::remove_cvref_t<U>, Type> && !SimilarTuples<U, F>)
    friend constexpr auto operator|(U&& u, F&& f)
    {
        return join(transform([&u](auto fn) { return u | fn; }, FWD(f)));
    }

    // provide hooks for some non-tuple like entities if they define certain memeber
    // properties
    //

    //
    // as_tuple is a defined on Tuples so this method will always pull out the tuple
    // base class
    //
    // template <typename U, typename F>
    //     requires std::derived_from<std::remove_cvref_t<U>, Type> &&
    //     (!TupleLike<U>)&&requires(U u, F f)
    // {
    //     u.as_Tuple() | f;
    // }
    // friend constexpr auto operator|(U&& u, F&& f) { return FWD(u).as_Tuple() |
    // FWD(f);
    // }

    //
    // to_tuple allows for a class to customize its tuple representation if as_tuple
    // is not sufficient
    //
    template <typename U, typename F>
        requires std::derived_from<std::remove_cvref_t<U>, Type> &&
            (!TupleLike<U>)&&requires(U u, F f)
        {
            u.to_Tuple() | f;
        }
        friend constexpr auto operator|(U&& u, F&& f)
        {
            return FWD(u).to_Tuple() | FWD(f);
        }
};

} // namespace ccs::detail
