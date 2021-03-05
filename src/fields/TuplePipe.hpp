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

    template <typename U, traits::PipeableOver<U> F>
    requires std::derived_from<std::remove_cvref_t<U>, Type> friend constexpr auto
    operator|(U&& u, F f)
    {
        return map(FWD(u), [f](auto&& e) { return e | f; });
    }

    template <typename U, traits::TuplePipeableOver<U> F>
    requires std::derived_from<std::remove_cvref_t<U>, Type> friend constexpr auto
    operator|(U&& u, F f)
    {
        return map(
            FWD(u), [f]<auto I>(auto&& e) { return FWD(e) | get<I>(f); });
    }
};

} // namespace ccs::field::tuple::lazy