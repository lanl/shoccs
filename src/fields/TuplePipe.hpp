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
        return transform([f](auto&& e) { return e | f; }, FWD(u));
    }

    template <typename U, traits::TuplePipeableOver<U> F>
    requires std::derived_from<std::remove_cvref_t<U>, Type> friend constexpr auto
    operator|(U&& u, F&& f)
    {
        return transform([](auto&& e, auto fn) { return FWD(e) | fn; }, FWD(u), FWD(f));
    }
};

} // namespace ccs::field::tuple::lazy