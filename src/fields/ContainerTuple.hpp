#pragma once

#include "TupleMath.hpp"
#include "TupleUtils.hpp"

namespace ccs::field::tuple
{

// Base clase for r_tuples which own the containers associated with the data
// i.e vectors and spans
template <typename... Args>
struct ContainerTuple : field::tuple::lazy::ContainerMath<ContainerTuple<Args...>> {
private:
    friend class ContainerAccess;
    using Type = ContainerTuple<Args...>;

public:
    static constexpr auto container_size = sizeof...(Args);

    std::tuple<Args...> c;

    ContainerTuple() = default;
    ContainerTuple(Args&&... args) : c{FWD(args)...} {}

    template <typename... T>
    requires(std::constructible_from<Args, T>&&...) explicit ContainerTuple(T&&... args)
        : c(FWD(args)...)
    {
    }

    // allow for constructing and assigning from input_ranges
    template <traits::Range... Ranges>
    ContainerTuple(Ranges&&... r) : c{Args{rs::begin(r), rs::end(r)}...}
    {
    }

    template <traits::Range... Ranges>
    ContainerTuple& operator=(Ranges&&... r)
    {
        static_assert(sizeof...(Args) == sizeof...(Ranges));

        [this]<auto... Is>(std::index_sequence<Is...>, auto&&... r)
        {
            (resize_and_copy(get<Is>(*this), FWD(r)), ...);
        }
        (std::make_index_sequence<sizeof...(Ranges)>{}, FWD(r)...);

        return *this;
    }

    // allow for constructing and assigning from container tuples of different types
    template <traits::OtherContainerTuple<Type> T>
    ContainerTuple(const T& t) : c{container_from_container<Args...>(t)}
    {
        static_assert(sizeof...(Args) == std::tuple_size_v<T>);
    }

    template <traits::OtherContainerTuple<Type> T>
    ContainerTuple& operator=(const T& t)
    {
        static_assert(sizeof...(Args) == std::tuple_size_v<T>);

        [ this, &t ]<auto... Is>(std::index_sequence<Is...>)
        {
            constexpr bool direct = requires(T t)
            {
                ((get<Is>(*this) = get<Is>(t)), ...);
            };
            if constexpr (direct)
                ((get<Is>(*this) = get<Is>(t)), ...);
            else
                (resize_and_copy(get<Is>(*this), get<Is>(t)), ...);
        }
        (std::make_index_sequence<sizeof...(Args)>{});

        return *this;
    }

    // allow for constructing and assigning from r_tuples
    template <traits::View_Tuple T>
    requires requires(T t)
    {
        container_from_view<Args...>(t);
    }
    ContainerTuple(T&& t) : c{container_from_view<Args...>(FWD(t))} {}

    template <traits::View_Tuple T>
    ContainerTuple& operator=(const T& t)
    {
        [ this, &t ]<auto... Is>(std::index_sequence<Is...>)
        {
            (resize_and_copy(get<Is>(*this), view<Is>(t)), ...);
        }
        (std::make_index_sequence<sizeof...(Args)>{});

        return *this;
    }

    template <Numeric T>
    ContainerTuple& operator=(T t)
    {
        [ this, t ]<auto... Is>(std::index_sequence<Is...>)
        {
            constexpr bool direct = requires(T t) { ((get<Is>(*this) = t), ...); };
            if constexpr (direct)
                ((get<Is>(*this) = t), ...);
            else
                (rs::fill(get<Is>(*this), t), ...);
        }
        (std::make_index_sequence<sizeof...(Args)>{});

        return *this;
    }

    ContainerTuple& container() { return *this; }
    const ContainerTuple& container() const { return *this; }
};

template <typename... Args>
ContainerTuple(Args&&...) -> ContainerTuple<std::remove_reference_t<Args>...>;

template<std::size_t I, traits::ContainerTupleType C>
auto&& get(C&& c) {
     return std::get<I>(FWD(c).c);
}
} // namespace ccs::field::tuple

// specialize tuple_size
namespace std
{
template <typename... Args>
struct tuple_size<ccs::field::tuple::ContainerTuple<Args...>>
    : std::integral_constant<size_t, sizeof...(Args)> {
};

template <size_t I, typename... Args>
struct tuple_element<I, ccs::field::tuple::ContainerTuple<Args...>>
    : tuple_element<I, tuple<Args...>> {
};

} // namespace std