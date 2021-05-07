#pragma once

#include "tuple_utils.hpp"

namespace ccs
{

// Base clase for r_tuples which own the containers associated with the data
// i.e vectors or nested tuples
template <typename... Args>
struct container_tuple {

    static constexpr auto container_size = sizeof...(Args);

    std::tuple<Args...> c;

    container_tuple() = default;
    constexpr container_tuple(Args&&... args) : c{FWD(args)...} {}

    template <typename... T>
        requires(std::constructible_from<Args, T>&&...)
    constexpr container_tuple(T&&... args) : c(FWD(args)...) {}

    // allow for constructing and assigning from input_ranges
    template <NonTupleRange... Ranges>
        requires(ConstructibleFromRange<Args, Ranges>&&...)
    constexpr container_tuple(Ranges&&... r) : c{Args{rs::begin(r), rs::end(r)}...} {}

    template <TupleToTuple<container_tuple> T>
    constexpr container_tuple(T&& t) : c{to<std::tuple<Args...>>(FWD(t))}
    {
    }

    template <typename T>
        requires OutputTuple<container_tuple, T>
    constexpr container_tuple& operator=(T&& t)
    {
        resize_and_copy(*this, FWD(t));
        return *this;
    }

    constexpr container_tuple& as_container() { return *this; }
    constexpr const container_tuple& as_container() const { return *this; }
};

template <typename... Args>
container_tuple(Args&&...) -> container_tuple<std::remove_reference_t<Args>...>;

template <std::size_t I, ContainerTuple C>
constexpr auto&& get(C&& c)
{
    return std::get<I>(FWD(c).c);
}
} // namespace ccs

// specialize tuple_size
namespace std
{
template <typename... Args>
struct tuple_size<ccs::container_tuple<Args...>>
    : std::integral_constant<size_t, sizeof...(Args)> {
};

template <size_t I, typename... Args>
struct tuple_element<I, ccs::container_tuple<Args...>>
    : tuple_element<I, tuple<Args...>> {
};

} // namespace std
