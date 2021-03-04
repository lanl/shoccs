#pragma once

#include "TupleMath.hpp"
#include "TupleUtils.hpp"

namespace ccs::field::tuple
{

// Base clase for r_tuples which own the containers associated with the data
// i.e vectors or nested tuples
template <typename... Args>
struct ContainerTuple {
private:
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
    template <traits::NonTupleRange... Ranges>
    requires(traits::FromRange<Ranges, Args>&&...) ContainerTuple(Ranges&&... r)
        : c{Args{rs::begin(r), rs::end(r)}...}
    {
    }

    template <traits::FromTuple<ContainerTuple> T>
    ContainerTuple(T&& t) : c{to_tuple<ContainerTuple>(FWD(t))}
    {
    }

    template <typename T>
    requires traits::OutputTuple<ContainerTuple, T> ContainerTuple& operator=(T&& t)
    {
        resize_and_copy(*this, FWD(t));
        return *this;
    }

    ContainerTuple& as_Container() { return *this; }
    const ContainerTuple& as_Container() const { return *this; }
};

template <typename... Args>
ContainerTuple(Args&&...) -> ContainerTuple<std::remove_reference_t<Args>...>;

template <std::size_t I, traits::ContainerTupleType C>
auto&& get(C&& c)
{
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