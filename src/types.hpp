#pragma once

#include <array>
#include <concepts>
#include <limits>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

#include "shoccs_config.hpp"

#define FWD(x) static_cast<decltype(x)&&>(x)
#define MOVE(x) static_cast<std::remove_reference_t<decltype(x)>&&>(x)

namespace ccs
{

template <typename T>
using span = std::span<T>;

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

// NumericTuple: a tuple-like type whose elements are all arithmetic.
// Matches std::array<real, N> (= real3), std::tuple<const real&, ...>
// (from cartesian_product_view), and similar fixed-size numeric containers.
namespace detail
{
template <typename T,
          typename = std::make_index_sequence<
              std::tuple_size_v<std::remove_cvref_t<T>>>>
struct all_elements_arithmetic : std::false_type {
};
template <typename T, std::size_t... Is>
struct all_elements_arithmetic<T, std::index_sequence<Is...>>
    : std::bool_constant<(std::is_arithmetic_v<std::remove_cvref_t<
                               std::tuple_element_t<Is, std::remove_cvref_t<T>>>> &&
                          ...)> {
};
} // namespace detail

template <typename T>
concept NumericTuple =
    requires { typename std::tuple_size<std::remove_cvref_t<T>>::type; } &&
    detail::all_elements_arithmetic<T>::value;

// Range: any input_range except int3 (prevents ambiguous overload with int3 args).
template <typename T>
concept Range =
    std::ranges::input_range<T> && (!std::same_as<int3, std::remove_cvref_t<T>>);

template <int N>
using lit = std::integral_constant<int, N>;

struct system_stats {
    std::vector<real> stats;
    real wall_time_s = 0.0;
};

template <typename T = real>
constexpr auto null_v = std::numeric_limits<T>::max();

// enum for labeling directions
enum class dim { X, Y, Z };

// types for matrix/operator accumulation policy
struct eq_t {
    template <typename X, typename Y>
    constexpr void operator()(X& x, Y&& y) const
    {
        x = FWD(y);
    }
};

struct plus_eq_t {
    template <typename X, typename Y>
    constexpr void operator()(X& x, Y&& y) const
    {
        x += FWD(y);
    }
};

constexpr auto eq = eq_t{};
constexpr auto plus_eq = plus_eq_t{};

struct index_slice {
    integer first;
    integer last;
};

} // namespace ccs

#ifndef NDEBUG
#include <cxxabi.h>
#include <memory>
#include <string>
#include <typeinfo>
namespace debug
{

inline std::string demangle(const char* name)
{
    int status{};
    auto res = std::unique_ptr<char, void (*)(void*)>{
        abi::__cxa_demangle(name, NULL, NULL, &status), std::free};

    return status == 0 ? res.get() : name;
}

template <typename T>
std::string type()
{
    return demangle(typeid(T).name());
}

} // namespace debug
#endif
