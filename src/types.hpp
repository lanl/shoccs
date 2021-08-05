#pragma once

#include <concepts>
#include <limits>
#include <span>
#include <tuple>
#include <vector>

#include "shoccs_config.hpp"

#define FWD(x) static_cast<decltype(x)&&>(x)
#define MOVE(x) static_cast<std::remove_reference_t<decltype(x)>&&>(x)

// do this for easy reference
namespace ranges::views
{
}

namespace ccs
{

template <typename T>
using span = std::span<T>;

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

namespace rs = ranges;
namespace vs = ranges::views;

template <int N>
using lit = std::integral_constant<int, N>;

struct system_stats {
    std::vector<real> stats;
};

template <typename T = real>
constexpr auto null_v = std::numeric_limits<T>::max();

// enum for labeling directions
enum class dim { X, Y, Z };

// types for matrix/operator accumulation policy
struct eq_t {
    template <typename X, typename Y>
    constexpr void operator()(X& x, Y&& y)
    {
        x = FWD(y);
    }
};

struct plus_eq_t {
    template <typename X, typename Y>
    constexpr void operator()(X& x, Y&& y)
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
