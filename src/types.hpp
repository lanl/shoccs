#pragma once

#include <array>
#include <concepts>
#include <limits>
#include <span>
#include <tuple>

#define FWD(x) static_cast<decltype(x)&&>(x)
#define MOVE(x) static_cast<std::remove_reference_t<decltype(x)>&&>(x)

// do this for easy reference
namespace ranges::views
{
}

namespace ccs
{
using real = double;
using integer = long; // prefer higher precision than regular int
using real3 = std::array<real, 3>;
using real2 = std::array<real, 2>;
using int3 = std::array<int, 3>;
using int2 = std::array<int, 2>;

template <typename T>
using span = std::span<T>;

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

namespace rs = ranges;
namespace vs = ranges::views;

template <int N>
using lit = std::integral_constant<int, N>;

struct SystemStats {
};

template <typename T = real>
constexpr auto null_v = std::numeric_limits<T>::max();

// enum for labeling directions
enum class dim { X, Y, Z };

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
