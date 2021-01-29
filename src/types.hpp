#pragma once

#include <array>
#include <concepts>
#include <span>
#include <limits>

#define FWD(x) static_cast<decltype(x)>(x)
#define MOVE(x) static_cast<std::remove_reference_t<decltype(x)>&&>(x)

// do this for easy reference
namespace ranges::views
{
}

namespace ccs
{
using real = double;
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

template<int N>
using lit = std::integral_constant<int, N>;

struct SystemStats {};

template<typename T = real>
constexpr auto null_v = std::numeric_limits<T>::max();

} // namespace ccs


#ifndef NDEBUG
#include <string>
#include <typeinfo>
#include <memory>
#include <cxxabi.h>
namespace debug {


inline std::string demangle(const char* name)
{
    int status {};
    auto res = std::unique_ptr<char>{abi::__cxa_demangle(name, NULL, NULL, &status)};

    return status == 0 ? res.get() : name;
}

template<typename T>
std::string type(const T& t)
{
    return demangle(typeid(t).name());
}

}
#endif
