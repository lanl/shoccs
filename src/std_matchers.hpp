#pragma once

#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

namespace Catch::Matchers
{
template <typename T, auto N>
struct ArrayMatcher final : MatcherBase<std::array<T, N>> {

    ArrayMatcher(std::array<T, N> const& comparator) : m_comparator(comparator) {}

    bool match(std::array<T, N> const& v) const override
    {
        for (std::size_t i = 0; i < v.size(); ++i)
            if (m_comparator[i] != approx(v[i])) return false;
        return true;
    }
    std::string describe() const override
    {
        return "is approx: " + ::Catch::Detail::stringify(m_comparator);
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    ArrayMatcher& epsilon(T const& newEpsilon)
    {
        approx.epsilon(static_cast<double>(newEpsilon));
        return *this;
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    ArrayMatcher& margin(T const& newMargin)
    {
        approx.margin(static_cast<double>(newMargin));
        return *this;
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    ArrayMatcher& scale(T const& newScale)
    {
        approx.scale(static_cast<double>(newScale));
        return *this;
    }

    std::array<T, N> const& m_comparator;
    mutable Catch::Approx approx = Catch::Approx::custom();
};

template <typename T, auto N>
ArrayMatcher<T, N> Approx(std::array<T, N> const& comparator)
{
    return ArrayMatcher<T, N>(comparator);
}
} // namespace Catch::Matchers