// matchers to be used for catch tests
#pragma once

#include "Tuple.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

namespace Catch::Matchers
{
template <typename T, int I>
struct ScalarFieldMatcher final : MatcherBase<ccs::scalar_field<T, I>> {

    ScalarFieldMatcher(ccs::scalar_field<T, I> const& comparator)
        : m_comparator(comparator)
    {
    }

    bool match(ccs::scalar_field<T, I> const& v) const override
    {
        if (m_comparator.size() != v.size()) return false;
        for (std::size_t i = 0; i < v.size(); ++i)
            if (m_comparator[i] != approx(v[i])) return false;
        return true;
    }
    std::string describe() const override
    {
        return "is approx: " + ::Catch::Detail::stringify(m_comparator);
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    ScalarFieldMatcher& epsilon(T const& newEpsilon)
    {
        approx.epsilon(static_cast<double>(newEpsilon));
        return *this;
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    ScalarFieldMatcher& margin(T const& newMargin)
    {
        approx.margin(static_cast<double>(newMargin));
        return *this;
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    ScalarFieldMatcher& scale(T const& newScale)
    {
        approx.scale(static_cast<double>(newScale));
        return *this;
    }

    ccs::scalar_field<T, I> const& m_comparator;
    mutable Catch::Approx approx = Catch::Approx::custom();
};

template <typename T>
struct VectorFieldMatcher final : MatcherBase<ccs::vector_field<T>> {

    VectorFieldMatcher(ccs::vector_field<T> const& comparator) : m_comparator(comparator)
    {
    }

    bool match(ccs::vector_field<T> const& v) const override
    {
        return ScalarFieldMatcher(m_comparator.x).match(v.x) &&
               ScalarFieldMatcher(m_comparator.y).match(v.y) &&
               ScalarFieldMatcher(m_comparator.z).match(v.z);
    }
    std::string describe() const override
    {
        return "is approx: " + ::Catch::Detail::stringify(m_comparator);
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    VectorFieldMatcher& epsilon(T const& newEpsilon)
    {
        approx.epsilon(static_cast<double>(newEpsilon));
        return *this;
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    VectorFieldMatcher& margin(T const& newMargin)
    {
        approx.margin(static_cast<double>(newMargin));
        return *this;
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    VectorFieldMatcher& scale(T const& newScale)
    {
        approx.scale(static_cast<double>(newScale));
        return *this;
    }

    ccs::vector_field<T> const& m_comparator;
    mutable Catch::Approx approx = Catch::Approx::custom();
};
template <typename T>
struct VectorRangeMatcher final : MatcherBase<ccs::vector_range<std::vector<T>>> {

    VectorRangeMatcher(ccs::vector_range<std::vector<T>> const& comparator)
        : m_comparator(comparator)
    {
    }

    bool match(ccs::vector_range<std::vector<T>> const& v) const override
    {
        return Approx(m_comparator.x).match(v.x) && Approx(m_comparator.y).match(v.y) &&
               Approx(m_comparator.z).match(v.z);
    }
    std::string describe() const override
    {
        return "is approx: " + ::Catch::Detail::stringify(m_comparator);
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    VectorRangeMatcher& epsilon(T const& newEpsilon)
    {
        approx.epsilon(static_cast<double>(newEpsilon));
        return *this;
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    VectorRangeMatcher& margin(T const& newMargin)
    {
        approx.margin(static_cast<double>(newMargin));
        return *this;
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    VectorRangeMatcher& scale(T const& newScale)
    {
        approx.scale(static_cast<double>(newScale));
        return *this;
    }

    ccs::vector_range<std::vector<T>> const& m_comparator;
    mutable Catch::Approx approx = Catch::Approx::custom();
};

template <typename T, int I>
struct ScalarMatcher final : MatcherBase<ccs::scalar<T, I>> {

    ScalarMatcher(ccs::scalar<T, I> const& comparator) : m_comparator(comparator) {}

    bool match(ccs::scalar<T, I> const& v) const override
    {
        return ScalarFieldMatcher(m_comparator.field).match(v.field) &&
               VectorRangeMatcher(m_comparator.obj).match(v.obj);
    }
    std::string describe() const override
    {
        return "is approx: " + ::Catch::Detail::stringify(m_comparator);
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    ScalarMatcher& epsilon(T const& newEpsilon)
    {
        approx.epsilon(static_cast<double>(newEpsilon));
        return *this;
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    ScalarMatcher& margin(T const& newMargin)
    {
        approx.margin(static_cast<double>(newMargin));
        return *this;
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    ScalarMatcher& scale(T const& newScale)
    {
        approx.scale(static_cast<double>(newScale));
        return *this;
    }

    ccs::scalar<T, I> const& m_comparator;
    mutable Catch::Approx approx = Catch::Approx::custom();
};

template <typename T>
struct VectorMatcher final : MatcherBase<ccs::vector<T>> {

    VectorMatcher(ccs::vector<T> const& comparator) : m_comparator(comparator) {}

    bool match(ccs::vector<T> const& v) const override
    {
        return VectorFieldMatcher(m_comparator.field).match(v.field) &&
               VectorRangeMatcher(m_comparator.obj).match(v.obj);
    }
    std::string describe() const override
    {
        return "is approx: " + ::Catch::Detail::stringify(m_comparator);
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    VectorMatcher& epsilon(T const& newEpsilon)
    {
        approx.epsilon(static_cast<double>(newEpsilon));
        return *this;
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    VectorMatcher& margin(T const& newMargin)
    {
        approx.margin(static_cast<double>(newMargin));
        return *this;
    }
    template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
    VectorMatcher& scale(T const& newScale)
    {
        approx.scale(static_cast<double>(newScale));
        return *this;
    }

    ccs::vector<T> const& m_comparator;
    mutable Catch::Approx approx = Catch::Approx::custom();
};

//! Creates a matcher that matches vectors that `comparator` as an element
template <typename T, int I>
ScalarFieldMatcher<T, I> Approx(ccs::scalar_field<T, I> const& comparator)
{
    return ScalarFieldMatcher<T, I>(comparator);
}

template <typename T, int I>
ScalarMatcher<T, I> Approx(ccs::scalar<T, I> const& comparator)
{
    return ScalarMatcher<T, I>(comparator);
}

//! Creates a matcher that matches vectors that `comparator` as an element
template <typename T>
VectorFieldMatcher<T> Approx(ccs::vector_field<T> const& comparator)
{
    return VectorFieldMatcher<T>(comparator);
}

template <typename T>
VectorRangeMatcher<T> Approx(ccs::vector_range<std::vector<T>> const& comparator)
{
    return VectorRangeMatcher<T>(comparator);
}

template <typename T>
VectorMatcher<T> Approx(ccs::vector<T> const& comparator)
{
    return VectorMatcher<T>(comparator);
}

} // namespace Catch::Matchers