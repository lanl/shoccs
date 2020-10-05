#pragma once

#include <optional>

namespace ccs
{
// Given an interval, this class will be true when the distance between inputs
// is greater than or equal to the initial interval, rollover is taken into account.
// rollover introduces some state which must be updated via the accept method
template <typename T>
class interval
{
    std::optional<T> distance;
    T first;

public:
    interval() : distance{}, first{} {};

    interval(T distance, T first = T{}) : distance{distance}, first{first} {}

    bool operator()(T val, T tolerance = T{}) const
    {
        return !distance ? false : val - first >= *distance - tolerance;
    }

    operator bool() const { return !!distance; }

    interval& operator++()
    {
        first += distance.value_or(T{});
        return *this;
    }

    interval operator++(int)
    {
        interval old{*this};
        ++*this;
        return old;
    }

    interval& operator--()
    {
        first -= distance.value_or(T{});
        return *this;
    }

    interval operator--(int)
    {
        interval old{*this};
        --*this;
        return old;
    }
};
} // namespace ccs
