#pragma once

#include "temporal/step_controller.hpp"
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

class d_interval
{
    interval<int> step_interval;
    interval<real> time_interval;
    int ndumps;
    bool step_ready;
    bool time_ready;

public:
    d_interval() = default;
    d_interval(interval<int> step_interval, interval<real> time_interval)
        : step_interval{MOVE(step_interval)}, time_interval{MOVE(time_interval)}, ndumps{}
    {
    }

    bool operator()(const step_controller& step, real dt)
    {
        if (!(step_interval || time_interval)) return false;

        step_ready = step_interval(step);
        time_ready = time_interval(step, dt);

        return step_ready || time_ready || (int)step == 0;
    }

    int current_dump() const { return ndumps; }

    d_interval& operator++()
    {
        if (step_ready) ++step_interval;
        if (time_ready) ++time_interval;
        return *this;
    }

    d_interval operator++(int)
    {
        d_interval old{*this};
        ++*this;
        return old;
    }
};
} // namespace ccs
