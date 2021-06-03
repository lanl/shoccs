#include "types.hpp"

namespace ccs
{

template <Numeric T>
struct bounded {
private:
    T t;
    T t_min;
    T t_max;

public:
    bounded() = default;
    bounded(T t_max) : t{}, t_min{}, t_max{t_max} {}
    bounded(T t, T t_min, T t_max) : t{t}, t_min{t_min}, t_max{t_max} {}

    operator bool() const { return (t_min <= t) && (t < t_max); }
    operator T() const { return t; }

    bounded& operator+=(T dt)
    {
        t += dt;
        return *this;
    }

    bounded& operator++()
    {
        ++t;
        return *this;
    }

    bounded operator++(int)
    {
        bounded prev{*this};
        ++*this;
        return prev;
    }

    bounded& operator--()
    {
        --t;
        return *this;
    }

    bounded operator--(int)
    {
        bounded prev{*this};
        --*this;
        return prev;
    }
};

} // namespace ccs
