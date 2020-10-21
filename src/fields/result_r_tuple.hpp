#pragma once

#include "r_tuple.hpp"
#include "indexing.hpp"

namespace ccs
{
// test out our crtp
template <typename U, typename V>
concept Compatible = requires(U u, V v)
{
    // check validity of this member
    typename U::compatible_types;
    // check membership in typelist
};

template<typename T>
struct lazy_math_crtp;

class lazy_math_access
{
    template <typename T>
    friend class lazy_math_crtp;

public:
    template <typename T, std::same_as<T> U>
    static constexpr decltype(auto) as_range(T t)
    {
        return t.as_range();
    }

    template<Numeric N, typename T>
    static constexpr decltype(auto) as_range(N n)
    {
        return n;
    }

    template<Numeric N, typename T>
    static constexpr decltype(auto) as_range(T t)
    {
        return t.as_range();
    }

    template<typename T, Numeric N>
    static constexpr decltype(auto) as_range(T t)
    {
        return t.as_range();
    }

    template<typename T, Numeric N>
    static constexpr decltype(auto) as_range(N n)
    {
        return n;
    }
    
};

template <typename T>
struct lazy_math_crtp {

private:

    template <typename U, typename V>
    //requires std::same_as<T, std::remove_cvref_t<U>> // && Compatible<U, V>
        friend constexpr auto operator+(U&& u, V&& v)
    {
        return T{lazy_math_access::as_range<U, V>(FWD(u)) +
                 lazy_math_access::as_range<U, V>(FWD(v))};
    }
};


class q : public lazy_math_crtp<q>
{
    float a;

    friend class lazy_math_access;

    constexpr float as_range() const { return a; }

public:
    q() = default;
    q(float a) : a{a} {};

    float value() const { return a; }
};

template<typename R, int I>
class directional_field : public r_tuple<R&>, public index::bounds<I> {};

template <All R, int I>
class directional_field<R, I> : public r_tuple<R>, public index::bounds<I> {
    
    public:
    directional_field() = default;
};


template<typename T, int I>
using owning_field = directional_field<std::vector<T>, I>;

using x_field = owning_field<real, 0>;
using y_field = owning_field<real, 1>;
using z_field = owning_field<real, 2>;

} // namespace ccs