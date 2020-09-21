#pragma once

#include "scalar_field.hpp"

namespace ccs
{

template <typename T = real>
class vector_field;

template <ranges::random_access_range X,
          ranges::random_access_range Y,
          ranges::random_access_range Z>
struct vector_range;

namespace detail
{
// define some traits and concepts to constrain our universal references and ranges
template <typename = void>
struct is_vector_field : std::false_type {
};
template <typename T>
struct is_vector_field<vector_field<T>> : std::true_type {
};

template <typename U>
constexpr bool is_vector_field_v = is_vector_field<std::remove_cvref_t<U>>::value;

template <typename = void>
struct is_vector_range : std::false_type {
};
template <typename X, typename Y, typename Z>
struct is_vector_range<vector_range<X, Y, Z>> : std::true_type {
};

template <typename U>
constexpr bool is_vector_range_v = is_vector_range<std::remove_cvref_t<U>>::value;

template <typename U>
constexpr bool is_vrange_or_vfield_v = is_vector_field_v<U> || is_vector_range_v<U>;

} // namespace detail

template <typename T>
concept Vector_Field = detail::is_vrange_or_vfield_v<T>;

template <ranges::random_access_range X,
          ranges::random_access_range Y,
          ranges::random_access_range Z>
struct vector_range {
    X x_;
    Y y_;
    Z z_;

#if 0
    const R& range() const& { return r; }
    R& range() & { return r; }
    R range() && { return std::move(r); }

    int3 extents() { return extents_; }

    // iterator interface
    size_t size() const noexcept { return r.size(); }

    auto begin() const { return r.begin(); }
    auto begin() { return r.begin(); }
    auto end() const { return r.end(); }
    auto end() { return r.end(); }

    decltype(auto) operator[](int i) & { return r[i]; }
    auto operator[](int i) && { return r[i]; }
    decltype(auto) operator[](int i) const& { return r[i]; }
#endif
};

struct vector_field_index {
    std::span<const int> x;
    std::span<const int> y;
    std::span<const int> z;
};

struct vector_field_bvalues
{
    std::span<const real> x;
    std::span<const real> y;
    std::span<const real> z;
};

namespace detail
{
template <typename T>
class vector_field_select
{
    const vector_field_index& indices;
    vector_field<T>& f;

public:
    vector_field_select(const vector_field_index& indices,
                        vector_field<T>& f)
        : indices{indices}, f{f}
    {
    }

    void operator=(const vector_field_bvalues& values)
    {
        f.x()(indices.x) = values.x;
        f.y()(indices.y) = values.y;
        f.z()(indices.z) = values.z;
    }
};
} // namespace detail
template <typename T>
class vector_field
{
    using X = scalar_field<T, 0>;
    using Y = scalar_field<T, 1>;
    using Z = scalar_field<T, 2>;

    X x_;
    Y y_;
    Z z_;

public:
    vector_field() = default;

    vector_field(int n, int3 ex) : x_{n, ex}, y_{n, ex}, z_{n, ex} {}

    vector_field(X&& x, Y&& y, Z&& z)
        : x_{std::move(x)}, y_{std::move(y)}, z_{std::move(z)}
    {
    }

    template <int I>
    vector_field(scalar_field<T, I>&& s) : x_{s}, y_{s}, z_{std::move(s)}
    {
    }

    X& x() { return x_; }
    const X& x() const { return x_; }
    Y& y() { return y_; }
    const Y& y() const { return y_; }
    Z& z() { return z_; }
    const Z& z() const { return z_; }

#define gen_operators(op, acc)                                                           \
    template <Field R>                                                                   \
    vector_field& op(R&& r)                                                              \
    {                                                                                    \
        x_ acc r;                                                                        \
        y_ acc r;                                                                        \
        z_ acc r;                                                                        \
                                                                                         \
        return *this;                                                                    \
    }                                                                                    \
    template <Numeric N>                                                                 \
    vector_field& op(N n)                                                                \
    {                                                                                    \
        x_ acc n;                                                                        \
        y_ acc n;                                                                        \
        z_ acc n;                                                                        \
        return *this;                                                                    \
    }

    // clang-format off

gen_operators(operator=, =)
gen_operators(operator+=, +=)
gen_operators(operator-=, -=)
gen_operators(operator*=, *=)
gen_operators(operator/=, /=)
#undef gen_operators

        // clang-format on

        int3 extents()
    {
        return x_.extents_;
    }

    auto operator()(const vector_field_index& i) &
    {
        return detail::vector_field_select<T>{i, *this};
    }
};

} // namespace ccs