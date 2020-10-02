#pragma once

#include "indexing.hpp"
#include "types.hpp"

#include "result_field.hpp"

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/view/repeat_n.hpp>

namespace ccs
{

namespace vs = ranges::views;

template <typename T = real, int = 2>
class scalar_field;

template <ranges::random_access_range R, int I>
struct scalar_range;

namespace traits
{
// define some traits and concepts to constrain our universal references and ranges
template <typename = void>
struct is_scalar_field : std::false_type {
};
template <typename T, int I>
struct is_scalar_field<scalar_field<T, I>> : std::true_type {
};

template <typename U>
constexpr bool is_scalar_field_v = is_scalar_field<std::remove_cvref_t<U>>::value;

template <typename = void>
struct is_scalar_range : std::false_type {
};
template <typename T, int I>
struct is_scalar_range<scalar_range<T, I>> : std::true_type {
};

template <typename U>
constexpr bool is_scalar_range_v = is_scalar_range<std::remove_cvref_t<U>>::value;

template <typename U>
constexpr bool is_range_or_field_v = is_scalar_field_v<U> || is_scalar_range_v<U>;

template <typename>
struct scalar_dim_;

template <typename T, int I>
struct scalar_dim_<scalar_range<T, I>> {
    static constexpr int dim = I;
};

template <typename T, int I>
struct scalar_dim_<scalar_field<T, I>> {
    static constexpr int dim = I;
};

template <typename T>
requires is_range_or_field_v<std::remove_cvref_t<T>> constexpr int scalar_dim =
    scalar_dim_<std::remove_cvref_t<T>>::dim;

template <typename T, typename U>
requires is_range_or_field_v<T>&& is_range_or_field_v<U> constexpr bool is_same_dim_v =
    scalar_dim<T> == scalar_dim<U>;

} // namespace traits

template <typename T>
concept Scalar = traits::is_range_or_field_v<T>;

template <typename T, typename U>
concept Compatible_Fields = Scalar<T>&& Scalar<U>&& traits::is_same_dim_v<T, U>;

template <typename T, typename U>
concept Transposable_Fields = Scalar<T>&& Scalar<U> && (!traits::is_same_dim_v<T, U>);

namespace detail
{
template <int I>
struct ext {
    int3 extents_;

    int3 extents() const { return extents_; }

    constexpr int index(const int3& ijk) const
    {
        constexpr int D = I;
        constexpr int S = index::dir<D>::slow;
        constexpr int F = index::dir<D>::fast;
        return ijk[S] * extents_[D] * extents_[F] + ijk[F] * extents_[D] + ijk[D];
    }
};
} // namespace detail

template <ranges::random_access_range R, int I>
struct scalar_range : result_range<R>, detail::ext<I> {
    using result_range<R>::operator();

    decltype(auto) operator()(const int3& ijk) const { return (*this)[this->index(ijk)]; }
    decltype(auto) operator()(const int3& ijk) { return (*this)[this->index(ijk)]; }
};


template <typename T, int I>
class scalar_field : public result_field<std::vector, T>, public detail::ext<I>
{
    using S = scalar_field<T, I>;
    using P = result_field<std::vector, T>;
    using E = detail::ext<I>;

public:
    scalar_field() = default;

    scalar_field(int3 ex) : P{ex[0] * ex[1] * ex[2]}, E{ex} {}

    scalar_field(std::vector<T> f, int3 ex) : P{std::move(f)}, E{ex} {}

    // this will not override default copy/move constructors
    template <typename R>
        requires Compatible_Fields<S, R> &&
        (!std::same_as<S, std::remove_cvref<R>>)scalar_field(R&& r)
        : P(r.size()),
    E(r.extents())
    {
        ranges::copy(r, this->begin());
    }

    template <typename R>
    requires Transposable_Fields<S, R> scalar_field(R&& r) : P(r.size()), E{r.extents()}
    {
        // invoke copy assignment
        *this = r;
    }

    // this will not override default copy/move assigmnent
    template <typename R>
        requires Compatible_Fields<S, R> &&
        (!std::same_as<S, std::remove_cvref<R>>)scalar_field& operator=(R&& r)
    {
        this->extents_ = r.extents();
        this->resize(r.size());
        ranges::copy(r, this->begin());
        return *this;
    }

#define gen_operators(op, acc)                                                           \
    template <typename R>                                                                \
    requires Transposable_Fields<S, R> scalar_field& op(R&& r)                           \
    {                                                                                    \
        this->extents_ = r.extents();                                                    \
        this->resize(r.size());                                                          \
        const auto& ex = this->extents();                                                \
                                                                                         \
        constexpr int AD = I;                                                            \
        constexpr int AS = index::dir<AD>::slow;                                         \
        constexpr int AF = index::dir<AD>::fast;                                         \
                                                                                         \
        constexpr int BD = traits::scalar_dim<R>;                                        \
        constexpr int BS = index::dir<BD>::slow;                                         \
        constexpr int BF = index::dir<BD>::fast;                                         \
                                                                                         \
        auto [nad, naf, nas] = int3{ex[AD], ex[AF], ex[AS]};                             \
        auto [nbd, nbf, nbs] = int3{ex[BD], ex[BF], ex[BS]};                             \
                                                                                         \
        for (int as = 0; as < nas; as++)                                                 \
            for (int af = 0; af < naf; af++)                                             \
                for (int ad = 0; ad < nad; ad++) {                                       \
                    auto [bd, bf, bs] = index::transpose<AD, BD>(int3{ad, af, as});      \
                    (*this)[as * naf * nad + af * nad + ad] acc                          \
                        r[bs * nbf * nbd + bf * nbd + bd];                               \
                }                                                                        \
        return *this;                                                                    \
    }                                                                                    \
    template <Numeric N>                                                                 \
    scalar_field& op(N n)                                                                \
    {                                                                                    \
        for (auto&& v : (*this)) v acc n;                                                \
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

        // bring base class call operator into scope so we don't shadow them
        using P::operator();

    const T& operator()(const int3& ijk) const { return (*this)[this->index(ijk)]; }
    T& operator()(const int3& ijk) { return (*this)[this->index(ijk)]; }


    operator std::span<const T>() const { return {&(*this)[0], this->size()}; }
    operator std::span<T>() { return {&(*this)[0], this->size()}; }

    // operator result_view_t<T>() { return {f}; }
}; // namespace ccs

#define ret_range(expr)                                                                  \
    return scalar_range<decltype(expr), I> { expr, t.extents() }

#define gen_operators(op, f)                                                             \
    template <typename T, typename U>                                                    \
    requires Compatible_Fields<T, U> constexpr auto op(T&& t, U&& u)                     \
    {                                                                                    \
        assert(t.extents() == u.extents());                                              \
        constexpr auto I = traits::scalar_dim<T>;                                        \
        ret_range(vs::zip_with(f, std::forward<T>(t), std::forward<U>(u)));              \
    }                                                                                    \
    template <Scalar T, Numeric U>                                                       \
    constexpr auto op(T&& t, U u)                                                        \
    {                                                                                    \
        constexpr auto I = traits::scalar_dim<T>;                                        \
        ret_range(vs::zip_with(f, std::forward<T>(t), vs::repeat_n(u, t.size())));       \
    }                                                                                    \
    template <Scalar T, Numeric U>                                                       \
    constexpr auto op(U u, T&& t)                                                        \
    {                                                                                    \
        constexpr auto I = traits::scalar_dim<T>;                                        \
        ret_range(vs::zip_with(f, vs::repeat_n(u, t.size()), std::forward<T>(t)));       \
    }

// clang-format off
gen_operators(operator+, std::plus{})
gen_operators(operator-, std::minus{})
gen_operators(operator*, std::multiplies{})
gen_operators(operator/, std::divides{})
#undef gen_operators
    // clang-format on

    template <Scalar T, typename ViewFn>
    requires rs::invocable_view_closure<ViewFn, T> constexpr auto
    operator>>(T&& t, vs::view_closure<ViewFn> f)
{
    constexpr auto I = traits::scalar_dim<T>;
    ret_range(std::forward<T>(t) | f);
}
#undef ret_range

using x_field = scalar_field<real, 0>;
using y_field = scalar_field<real, 1>;
using z_field = scalar_field<real, 2>;

} // namespace ccs