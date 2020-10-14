#pragma once

#include "index_view.hpp"
#include "types.hpp"

#include "contract.hpp"
#include "result_field.hpp"
#include "scalar_field_fwd.hpp"

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/copy_n.hpp>
#include <range/v3/view/repeat_n.hpp>

namespace ccs
{

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

    scalar_field(std::vector<T> f, int3 ex) : P{MOVE(f)}, E{ex} {}
    template<rs::input_range R> scalar_field(R&& r, int3 ex) : P{ex[0] * ex[1] * ex[2]}, E{ex}
    {
        rs::copy_n(rs::begin(r), this->size(), this->begin());
    }

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

    template <Contraction C>
    scalar_field(C&& c) : P(c.size()), E{c.extents()}
    {
        for (auto&& ijk : index_view<I>(c.extents())) (*this)(ijk) = c(ijk);
    }

    template <Contraction C>
    scalar_field& operator=(C&& c)
    {
        this->extents_ = c.extents();
        this->resize(c.size());
        for (auto&& ijk : index_view<I>(c.extents())) (*this)(ijk) = c(ijk);
        return *this;
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

    template <typename R>
        requires rs::input_range<R> && (!Scalar_Field<R>)scalar_field& operator=(R&& r)
    {
        rs::copy_n(rs::begin(r), this->size(), this->begin());
        return *this;
    }

#define SHOCCS_GEN_OPERATORS_(op, acc)                                                   \
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

    SHOCCS_GEN_OPERATORS_(operator=, =)

#define SHOCCS_GEN_OPERATORS(op, acc)                                                    \
    SHOCCS_GEN_OPERATORS_(op, acc)                                                       \
                                                                                         \
    template <typename R>                                                                \
    requires Compatible_Fields<S, R> scalar_field& op(R&& r)                             \
    {                                                                                    \
        int sz = this->size();                                                           \
        for (int i = 0; i < sz; i++) (*this)[i] acc r[i];                                \
        return *this;                                                                    \
    }

    SHOCCS_GEN_OPERATORS(operator+=, +=)
    SHOCCS_GEN_OPERATORS(operator-=, -=)
    SHOCCS_GEN_OPERATORS(operator*=, *=)
    SHOCCS_GEN_OPERATORS(operator/=, /=)
#undef SHOCCS_GEN_OPERATORS
#undef SHOCCS_GEN_OPERATORS_

    // bring base class call operator into scope so we don't shadow them
    using P::operator();

    const T& operator()(const int3& ijk) const { return (*this)[this->index(ijk)]; }
    T& operator()(const int3& ijk) { return (*this)[this->index(ijk)]; }

    operator std::span<const T>() const { return {&(*this)[0], this->size()}; }
    operator std::span<T>() { return {&(*this)[0], this->size()}; }

    // operator result_view_t<T>() { return {f}; }
};

#define ret_range(expr)                                                                  \
    return scalar_range<decltype(expr), I> { expr, t.extents() }

#define SHOCCS_GEN_OPERATORS(op, f)                                                      \
    template <typename T, typename U>                                                    \
    requires Compatible_Fields<T, U> constexpr auto op(T&& t, U&& u)                     \
    {                                                                                    \
        assert(t.extents() == u.extents());                                              \
        constexpr auto I = traits::scalar_dim<T>;                                        \
        ret_range(vs::zip_with(f, FWD(t), FWD(u)));                                      \
    }                                                                                    \
    template <Scalar_Field T, Numeric U>                                                 \
    constexpr auto op(T&& t, U u)                                                        \
    {                                                                                    \
        constexpr auto I = traits::scalar_dim<T>;                                        \
        ret_range(vs::zip_with(f, FWD(t), vs::repeat_n(u, t.size())));                   \
    }                                                                                    \
    template <Scalar_Field T, Numeric U>                                                 \
    constexpr auto op(U u, T&& t)                                                        \
    {                                                                                    \
        constexpr auto I = traits::scalar_dim<T>;                                        \
        ret_range(vs::zip_with(f, vs::repeat_n(u, t.size()), FWD(t)));                   \
    }

SHOCCS_GEN_OPERATORS(operator+, std::plus{})
SHOCCS_GEN_OPERATORS(operator-, std::minus{})
SHOCCS_GEN_OPERATORS(operator*, std::multiplies{})
SHOCCS_GEN_OPERATORS(operator/, std::divides{})
#undef SHOCCS_GEN_OPERATORS

template <Scalar_Field T, typename ViewFn>
requires rs::invocable_view_closure<ViewFn, T> constexpr auto
operator>>(T&& t, vs::view_closure<ViewFn> f)
{
    constexpr auto I = traits::scalar_dim<T>;
    ret_range(FWD(t) | f);
}
#undef ret_range

using x_field = scalar_field<real, 0>;
using y_field = scalar_field<real, 1>;
using z_field = scalar_field<real, 2>;

} // namespace ccs