#pragma once

#include "boundaries.hpp"
#include "types.hpp"
#include <concepts>

namespace ccs
{

// A stencil is comprised of both an interior discretization and its
// associated numerical boundary scheme.
// The interior discretization is of order 2p and containts 2p + 1 points
// The nbs is a dense small dense rxt matrix
// A stencil can have extra information associated with it (like Neumann BC's)
struct stencil_info {
    int p;
    int r;
    int t;
    int nextra;
};

// satisfaction of Stencil makes a type a stencil
template <typename T>
concept Stencil = requires(const T& stencil,
                           boundary b,
                           real h,
                           real psi,
                           bool ray_outside,
                           std::span<real> c,
                           std::span<real> extra)
{
    {
        stencil.query(b)
    }
    ->std::same_as<stencil_info>;

    { stencil.query_max() } ->std::same_as<stencil_info>;

    {stencil.nbs(h, b, psi, ray_outside, c, extra)};

    {stencil.interior(h, c)};
};

class stencil
{
    // private nested classes for type erasure
    class any_stencil
    {
    public:
        virtual ~any_stencil() {}
        virtual any_stencil* clone() const = 0;
        virtual stencil_info query(boundary) const = 0;
        virtual stencil_info query_max() const = 0;
        virtual void nbs(real h,
                         boundary,
                         real psi,
                         bool ray_outside,
                         std::span<real> coeffs,
                         std::span<real> extra) const = 0;
        virtual void interior(real c, std::span<real> coeffs) const = 0;
    };

    template <Stencil S>
    class any_stencil_impl : public any_stencil
    {
        S s;

    public:
        any_stencil_impl(const S& s) : s{s} {}
        any_stencil_impl(S&& s) : s{std::move(s)} {}

        any_stencil_impl* clone() const override { return new any_stencil_impl(s); }

        stencil_info query(boundary b) const override { return s.query(b); }
        stencil_info query_max() const override { return s.query_max(); }

        void nbs(real h,
                 boundary b,
                 real psi,
                 bool ray_outside,
                 std::span<real> c,
                 std::span<real> extra) const override
        {
            return s.nbs(h, b, psi, ray_outside, c, extra);
        }

        void interior(real h, std::span<real> c) const override
        {
            return s.interior(h, c);
        }
    };

    any_stencil* s;

public:
    stencil() : s{nullptr} {}

    stencil(const stencil& other) : s{nullptr}
    {
        if (other) s = other.s->clone();
    }

    stencil(stencil&& other) : s{std::exchange(other.s, nullptr)} {}

    // construction from anything with a hit method
    template <typename T>
        requires Stencil<T> &&
        (!std::same_as<stencil, std::remove_cvref_t<T>>)stencil(T&& other)
        : s{new any_stencil_impl{std::forward<T>(other)}}
    {
    }

    stencil& operator=(const stencil& other)
    {
        stencil tmp{other};
        swap(*this, tmp);
        return (*this);
    }

    stencil& operator=(stencil&& other)
    {
        delete s;
        s = std::exchange(other.s, nullptr);
        return *this;
    }

    ~stencil() { delete s; }

    friend void swap(stencil& x, stencil& y) { std::swap(x.s, y.s); }

    explicit operator bool() const { return s != nullptr; }

    stencil_info query(boundary b) const { return s->query(b); }
    stencil_info query_max() const { return s->query_max(); }

    void nbs(real h,
             boundary b,
             real psi,
             bool ray_outside,
             std::span<real> c,
             std::span<real> ex) const
    {
        return s->nbs(h, b, psi, ray_outside, c, ex);
    }

    void interior(real h, std::span<real> c) const { return s->interior(h, c); }
};

stencil make_E2_2();
} // namespace ccs
