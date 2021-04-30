#pragma once

#include "types.hpp"
#include <array>
#include <cassert>
#include <concepts>

namespace ccs
{

// clang-format off
template <typename M>
concept MSolution = requires(const M& ms, real time, const real3& loc, int dim) {
    { ms(time, loc) } -> std::same_as<real>;
    { ms.ddt(time, loc) } -> std::same_as<real>;
    { ms.gradient(time, loc) } -> std::same_as<real3>;
    { ms.divergence(time, loc) } -> std::same_as<real>;
    { ms.laplacian(time, loc) } -> std::same_as<real>;
};
// clang-format on

class ManufacturedSolution
{
    class any_sol
    {
    public:
        // soution evaulation
        virtual real operator()(real time, const real3& loc) const = 0;

        // time derivative of solution
        virtual real ddt(real time, const real3& loc) const = 0;

        // gradient of solution
        virtual real3 gradient(real time, const real3& loc) const = 0;

        // divergence of solution
        virtual real divergence(real time, const real3& loc) const = 0;

        // laplacian of solution
        virtual real laplacian(real time, const real3& loc) const = 0;

        virtual ~any_sol() = default;

        virtual any_sol* clone() const = 0;
    };

    template <MSolution M>
    class any_sol_impl : public any_sol
    {
        M m;

    public:
        any_sol_impl(const M& m) : m{m} {}
        any_sol_impl(M&& m) : m{std::move(m)} {}

        any_sol_impl* clone() const override { return new any_sol_impl(m); }

        real operator()(real time, const real3& loc) const override
        {
            return m(time, loc);
        }

        real ddt(real time, const real3& loc) const override { return m.ddt(time, loc); }

        real3 gradient(real time, const real3& loc) const override
        {
            return m.gradient(time, loc);
        }

        real divergence(real time, const real3& loc) const override
        {
            return m.divergence(time, loc);
        }

        real laplacian(real time, const real3& loc) const override
        {
            return m.laplacian(time, loc);
        }
    };

    any_sol* s;

public:
    ManufacturedSolution() : s{nullptr} {}

    ManufacturedSolution(const ManufacturedSolution& other) : s{nullptr}
    {
        if (other) s = other.s->clone();
    }

    ManufacturedSolution(ManufacturedSolution&& other)
        : s{std::exchange(other.s, nullptr)}
    {
    }

    // construc // construction from anything with a hit method
    template <typename T>
        requires MSolution<T> &&
        (!std::same_as<ManufacturedSolution, std::remove_cvref_t<T>>)
            ManufacturedSolution(T&& other)
        : s{new any_sol_impl{std::forward<T>(other)}}
    {
    }

    ManufacturedSolution& operator=(const ManufacturedSolution& other)
    {
        ManufacturedSolution tmp{other};
        swap(*this, tmp);
        return (*this);
    }

    ManufacturedSolution& operator=(ManufacturedSolution&& other)
    {
        delete s;
        s = std::exchange(other.s, nullptr);
        return *this;
    }

    ~ManufacturedSolution() { delete s; }

    friend void swap(ManufacturedSolution& x, ManufacturedSolution& y)
    {
        std::swap(x.s, y.s);
    }

    explicit operator bool() const { return s != nullptr; }

    real operator()(real time, const real3& loc) const
    {
        assert(s);
        return (*s)(time, loc);
    }

    real ddt(real time, const real3& loc) const
    {
        assert(s);
        return s->ddt(time, loc);
    }

    real3 gradient(real time, const real3& loc) const
    {
        assert(s);
        return s->gradient(time, loc);
    }

    real divergence(real time, const real3& loc) const
    {
        assert(s);
        return s->divergence(time, loc);
    }

    real laplacian(real time, const real3& loc) const
    {
        assert(s);
        return s->laplacian(time, loc);
    }
};
// factories
ManufacturedSolution build_ms_gauss1d(std::span<const real3> center,
                                       std::span<const real3> variance,
                                       std::span<const real> amplitude,
                                       std::span<const real> frequency);

ManufacturedSolution build_ms_gauss2d(std::span<const real3> center,
                                       std::span<const real3> variance,
                                       std::span<const real> amplitude,
                                       std::span<const real> frequency);

ManufacturedSolution build_ms_gauss3d(std::span<const real3> center,
                                       std::span<const real3> variance,
                                       std::span<const real> amplitude,
                                       std::span<const real> frequency);

ManufacturedSolution inline build_ms_gauss(int dims,
                                            std::span<const real3> center,
                                            std::span<const real3> variance,
                                            std::span<const real> amplitude,
                                            std::span<const real> frequency)
{
    switch (dims) {
    case 1:
        return build_ms_gauss1d(center, variance, amplitude, frequency);
    case 2:
        return build_ms_gauss2d(center, variance, amplitude, frequency);
    case 3:
        return build_ms_gauss3d(center, variance, amplitude, frequency);
    }
    return {};
}

} // namespace ccs
