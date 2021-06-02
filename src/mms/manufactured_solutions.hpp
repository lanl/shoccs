#pragma once

#include "fields/tuple_utils.hpp"
#include <array>
#include <cassert>
#include <concepts>
#include <optional>
#include <range/v3/view/transform.hpp>

#include <sol/forward.hpp>

namespace ccs
{

// clang-format off
template <typename M>
concept ManufacturedSolution = requires(const M& ms, real time, const real3& loc, int dim) {
    { ms(time, loc) } -> std::same_as<real>;
    { ms.ddt(time, loc) } -> std::same_as<real>;
    { ms.gradient(time, loc) } -> std::same_as<real3>;
    { ms.divergence(time, loc) } -> std::same_as<real>;
    { ms.laplacian(time, loc) } -> std::same_as<real>;
};
// clang-format on

class manufactured_solution
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

    template <ManufacturedSolution M>
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
    manufactured_solution() : s{nullptr} {}

    manufactured_solution(const manufactured_solution& other) : s{nullptr}
    {
        if (other) s = other.s->clone();
    }

    manufactured_solution(manufactured_solution&& other)
        : s{std::exchange(other.s, nullptr)}
    {
    }

    // construc // construction from anything with a hit method
    template <typename T>
        requires ManufacturedSolution<T> &&
            (!std::same_as<manufactured_solution, std::remove_cvref_t<T>>)
                manufactured_solution(T&& other)
            : s{new any_sol_impl{std::forward<T>(other)}}
        {
        }

        manufactured_solution& operator=(const manufactured_solution& other)
        {
            manufactured_solution tmp{other};
            swap(*this, tmp);
            return (*this);
        }

        manufactured_solution& operator=(manufactured_solution&& other)
        {
            delete s;
            s = std::exchange(other.s, nullptr);
            return *this;
        }

        ~manufactured_solution() { delete s; }

        friend void swap(manufactured_solution& x, manufactured_solution& y)
        {
            std::swap(x.s, y.s);
        }

        explicit operator bool() const { return s != nullptr; }

        static std::optional<manufactured_solution> from_lua(const sol::table&,
                                                             int dims = 3);

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

        template <TupleLike L>
        requires ArrayFromTuple<real3, L> real operator()(real time, L&& loc) const
        {
            assert(s);
            return (*s)(time, to<real3>(FWD(loc)));
        }

        template <TupleLike L>
            requires ArrayFromTuple<real3, L> real ddt(real time, L&& loc)
        const
        {
            assert(s);
            return s->ddt(time, to<real3>(FWD(loc)));
        }

        template <TupleLike L>
            requires ArrayFromTuple<real3, L> real3 gradient(real time, L&& loc)
        const
        {
            assert(s);
            return s->gradient(time, to<real3>(FWD(loc)));
        }

        template <TupleLike L>
            requires ArrayFromTuple<real3, L> real divergence(real time, L&& loc)
        const
        {
            assert(s);
            return s->divergence(time, to<real3>(FWD(loc)));
        }

        template <TupleLike L>
            requires ArrayFromTuple<real3, L> real laplacian(real time, L&& loc)
        const
        {
            assert(s);
            return s->laplacian(time, to<real3>(FWD(loc)));
        }

        auto operator()(real time) const
        {
            assert(s);
            return vs::transform(
                [this, time](auto&& loc) { return (*this)(time, FWD(loc)); });
        }

        auto ddt(real time) const
        {
            assert(s);
            return vs::transform(
                [this, time](auto&& loc) { return ddt(time, FWD(loc)); });
        }

        auto gradient(real time) const
        {
            assert(s);
            return vs::transform(
                [this, time](auto&& loc) { return gradient(time, FWD(loc)); });
        }

        auto gradient(int i, real time) const
        {
            assert(s);
            return vs::transform(
                [this, i, time](auto&& loc) { return gradient(time, FWD(loc))[i]; });
        }

        auto divergence(real time) const
        {
            assert(s);
            return vs::transform(
                [this, time](auto&& loc) { return divergence(time, FWD(loc)); });
        }

        auto laplacian(real time) const
        {
            assert(s);
            return vs::transform(
                [this, time](auto&& loc) { return laplacian(time, FWD(loc)); });
        }
};

} // namespace ccs
