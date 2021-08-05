#pragma once

#include "mesh/mesh_types.hpp"
#include "operators/boundaries.hpp"
#include "types.hpp"

#include <concepts>
#include <optional>

#include <sol/forward.hpp>

namespace ccs::stencils
{

// A stencil is comprised of both an interior discretization and its
// associated numerical boundary scheme.
// The interior discretization is of order 2p and containts 2p + 1 points
// The nbs is a dense small dense rxt matrix
// A stencil can have extra information associated with it (like Neumann BC's)
struct info {
    int p;
    int r;
    int t;
    int nextra;
};

struct interp_info {
    int p;
    int t;
};

struct interp_line {
    std::span<const real> v;
    boundary left;
    boundary right;
};

// satisfaction of Stencil makes a type a stencil
template <typename T>
concept Stencil = requires(const T& st,
                           bcs::type b,
                           real h,
                           real psi,
                           bool ray_outside,
                           std::span<real> c,
                           std::span<real> extra)
{
    {
        st.query(b)
        } -> std::same_as<info>;

    {
        st.query_max()
        } -> std::same_as<info>;

    {st.nbs(h, b, psi, ray_outside, c, extra)};

    {st.interior(h, c)};
};

class stencil
{
    // private nested classes for type erasure
    class any_stencil
    {
    public:
        virtual ~any_stencil() {}
        virtual any_stencil* clone() const = 0;
        virtual info query(bcs::type) const = 0;
        virtual info query_max() const = 0;
        virtual interp_info query_interp() const = 0;
        virtual void nbs(real h,
                         bcs::type,
                         real psi,
                         bool ray_outside,
                         std::span<real> coeffs,
                         std::span<real> extra) const = 0;
        virtual void interior(real c, std::span<real> coeffs) const = 0;
        virtual std::span<const real> interp_interior(real, std::span<real>) const = 0;
        virtual std::span<const real>
        interp_wall(int i, real y, real psi, std::span<real> c, bool right) const = 0;
    };

    template <Stencil S>
    class any_stencil_impl : public any_stencil
    {
        S s;

    public:
        any_stencil_impl(const S& s) : s{s} {}
        any_stencil_impl(S&& s) : s{std::move(s)} {}

        any_stencil_impl* clone() const override { return new any_stencil_impl(s); }

        info query(bcs::type b) const override { return s.query(b); }
        info query_max() const override { return s.query_max(); }
        interp_info query_interp() const override { return s.query_interp(); }

        void nbs(real h,
                 bcs::type b,
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

        std::span<const real> interp_interior(real y, std::span<real> c) const override
        {
            return s.interp_interior(y, c);
        }

        std::span<const real>
        interp_wall(int i, real y, real psi, std::span<real> c, bool right) const override
        {
            return s.interp_wall(i, y, psi, c, right);
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

        info query(bcs::type b) const { return s->query(b); }
        info query_max() const { return s->query_max(); }

        void nbs(real h,
                 bcs::type b,
                 real psi,
                 bool ray_outside,
                 std::span<real> c,
                 std::span<real> ex) const
        {
            return s->nbs(h, b, psi, ray_outside, c, ex);
        }

        void interior(real h, std::span<real> c) const { return s->interior(h, c); }

        std::span<const real> interp_interior(real y, std::span<real> c) const
        {
            return s->interp_interior(y, c);
        }

        std::span<const real>
        interp_wall(int i, real y, real psi, std::span<real> c, bool right) const
        {
            return s->interp_wall(i, y, psi, c, right);
        }

        interp_line interp(int dir,
                           int3 closest,
                           real y,
                           const boundary& left,
                           const boundary& right,
                           std::span<real> c) const
        {
            const auto&& [p, t] = s->query_interp();

            const int ic = closest[dir];
            const int lc = left.mesh_coordinate[dir];
            const int rc = right.mesh_coordinate[dir];
            int st1 = p & 1 ? (p - 1) / 2 : p / 2;
            int st2 = (p - 1) / 2;
            if (y > 0) std::swap(st1, st2);

            // we use an interior interpolant if the boundaries are further away than p on
            // both sides, or if they are are equal to p and not an object
            if ((ic - lc > st1 || (ic - lc == st1 && !left.object)) &&
                (rc - ic > st2 || (rc - ic == st2 && !right.object))) {
                int3 l{closest};
                int3 r{closest};
                l[dir] -= st1;
                r[dir] += st2;

                return {s->interp_interior(y, c),
                        boundary{l, std::nullopt},
                        boundary{r, std::nullopt}};
            } else if (ic - lc < rc - ic) {
                // left boundary is closer
                int3 r{left.mesh_coordinate};
                r[dir] += t - 1;

                const auto& obj = left.object;
                real psi = obj ? obj->psi : 1.0;

                // adjust y if the closest point is in the solid
                if (obj && lc == ic) y += psi - 1;

                return {s->interp_wall(ic - lc, y, psi, c, false),
                        left,
                        boundary{r, std::nullopt}};

            } else {
                // right boundary is closer
                int3 l{right.mesh_coordinate};
                l[dir] -= t - 1;

                const auto& obj = right.object;
                real psi = obj ? obj->psi : 1.0;

                if (obj && rc == ic) y += 1 - psi;

                return {s->interp_wall(rc - ic, y, psi, c, true),
                        boundary{l, std::nullopt},
                        right};
            }
        }

        static std::optional<stencil> from_lua(const sol::table&);
};

stencil make_E2_2();
stencil make_E4_2();
stencil make_E2_1(std::span<const real>);

namespace second
{
extern stencil E2;
extern stencil E4;
} // namespace second

} // namespace ccs::stencils

namespace ccs
{
using stencils::stencil;
}
