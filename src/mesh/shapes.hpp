#pragma once

#include "ray.hpp"
#include <concepts>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace ccs
{

struct hit_info {
    real t;           // t for which the incident ray struck the object
    real3 position;   // ray.position(t)
    bool ray_outside; // true if ray came from outside object
    int shape_id;
};

// shape concept
template <typename S>
concept Shape = requires(const S& shape, const ray& r, real t, const real3& pos)
{
    {
        shape.hit(r, t, t)
        } -> std::same_as<std::optional<hit_info>>;

    {
        shape.normal(pos)
        } -> std::same_as<real3>;
};

// use type-erasure for defining shapes so we can more easily
// interact with lua and keep value semantics
class shape
{
    // private nested classes for type erasure
    class any_shape
    {
    public:
        virtual ~any_shape() {}
        virtual any_shape* clone() const = 0;
        virtual std::optional<hit_info> hit(const ray&, real, real) const = 0;
        virtual real3 normal(const real3&) const = 0;
    };

    template <Shape S>
    class any_shape_impl : public any_shape
    {
        S s;

    public:
        any_shape_impl(const S& s) : s{s} {}
        any_shape_impl(S&& s) : s{std::move(s)} {}

        any_shape_impl* clone() const override { return new any_shape_impl(s); }

        std::optional<hit_info> hit(const ray& r, real t_min, real t_max) const override
        {
            return s.hit(r, t_min, t_max);
        }

        real3 normal(const real3& pos) const override { return s.normal(pos); }
    };

    any_shape* s;

public:
    shape() : s{nullptr} {}

    shape(const shape& other) : s{nullptr}
    {
        if (other) s = other.s->clone();
    }

    shape(shape&& other) : s{std::exchange(other.s, nullptr)} {}

    // construction from anything with a hit method
    template <typename T>
        requires Shape<T> &&(!std::same_as<shape, std::remove_cvref_t<T>>)shape(T&& other)
            : s{new any_shape_impl{std::forward<T>(other)}}
        {
        }

        shape& operator=(const shape& other)
        {
            shape tmp{other};
            swap(*this, tmp);
            return (*this);
        }

        shape& operator=(shape&& other)
        {
            delete s;
            s = std::exchange(other.s, nullptr);
            return *this;
        }

        ~shape() { delete s; }

        friend void swap(shape& x, shape& y) { std::swap(x.s, y.s); }

        explicit operator bool() const { return s != nullptr; }

        std::optional<hit_info> hit(const ray& r, real t_min, real t_max) const
        {
            if (*this)
                return s->hit(r, t_min, t_max);
            else
                return std::nullopt;
        }

        real3 normal(const real3& pos) const
        {
            if (*this)
                return s->normal(pos);
            else
                return {};
        }
};

// factory functions
shape make_sphere(int id, const real3& origin, real radius);
shape make_xy_rect(int id, const real3& corner0, const real3& corner1, real fluid_normal);
shape make_xz_rect(int id, const real3& corner0, const real3& corner1, real fluid_normal);
shape make_yz_rect(int id, const real3& corner0, const real3& corner1, real fluid_normal);

} // namespace ccs
