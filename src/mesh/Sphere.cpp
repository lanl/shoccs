#include "real3_operators.hpp"
#include "Shapes.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

namespace ccs
{

struct sphere {
    real3 origin;
    real radius;
    int id;

    std::optional<hit_info> hit(const ray& r, real t_min, real t_max) const
    {
        // check for intersection of ray with objects for t in [t_min, t_max]
        const auto oc = r.origin - origin;
        const auto a = dot(r.direction, r.direction);
        const auto b = dot(oc, r.direction);
        const auto c = dot(oc, oc) - radius * radius;
        const auto discriminant = b * b - a * c;

        if (discriminant > 0) {
            const auto sqr = std::sqrt(discriminant);

            if (auto t = (-b - sqr) / a; t > t_min && t < t_max) {
                auto p = r.position(t);    
                return hit_info{t, p, dot(r.direction, p - origin) < 0, id};
            }

            if (auto t = (-b + sqr) / a; t > t_min && t < t_max) {
                auto p = r.position(t);
                return hit_info{t, p, dot(r.direction, p - origin) < 0, id};
            }
        }
        return std::nullopt;
    }
};

// factory function
shape make_sphere(int id, const real3& origin, real radius)
{
    return {sphere{origin, radius, id}};
}

} // namespace ccs
