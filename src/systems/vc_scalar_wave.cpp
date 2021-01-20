#include "vc_scalar_wave.hpp"

#include <cassert>
#include <cmath>
#include <numbers>

constexpr real twoPI = 2 * std::numbers::pi_v<real>;

namespace ccs
{

// negative gradient - coefficients of gradient
template <int I>
struct neg_G {
    real3 center;
    real radius;

    real operator()(const real3& location) const
    {
        return -(location[I] - center[I]) / length(location - center);
    }
};

struct sol {
    real3 center;
    real radius;
    real time;

    real operator()(const mesh_object_info& info) const { return (*this)(info.position); }

    real operator()(const real3& location) const
    {
        return std::sin(twoPI * (length(location - center) - radius - time));
    }

    // gradient for neumann boundaries
    real3 grad(const real3& location) const
    {
        const auto d = location - center;
        return twoPI * d * std::cos(twoPI * (length(d) - radius - time)) / length(d);
    }
};

vc_scalar_wave::vc_scalar_wave(cart_mesh&& cart_,
                               mesh&& cut_mesh_,
                               discrete_operator&& grad_,
                               field_io& io,
                               std::array<double, 3> center_,
                               double radius_,
                               double stats_begin_accumulate_)
    : system{std::move(cart_), std::move(cut_mesh_)},
      grad{std::move(grad_)},
      center{center_},
      radius{radius_},
      stats0{},
      stats_begin_accumulate{stats_begin_accumulate_}
{
    // allocate mesh wide data
    x_field u0{m.extents()};
    x_field u{m.extents()};
    x_field error{m.extents()}; // or should this be a result_field?
    v_field grad_u{m.extents()};
    v_vield coeffs{m.extents()};
    x_field du{m.extents()}; // dummy for neumann bc's

    // initialize all data sin 2pi(G-t)
    u.field = location_view<0>(m) | vs::transform(solution{center, radius, 0});
    // or what about
    u >> field_select() <<=
        location_view<0>(m) | vs::transform(solution{center, radius, 0});
    // if we had domain boundaries to set, it would be something like:
    u.field >> select(index_view<0>(m, 0)) <<=
        location_view<0>(m, 0) >> vs::transform(solution{center, radius, 0});
    // or
    u >> field_select(index_view<0>(0)) <<= location_view<0>(m, 0)) >>
        vs::transform(solution{center, radius, 0});
    u >> obj_select() <<= geom.Rxyz() >> vs::transform(solution{center, radius, 0});

    // or could we do the whole thing in one shot?
    u = scalar_proxy{location_view<0>(m) | vs::transform(solution{center, radius, 0}),
                     geom.Rxyz() >> vs::transform(solution{center, radius, 0})};
    // or something like
    u = scalar{arg_{...}, arg_{...}}
    // don't really need this if we make such things automatic in the field/
    u >> select(geom.Sx()) <<= 0;
    // there could be multiple objects represented by the object range, we could set all
    // bc's with something like
    u >> obj_select() <<= geom.Rxyz() >> vs::transform([](auto&& info) {
                              if (info.shape_id == 1)
                                  return x;
                              else
                                  return y;
                          });
    // and this would automatically set the field values to the boundary values using the
    // solid points in the domain.  But for a scalar associated with a particular direction
    // it would only make sense to set the solid points (boundary points) associated with
    // that direction.

    // or if we already want a scalar quantity to have a mapping of solid points, could we
    // also have a mapping of the mesh-object-info struct?  In that case obj_select, could
    // take a predication that would be used to filter out the points in the obj..
    u >> obj_select([](auto&& info) { return info.shape_id == 1; });
    // but setting the values would still require applying the filter to geom.Rxyz and
    // then doing a transform.  Maybe this isn't worthwhile.

    // precompute the coefficients grad(psi) and zero the solid points
    coeffs.x = location_view<0>(m) | vs::transform(neg_G<0>{center, radius});
    coeffs.y = location_view<1>(m) | vs::transform(neg_G<1>{center, radius});
    coeffs.z = location_view<2>(m) | vs::transform(neg_G<2>{center, radius});
    coeffs >> select(geom.Sxyz()) <<= 0;
    // Representing this as a vector we would want somthing like:
    coeffs = {vector_range{location_view_all(m)} >>
                  vector_range{vs::transform(neg_G<0>{center, radius, time}),
                               vs::transform(neg_G<1>{center, radius, time}),
                               vs::transform(neg_G<2>{center, radius, time})},
              0};
    // or
    coeffs.field = 
    coeffs >> obj_select() << 0;
    // or
    coeffs >> obj_select() <<=
        geom.Rxyz() >> v_transform(neg_G<0>{center, radius, time}, ...)

    // there should be some processing of boundary conditions and operator generation here
    // only valid bc's for this problem are outflow around the domain and inflow from the
    // embedded object

    // register 'u' and 'error' with io.  The current setup of capturing this pointer
    // means we can't invalidate the vector via moves at any point.  Ranges likes to
    // do this so we use iterators instead (for now)
    // io.add("U", &u[0]);
    // io.add("Error", &error[0]);
}

real vc_scalar_wave::system_timestep_size(real cfl) const
{
    return cfl * std::min(m.h());
}

// Evaluate the rhs of the system using the current u
// this shouldn't be a std::vector<double>... This should be encapsulated in some
// kind of field, vector/scalar field seems inadequate.  What would a system field look
// like? It could contain vector and scalar fields.  Should it be owning (vectors) or
// non-owning (spans)? If system_field be a templated entity than the call operator can't
// be virtual (perhaps tag_invoke)?
void vc_scalar_wave::operator()(std::vector<double>& rhs, double time)
{
    // prepare bc's
    f_bvalues <<=
        geom.Rxyz() >> vs::transform([s = solution{center, radius, time}](auto&& info) {
            return s(info.position);
        });

    grad(u, du, u_bvalues, du_bvalues, dxyz);
    rhs = contract(grad_G * dxyz, std::plus{});
}

system_stats vc_scalar_wave::stats(double time)
{
    this->stats_(stats0, time >= stats_begin_accumulate, time, solution{center, radius});
    return stats0;
}

int_t vc_scalar_wave::rhs_size() const { return cart.total_size(); }

std::unique_ptr<system> build_system(cart_mesh&& cart,
                                     mesh&& cut_mesh,
                                     discrete_operator&& grad,
                                     field_io& io,
                                     std::array<double, 3> center,
                                     double radius,
                                     double stats_begin_accumulate)
{
    return std::make_unique<vc_scalar_wave>(std::move(cart),
                                            std::move(cut_mesh),
                                            std::move(grad),
                                            io,
                                            center,
                                            radius,
                                            stats_begin_accumulate);
}

} // namespace ccs
