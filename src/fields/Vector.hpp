#pragma once

#include "Scalar.hpp"
#include "Tuple.hpp"
#include "Vector_fwd.hpp"

#include "Location.hpp"

namespace ccs::field::tuple
{

// Vector should really have a ThreeTuple of ThreeTuples corresponding to the
// vector quantities for each component on Rx, Ry, and Rz.  We leave as a single
// ThreeTuple of Ranges for now.
template <traits::ThreeTuple U, traits::ThreeTuple V>
class Vector : public Tuple<U, V>
{
    using Base = Tuple<U, V>;
    using D = U;
    using R = V;

    template <traits::ThreeTuple, traits::ThreeTuple>
    friend class Vector;

    Base& base() & { return static_cast<Base&>(*this); }
    const Base& base() const& { return static_cast<const Base&>(*this); }
    Base&& base() && { return static_cast<Base&&>(*this); }

    const mesh::Location* location_ = nullptr;

public:
    explicit Vector() = default;

    explicit Vector(U&& u, V&& v) : Base{FWD(u), FWD(v)}, location_{nullptr} {}
    explicit Vector(const mesh::Location* location, U&& u, V&& v)
        : Base{FWD(u), FWD(v)}, location_{location}
    {
    }

    explicit Vector(const mesh::Location* location)
        : Base{Tuple{location->x.size() * location->y.size() * location->z.size(),
                     location->x.size() * location->y.size() * location->z.size(),
                     location->x.size() * location->y.size() * location->z.size()},
               Tuple{location->rx.size(), location->ry.size(), location->rz.size()}},
          location_{location}
    {
    }

    template <traits::VectorType S>
    requires std::constructible_from<Base, typename std::remove_cvref_t<S>::Base>
    Vector(S&& s) : Base{FWD(s).base()}, location_{FWD(s).location_}
    {
    }

    template <traits::VectorType S>
    Vector& operator=(S&& s)
    {
        location_ = s.location();
        Base::operator=(FWD(s).base());
        return *this;
    }

    Vector(const Vector&) = default;
    Vector(Vector&&) = default;

    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&&) = default;

    template <std::invocable<Vector&> Fn>
    Vector& operator=(Fn&& f)
    {
        std::invoke(FWD(f), *this);
        return *this;
    }

    ScalarView_Const x() const
    {
        return ScalarView_Const{location_,
                                Tuple{view<0>(this->template get<0>())},
                                this->template get<1>()};
    }

    ScalarView_Mutable x()
    {
        return ScalarView_Mutable{location_,
                                  Tuple{view<0>(this->template get<0>())},
                                  this->template get<1>()};
    }
    ScalarView_Const y() const
    {
        return ScalarView_Const{location_,
                                Tuple{view<1>(this->template get<0>())},
                                this->template get<1>()};
    }

    ScalarView_Mutable y()
    {
        return ScalarView_Mutable{location_,
                                  Tuple{view<1>(this->template get<0>())},
                                  this->template get<1>()};
    }
    ScalarView_Const z() const
    {
        return ScalarView_Const{location_,
                                Tuple{view<2>(this->template get<0>())},
                                this->template get<1>()};
    }

    ScalarView_Mutable z()
    {
        return ScalarView_Mutable{location_,
                                  Tuple{view<2>(this->template get<0>())},
                                  this->template get<1>()};
    }

    const mesh::Location* location() const { return location_; }
};

} // namespace ccs::field::tuple