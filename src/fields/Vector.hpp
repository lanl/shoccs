#pragma once

#include "Scalar_fwd.hpp"
#include "Tuple.hpp"
#include "Vector_fwd.hpp"

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

    template <traits::VectorType S>
    requires std::constructible_from<
        Base,
        typename std::remove_cvref_t<S>::Base> explicit Vector(S&& s)
        : Base{FWD(s).base()}, location_{FWD(s).location_}
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

    const mesh::Location* location() const { return location_; }
};

} // namespace ccs::field::tuple