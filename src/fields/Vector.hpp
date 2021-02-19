#pragma once

#include "Tuple.hpp"
#include "Vector_fwd.hpp"
// can make this Scalar_fwd once the return of dot depends on the template parameter
#include "Scalar.hpp"

namespace ccs::field::tuple
{
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

    template<typename T>
    constexpr friend auto dot(const Vector&, const T&) {
        return SimpleScalar<std::vector<real>>{};
    }

public:
    Vector() = default;

    Vector(U&& u, V&& v) : Base{FWD(u), FWD(v)}, location_{nullptr} {}
    Vector(const mesh::Location* location, U&& u, V&& v)
        : Base{FWD(u), FWD(v)}, location_{location}
    {
    }

    template <traits::VectorType S>
    requires std::constructible_from<Base, typename std::remove_cvref_t<S>::Base>
    Vector(S&& s) : Base{FWD(s).base()}, location_{FWD(s).location_}
    {
    }

    Vector(const Vector&) = default;
    Vector(Vector&&) = default;

    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&&) = default;

    const mesh::Location* location() const { return location_; }
};

} // namespace ccs::field::tuple