#pragma once

#include "Tuple.hpp"

#include "Scalar_fwd.hpp"

namespace ccs::field::tuple
{

// template <typename D, typename X = D, typename Y = D, typename Z = D>
// using Scalar = Tuple<Tuple<D>, Tuple<X, Y, Z>>;

template <traits::OneTuple U, traits::ThreeTuple V>
class Scalar : public Tuple<U, V>
{
    using Base = Tuple<U, V>;
    using D = U;
    using R = V;

    template <traits::OneTuple, traits::ThreeTuple>
    friend class Scalar;

    Base& base() & { return static_cast<Base&>(*this); }
    const Base& base() const& { return static_cast<const Base&>(*this); }
    Base&& base() && { return static_cast<Base&&>(*this); }

    const mesh::Location* location_ = nullptr;

public:
    explicit Scalar() = default;
    explicit Scalar(U&& u, V&& v) : Base{FWD(u), FWD(v)}, location_{nullptr} {}
    explicit Scalar(const mesh::Location* location, U&& u, V&& v)
        : Base{FWD(u), FWD(v)}, location_{location}
    {
    }

    template <traits::ScalarType S>
    requires std::constructible_from<
        Base,
        typename std::remove_cvref_t<S>::Base> explicit Scalar(S&& s)
        : Base{FWD(s).base()}, location_{FWD(s).location_}
    {
    }

    template <traits::ScalarType S>
    // requires std::assignable_from<Base&, typename std::remove_cvref_t<S>::Base>
    Scalar& operator=(S&& s)
    {
        location_ = s.location();
        Base::operator=(FWD(s).base());
        return *this;
    }

    Scalar(const Scalar&) = default;
    Scalar(Scalar&&) = default;

    Scalar& operator=(const Scalar&) = default;
    Scalar& operator=(Scalar&&) = default;

    template <std::invocable<Scalar&> Fn>
    Scalar& operator=(Fn&& f)
    {
        std::invoke(FWD(f), *this);
        return *this;
    }

    const mesh::Location* location() const { return location_; }
};

} // namespace ccs::field::tuple
