#pragma once

#include "Tuple.hpp"
#include "Vector_fwd.hpp"

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

public:
    Vector() = default;
};

} // namespace ccs::field::tuple