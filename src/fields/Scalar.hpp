#pragma once
#include "Tuple.hpp"

namespace ccs::field
{

template <typename D, typename X = D, typename Y = D, typename Z = D>
using Scalar = Tuple<Tuple<D>, Tuple<X, Y, Z>>;
// template <typename T>
// class Scalar
// {
//     T t;

// public:
//     Scalar() = default;
//     Scalar(T&& t) : t{FWD(t)} {}

//     decltype(auto) D() { return (t); }
//     decltype(auto) D() const { return (t); }

//     decltype(auto) Rx() { return (t); }
//     decltype(auto) Rx() const { return (t); }

//     decltype(auto) Ry() { return (t); }
//     decltype(auto) Ry() const { return (t); }

//     decltype(auto) Rz() { return (t); }
//     decltype(auto) Rz() const { return (t); }
// };

// template <typename T>
// Scalar(T&&) -> Scalar<T>;
} // namespace ccs::field