#pragma once

namespace ccs::field
{
template <typename... T>
class Vector
{
public:
    Vector() = default;
    Vector(T&&...) {}
};

template <typename... T>
Vector(T&&...) -> Vector<T...>;

} // namespace ccs::field