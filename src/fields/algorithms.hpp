#pragma once

#include "Vector.hpp"

namespace ccs::field::tuple
{

// template <traits::TupleType T, typename Fn>
// constexpr auto reduce(T&& t, Fn f)
// {
//     return [&f]<auto... Is>(std::index_sequence<Is...>, auto&& tup)
//     {
//         return Tuple(f(Tuple{view<Is>(tup)}...));
//     }
//     (std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>{}, FWD(t));
// }

template <traits::VectorType T, traits::VectorType U>
constexpr auto dot(T&& t, U&& u)
{
    auto [x, y, z] = t * u;
    return x + y + z;
}

} // namespace ccs::field::tuple