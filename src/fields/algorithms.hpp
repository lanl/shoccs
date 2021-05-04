#pragma once

#include "vector.hpp"

namespace ccs
{

// template <traits::Tuple T, typename Fn>
// constexpr auto reduce(T&& t, Fn f)
// {
//     return [&f]<auto... Is>(std::index_sequence<Is...>, auto&& tup)
//     {
//         return Tuple(f(Tuple{view<Is>(tup)}...));
//     }
//     (std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>{}, FWD(t));
// }

template <Vector T, Vector U>
constexpr auto dot(T&& t, U&& u)
{
    auto [x, y, z] = t * u;
    return x + y + z;
}

} // namespace ccs
