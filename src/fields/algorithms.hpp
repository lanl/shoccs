#pragma once

#include "Tuple_fwd.hpp"
#include "Vector_fwd.hpp"

namespace ccs::field::tuple
{

template <traits::TupleType T, typename Fn>
constexpr auto reduce(T&& t, Fn f)
{
    return [&f]<auto... Is>(std::index_sequence<Is...>, auto&& tup)
    {
        return Tuple(f(Tuple{view<Is>(tup)}...));
    }
    (std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>{}, FWD(t));
}

template <traits::VectorType T, traits::VectorType U>
constexpr auto dot(T&& t, U&& u)
{

    auto tu = FWD(t) * FWD(u);
    return Scalar(t.location(),
                  reduce((tu | selector::D).as_Tuple(),
                         [](auto&&... args) { return (FWD(args) + ...); }),
                  (tu | selector::Rxyz).as_Tuple());
}

} // namespace ccs::field::tuple