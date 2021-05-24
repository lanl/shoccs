#pragma once

#include "field_fwd.hpp"

namespace ccs
{

template <Field... T, typename F>
constexpr void for_each_scalar(F&& f, T&&... t)
{
    for (auto&& s : vs::zip(FWD(t).scalars()...)) { std::apply(f, FWD(s)); }
}

template <Field... T, typename F>
constexpr void for_each_vector(F&& f, T&&... t)
{
    for (auto&& s : vs::zip(FWD(t).vectors()...)) { std::apply(f, FWD(s)); }
}

template <Field... T, typename F>
constexpr void for_each(F&& f, T&&... t)
{
    for_each_scalar(f, t...);
    for_each_vector(FWD(f), FWD(t)...);
}

template <Field... T, typename F>
constexpr auto transform_scalar(F&& f, T&&... t)
{
    return vs::zip_with(f, FWD(t).scalars()...);
}

template <Field... T, typename F>
constexpr auto transform_vector(F&& f, T&&... t)
{
    return vs::zip_with(f, FWD(t).vectors()...);
}

template <Field... T, typename F>
constexpr auto transform(F&& f, T&&... t)
{
    return detail::field{transform_scalar(f, t...), transform_vector(f, t...)};
}
} // namespace ccs
