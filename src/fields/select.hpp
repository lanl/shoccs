#include "result_field.hpp"
#include "scalar.hpp"
#include "scalar_field.hpp"
#include "types.hpp"
#include "vector.hpp"
#include "vector_field.hpp"

#include <range/v3/range/concepts.hpp>
#include <range/v3/view/iota.hpp>

namespace ccs
{

// forward decl public api
template <typename R, typename I>
auto select(R&& r, I&& i);

template <typename I>
auto select(I&& i);

namespace detail
{
// should think about constraints on R, something like random access range or vector field
template <typename R, typename I, typename A>
struct selection {
    R r;
    I i;
    A action;
};

template <typename R, typename I>
selection(R&&, I&&) -> selection<R, I, std::identity>;

template <typename R, typename I, typename A>
selection(R&&, I&&, A&&) -> selection<R, I, A>;

template <typename I, typename P, typename A>
struct selection_helper {
    I i;
    P proj;
    A action;
};
template <typename I>
selection_helper(I&&) -> selection_helper<I, std::identity, std::identity>;

template <typename I, typename P>
selection_helper(I&&, P&&) -> selection_helper<I, P, std::identity>;

template <typename I, typename P, typename A>
selection_helper(I&&, P&&, A&&) -> selection_helper<I, P, A>;

namespace traits
{
template <typename = void>
struct is_selection : std::false_type {
};

template <typename R, typename I, typename A>
struct is_selection<selection<R, I, A>> : std::true_type {
};

template <typename T>
constexpr bool is_selection_v = is_selection<std::remove_cvref_t<T>>::value;

template <typename = void>
struct is_selection_helper : std::false_type {
};

template <typename I, typename P, typename A>
struct is_selection_helper<selection_helper<I, P, A>> : std::true_type {
};

template <typename T>
constexpr bool is_selection_helper_v = is_selection_helper<std::remove_cvref_t<T>>::value;

template <typename T>
concept Selection = is_selection_v<T>;

template <typename T>
concept Selection_Helper = is_selection_helper_v<T>;

// exctract range type
template <typename>
struct selection_range;

template <typename R, typename I, typename A>
struct selection_range<selection<R, I, A>> {
    using type = R;
};
template <typename U>
using selection_range_t = typename selection_range<std::remove_cvref_t<U>>::type;

// extract index type
template <typename>
struct selection_index;
template <typename R, typename I, typename A>
struct selection_index<selection<R, I, A>> {
    using type = I;
};
template <typename U>
using selection_index_t = typename selection_index<std::remove_cvref_t<U>>::type;

template <typename T>
concept Scalar_Selection_Range =
    Selection<T>&& rs::random_access_range<selection_range_t<T>>;
template <typename T>
concept Scalar_Selection_Index = Selection<T>&& rs::input_range<selection_index_t<T>>;

template <typename T>
concept Vector_Selection_Range =
    Selection<T> && (!Scalar_Selection_Range<T>)&&Vector_Field<selection_range_t<T>>;
template <typename T>
concept Vector_Selection_Index =
    Selection<T> && (!Scalar_Selection_Index<T>)&&Vector_Field<selection_index_t<T>>;

template <typename T>
concept Scalar_Scalar_Selection = Scalar_Selection_Range<T>&& Scalar_Selection_Index<T>;
template <typename T>
concept Vector_Scalar_Selection = Vector_Selection_Range<T>&& Scalar_Selection_Index<T>;
template <typename T>
concept Scalar_Vector_Selection = Scalar_Selection_Range<T>&& Vector_Selection_Index<T>;
template <typename T>
concept Vector_Vector_Selection = Vector_Selection_Range<T>&& Vector_Selection_Index<T>;

} // namespace traits

template <typename R, traits::Selection_Helper S>
auto operator>>(R&& r, S&& s)
{
    return selection{std::invoke(s.proj, FWD(r)),
                     FWD(s).i,
                     [action = FWD(s).action, &r]() { std::invoke(action, r); }};
}

// allow for chaining transformations on the selections
template <traits::Selection S, typename ViewFn>
auto operator>>(S&& s, vs::view_closure<ViewFn> t)
{
    return selection{FWD(s).r >> t, FWD(s).i, FWD(s).action};
}

// support: field >> select(input_range) <<= container
// this modifies the selected indices in field
// Combinations that make sense:
// container_type, field_type , input_range_type
// ---------------------------------------------
// scalar, scalar, scalar
// scalar, vector, scalar
// scalar, vector, vector
// vector, vector, scalar
// vector, vector, vector
template <traits::Scalar_Scalar_Selection S, rs::input_range R>
void select_in_place(S&& s, R&& r)
{
    for (auto&& [i, v] : vs::zip(s.i, r)) s.r(i) = v;
}

template <traits::Vector_Scalar_Selection S, rs::input_range R>
void select_in_place(S&& s, R&& r)
{
    for (auto&& [i, v] : vs::zip(s.i, r)) s.r.xi(i) = v;
    for (auto&& [i, v] : vs::zip(s.i, r)) s.r.yi(i) = v;
    for (auto&& [i, v] : vs::zip(s.i, r)) s.r.zi(i) = v;
}

template <traits::Vector_Vector_Selection S, rs::input_range R>
void select_in_place(S&& s, R&& r)
{
    for (auto&& [i, v] : vs::zip(s.i.x, r)) s.r.xi(i) = v;
    for (auto&& [i, v] : vs::zip(s.i.y, r)) s.r.yi(i) = v;
    for (auto&& [i, v] : vs::zip(s.i.z, r)) s.r.zi(i) = v;
}

template <traits::Vector_Scalar_Selection S, Vector_Field R>
void select_in_place(S&& s, R&& r)
{
    for (auto&& [i, v] : vs::zip(s.i, r.x)) s.r.xi(i) = v;
    for (auto&& [i, v] : vs::zip(s.i, r.y)) s.r.yi(i) = v;
    for (auto&& [i, v] : vs::zip(s.i, r.z)) s.r.zi(i) = v;
}

template <traits::Vector_Vector_Selection S, Vector_Field R>
void select_in_place(S&& s, R&& r)
{
    for (auto&& [i, v] : vs::zip(s.i.x, r.x)) s.r.xi(i) = v;
    for (auto&& [i, v] : vs::zip(s.i.y, r.y)) s.r.yi(i) = v;
    for (auto&& [i, v] : vs::zip(s.i.z, r.z)) s.r.zi(i) = v;
}

template <traits::Selection S, typename R>
auto operator<<=(S&& s, R&& r) -> decltype((void)select_in_place(s, FWD(r)))
{
    select_in_place(s, FWD(r));
    std::invoke(s.action);
}

namespace detail
{
template <Numeric R>
constexpr auto sized_repeat(R r, int s)
{
    return vs::repeat_n(r, s);
}

template <Numeric R>
constexpr auto sized_repeat(R r, int3 s)
{
    return vector_range{
        vs::repeat_n(r, s[0]), vs::repeat_n(r, s[1]), vs::repeat_n(r, s[2])};
}
} // namespace detail

template <traits::Selection S, Numeric R>
auto operator<<=(S&& s, R r)
{
    constexpr bool sized_index = requires(S s, R r)
    {
        detail::sized_repeat(r, s.i.size());
    };
    constexpr bool sized_range = requires(S s, R r)
    {
        detail::sized_repeat(r, s.r.size());
    };
    static_assert(sized_index || sized_range);
    if constexpr (sized_index)
        return FWD(s) <<= detail::sized_repeat(r, s.i.size());
    else
        return FWD(s) <<= detail::sized_repeat(r, s.r.size());
}

// support: container <<= field >> select(input_range)
// this fills the container with the selected indicies from the field
// Combinations that make sense:
// container_type, field_type , input_range_type
// ---------------------------------------------
// scalar,  scalar, scalar
// vector,  scalar, scalar <- copies identical data to each component
// vector,  scalar, vector
// vector,  vector, scalar
// vector,  vector, vector
template <rs::range R, traits::Scalar_Scalar_Selection S>
void select_copy_into(R&& r, S&& s)
{
    for (auto&& [i, v] : vs::zip(s.i, r)) v = s.r(i);
}

template <Vector_Field R, traits::Scalar_Scalar_Selection S>
void select_copy_into(R&& r, S&& s)
{
    for (auto&& [i, v] : vs::zip(s.i, r.x)) v = s.r(i);
    for (auto&& [i, v] : vs::zip(s.i, r.y)) v = s.r(i);
    for (auto&& [i, v] : vs::zip(s.i, r.z)) v = s.r(i);
}

template <Vector_Field R, traits::Scalar_Vector_Selection S>
void select_copy_into(R&& r, S&& s)
{
    for (auto&& [i, v] : vs::zip(s.i.x, r.x)) v = s.r(i);
    for (auto&& [i, v] : vs::zip(s.i.y, r.y)) v = s.r(i);
    for (auto&& [i, v] : vs::zip(s.i.z, r.z)) v = s.r(i);
}

template <Vector_Field R, traits::Vector_Scalar_Selection S>
void select_copy_into(R&& r, S&& s)
{
    for (auto&& [i, v] : vs::zip(s.i, r.x)) v = s.r.xi(i);
    for (auto&& [i, v] : vs::zip(s.i, r.y)) v = s.r.yi(i);
    for (auto&& [i, v] : vs::zip(s.i, r.z)) v = s.r.zi(i);
}

template <Vector_Field R, traits::Vector_Vector_Selection S>
void select_copy_into(R&& r, S&& s)
{
    for (auto&& [i, v] : vs::zip(s.i.x, r.x)) v = s.r.xi(i);
    for (auto&& [i, v] : vs::zip(s.i.y, r.y)) v = s.r.yi(i);
    for (auto&& [i, v] : vs::zip(s.i.z, r.z)) v = s.r.zi(i);
}

template <typename R, traits::Selection S>
auto operator<<=(R&& r, S&& s) -> decltype(select_copy_into(FWD(r), FWD(s)))
{
    // attempt to resize the output range if we can
    constexpr bool can_resize = requires(R a, S b) { a.resize(b.i.size()); };
    if constexpr (can_resize) r.resize(s.i.size());

    return select_copy_into(FWD(r), FWD(s));
}

} // namespace detail

template <typename R, typename I>
auto select(R&& r, I&& i)
{
    return detail::selection{FWD(r), FWD(i)};
}

// single argument version
template <typename I>
auto select(I&& i)
{
    return detail::selection_helper{FWD(i)};
}

// for scalars/vector
template <typename I>
auto field_select(I&& i)
{
    return detail::selection_helper{FWD(i),
                                    [](auto&& s) -> decltype(auto) { return (s.field); }};
}

template <typename I>
auto x_field_select(I&& i)
{
    return detail::selection_helper{
        FWD(i), [](Vector auto&& v) -> decltype(auto) { return (v.field.x); }};
}

template <typename I>
auto y_field_select(I&& i)
{
    return detail::selection_helper{
        FWD(i), [](Vector auto&& v) -> decltype(auto) { return (v.field.y); }};
}

template <typename I>
auto z_field_select(I&& i)
{
    return detail::selection_helper{
        FWD(i), [](Vector auto&& v) -> decltype(auto) { return (v.field.z); }};
}

template <typename I>
auto obj_select(I&& i)
{
    return detail::selection_helper{
        FWD(i),
        [](auto&& s) -> decltype(auto) { return (s.obj); },
        []<typename T>(T&& s) {
            if constexpr (Scalar<T>) {
                using Field = typename std::remove_cvref_t<T>::S;
                s.field >> select(s.m.template get<traits::scalar_dim<Field>>()) <<=
                    s.obj.template get<traits::scalar_dim<Field>>();
            } else { // a vector
                s.field >> select(s.m) <<= s.obj;
            }
        }};
}

auto obj_select() { return obj_select(vs::iota(0)); }

} // namespace ccs