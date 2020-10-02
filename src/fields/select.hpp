#include "result_field.hpp"
#include "scalar_field.hpp"
#include "types.hpp"
#include "vector_field.hpp"
#include <range/v3/range/concepts.hpp>

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
template <typename R, typename I>
struct selection {
    R r;
    I i;
};

template <typename R, typename I>
selection(R&&, I &&) -> selection<R, I>;

template <typename I>
struct selection_helper {
    I i;
};
template <typename I>
selection_helper(I &&) -> selection_helper<I>;

namespace traits
{
template <typename = void>
struct is_selection : std::false_type {
};

template <typename R, typename I>
struct is_selection<selection<R, I>> : std::true_type {
};

template <typename T>
constexpr bool is_selection_v = is_selection<std::remove_cvref_t<T>>::value;

template <typename = void>
struct is_selection_helper : std::false_type {
};

template <typename I>
struct is_selection_helper<selection_helper<I>> : std::true_type {
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

template <typename R, typename I>
struct selection_range<selection<R, I>> {
    using type = R;
};
template <typename U>
using selection_range_t = typename selection_range<std::remove_cvref_t<U>>::type;

// extract index type
template <typename>
struct selection_index;
template <typename R, typename I>
struct selection_index<selection<R, I>> {
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
    return selection{std::forward<R>(r), std::forward<S>(s).i};
}

// allow for chaining transformations on the selections
template <traits::Selection S, typename ViewFn>
auto operator>>(S&& s, vs::view_closure<ViewFn> t)
{
    return selection{std::forward<S>(s).r >> t, std::forward<S>(s).i};
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
    for (auto&& [i, v] : vs::zip(s.i, r)) s.r.x(i) = v;
    for (auto&& [i, v] : vs::zip(s.i, r)) s.r.y(i) = v;
    for (auto&& [i, v] : vs::zip(s.i, r)) s.r.z(i) = v;
}

template <traits::Vector_Vector_Selection S, rs::input_range R>
void select_in_place(S&& s, R&& r)
{
    for (auto&& [i, v] : vs::zip(s.i.x, r)) s.r.x(i) = v;
    for (auto&& [i, v] : vs::zip(s.i.y, r)) s.r.y(i) = v;
    for (auto&& [i, v] : vs::zip(s.i.z, r)) s.r.z(i) = v;
}

template <traits::Vector_Scalar_Selection S, Vector_Field R>
void select_in_place(S&& s, R&& r)
{
    for (auto&& [i, v] : vs::zip(s.i, r.x)) s.r.x(i) = v;
    for (auto&& [i, v] : vs::zip(s.i, r.y)) s.r.y(i) = v;
    for (auto&& [i, v] : vs::zip(s.i, r.z)) s.r.z(i) = v;
}

template <traits::Vector_Vector_Selection S, Vector_Field R>
void select_in_place(S&& s, R&& r)
{
    for (auto&& [i, v] : vs::zip(s.i.x, r.x)) s.r.x(i) = v;
    for (auto&& [i, v] : vs::zip(s.i.y, r.y)) s.r.y(i) = v;
    for (auto&& [i, v] : vs::zip(s.i.z, r.z)) s.r.z(i) = v;
}

template <traits::Selection S, typename R>
auto operator<<=(S&& s, R&& r) -> decltype(select_in_place(std::forward<S>(s), std::forward<R>(r)))
{
    return select_in_place(std::forward<S>(s), std::forward<R>(r));
}

template <traits::Selection S, Numeric R>
auto operator<<=(S&& s, R r) -> decltype(select_in_place(std::forward<S>(s), vs::repeat(r)))
{
    return select_in_place(std::forward<S>(s), vs::repeat(r));
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
    for (auto&& [i, v] : vs::zip(s.i, r.x)) v = s.r.x(i);
    for (auto&& [i, v] : vs::zip(s.i, r.y)) v = s.r.y(i);
    for (auto&& [i, v] : vs::zip(s.i, r.z)) v = s.r.z(i);
}

template <Vector_Field R, traits::Vector_Vector_Selection S>
void select_copy_into(R&& r, S&& s)
{
    for (auto&& [i, v] : vs::zip(s.i.x, r.x)) v = s.r.x(i);
    for (auto&& [i, v] : vs::zip(s.i.y, r.y)) v = s.r.y(i);
    for (auto&& [i, v] : vs::zip(s.i.z, r.z)) v = s.r.z(i);
}

template <typename R, traits::Selection S>
auto operator<<=(R&& r, S&& s) -> decltype(select_copy_into(std::forward<R>(r), std::forward<S>(s)))
{
    // attempt to resize the output range if we can
    constexpr bool can_resize = requires(R a, S b) { a.resize(b.i.size()); };
    if constexpr (can_resize) r.resize(s.i.size());

    return select_copy_into(std::forward<R>(r), std::forward<S>(s));
}

} // namespace detail

template <typename R, typename I>
auto select(R&& r, I&& i)
{
    return detail::selection{std::forward<R>(r), std::forward<I>(i)};
}

// single argument version
template <typename I>
auto select(I&& i)
{
    return detail::selection_helper{std::forward<I>(i)};
}
} // namespace ccs