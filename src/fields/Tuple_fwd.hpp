#pragma once

#include "types.hpp"
#include <concepts>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/common.hpp>
#include <range/v3/view/view.hpp>

#include <boost/mp11.hpp>

// Can we leverage something like MP11 to rewrite
// most of these traits?

// Can we use mp11's mapping capabilities

namespace ccs::field::tuple
{

using std::get;

// Concept for being able to apply vs::all.  Need to exclude int3 as
// an "Allable" range to trigger proper tuple construction calls
template <typename T>
concept All = rs::range<T&>&& rs::viewable_range<T> &&
              (!std::same_as<int3, std::remove_cvref_t<T>>);

// have a more limited definition of input_range so that
// int3 extents are not considered an input_range
namespace traits
{
using namespace boost::mp11;

} // namespace traits

// Forward decls for main types
template <typename...>
struct Tuple;

// some operations (such as the construction of nested r_tuples)
// will require an extra argument to chose the right templated
// function (ie prefering a constructor over a converting constructor)
struct tag {
};

// forward decls for component types
template <typename...>
struct ContainerTuple;

template <typename...>
struct ViewBaseTuple;

template <typename...>
struct ViewTuple;

// traits and concepts for component types
namespace traits
{

//
// ContainerTuple concepts
//
template <typename>
struct is_ContainerTuple : std::false_type {
};

template <typename... Args>
struct is_ContainerTuple<ContainerTuple<Args...>> : std::true_type {
};

template <typename T>
concept ContainerTupleType = is_ContainerTuple<std::remove_cvref_t<T>>::value;

//
// ViewBase/View Tuples
//
template <typename>
struct is_ViewBaseTuple : std::false_type {
};

template <typename... Args>
struct is_ViewBaseTuple<ViewBaseTuple<Args...>> : std::true_type {
};

template <typename T>
concept ViewBaseTupleType = is_ViewBaseTuple<std::remove_cvref_t<T>>::value;

template <typename>
struct is_ViewTuple : std::false_type {
};

template <typename... Args>
struct is_ViewTuple<ViewTuple<Args...>> : std::true_type {
};

template <typename T>
concept ViewTupleType = is_ViewTuple<std::remove_cvref_t<T>>::value;

//
// Tuple concepts
//
template <typename>
struct is_Tuple : std::false_type {
};

template <typename... Args>
struct is_Tuple<Tuple<Args...>> : std::true_type {
};

template <typename T>
concept TupleType = is_Tuple<std::remove_cvref_t<T>>::value;

//
// "Owning" Tuples are those that have the datastructure associated with the view
// In the current implementation, Owning Tuples always inherit from ContainerTuple
//
namespace detail
{
template <typename>
struct has_ContainerTuple : std::false_type {
};

template <template <typename...> typename T, typename... Args>
struct has_ContainerTuple<T<Args...>>
    : std::is_base_of<ContainerTuple<Args...>, T<Args...>> {
};
} // namespace detail

template <typename T>
concept OwningTuple = detail::has_ContainerTuple<std::remove_cvref_t<T>>::value;

//
// The concepts are supposed to work with all manner of the tuples in the code (i.e.
// ViewTuples, ContainerTuples, Tuples, std::tuple, ...).  As a generic test for a tuple
// we simply check that `get` works
//
template <typename T>
concept TupleLike = requires(T t)
{
    get<0>(t);
};

//
// Define constrained template metafunctions to easily work with mp11
//
namespace detail
{
template <typename>
struct is_tuple_like_impl : std::false_type {
};

template <TupleLike T>
struct is_tuple_like_impl<T> : std::true_type {
};
} // namespace detail
template <typename T>
using is_tuple_like = detail::is_tuple_like_impl<T>::type;

template <typename T>
concept NonOwningTuple = TupleLike<T> && (!OwningTuple<T>);

template <typename T>
concept NonTupleRange = rs::input_range<T> && (!TupleLike<T>);

//
// Will need index_sequences for the tuple indices
//
template <TupleLike T>
using Seq = std::make_index_sequence<mp_size<std::remove_cvref_t<T>>::value>;

//
// The Tuples template arguments are non-req qualified versions of the types that
// result from using `get` to extract different pieces.  This maps the Tuple to an
// mp_list of the types produced by applying `get` to access every element
//
namespace detail
{
template <typename T, typename I>
struct tuple_get_types_impl;
template <typename T, auto... Is>
struct tuple_get_types_impl<T, std::index_sequence<Is...>> {
    using type = traits::mp_list<decltype(get<Is>(std::declval<T>()))...>;
};
} // namespace detail

template <TupleLike T>
using tuple_get_types = detail::tuple_get_types_impl<T, Seq<T>>::type;

//
// concept for nested tuples.  Will allow for more intuitive mapping and for_each
//
template <typename T>
concept NestedTupleLike = TupleLike<T>&&
    mp_apply<mp_all, mp_transform<is_tuple_like, tuple_get_types<T>>>::value;

//
// need to determine the shape of tuples to make sure we only appy the mp11 function
// on similarly shaped tuples to prevent disasters
//
namespace detail
{
template <typename>
struct tuple_shape_impl;
}
template <TupleLike T>
using tuple_shape = typename detail::tuple_shape_impl<std::remove_cvref_t<T>>::type;

namespace detail
{
template <typename T>
struct tuple_shape_impl {
    using type = mp_list<mp_size<T>>;
};

template <NestedTupleLike T>
struct tuple_shape_impl<T> {
    using type = mp_transform<tuple_shape, mp_apply<mp_list, T>>;
};
} // namespace detail

template <typename T, typename U>
concept SimilarTuples =
    TupleLike<T>&& TupleLike<U>&& std::same_as<tuple_shape<T>, tuple_shape<U>>;

//
// Range concepts
//
template <typename T>
concept Range = rs::input_range<T> && (!std::same_as<int3, std::remove_cvref_t<T>>);

template <typename T>
concept AnyOutputRange = rs::range<T>&& rs::output_range<T, rs::range_value_t<T>>;

//
// Trait for OutputRange.  Used to constrain self-modify math operations.
// These need to be based on type traits so they can be mapped over nested tuples
//

namespace detail
{
template <typename Rout, typename Rin>
struct is_output_range_impl : std::false_type {
};

template <typename Rout, typename Rin>
    requires rs::range<Rout> &&
    ((rs::input_range<Rin> && rs::output_range<Rout, rs::range_value_t<Rin>>) ||
     (rs::output_range<Rout, Rin>)) struct is_output_range_impl<Rout, Rin>
    : std::true_type {
};
} // namespace detail

template <typename Rout, typename Rin>
using is_output_range = detail::is_output_range_impl<Rout, Rin>::type;

template <typename Rout, typename Rin>
concept OutputRange = is_output_range<Rout, Rin>::value;

namespace detail
{
template <typename, typename>
struct is_output_tuple_impl;
}
template <typename Out, typename In>
using is_output_tuple = detail::is_output_tuple_impl<Out, In>::type;

namespace detail
{
template <typename Out, typename In>
struct is_output_tuple_impl {
    using type = is_output_range<Out, In>;
};

template <TupleLike Out, typename In>
struct is_output_tuple_impl<Out, In> {
    using type = mp_flatten<
        mp_transform_q<mp_bind_back<is_output_tuple, In>, tuple_get_types<Out>>>;
};

template <TupleLike Out, TupleLike In>
struct is_output_tuple_impl<Out, In> {
    using type = mp_flatten<
        mp_transform<is_output_tuple, tuple_get_types<Out>, tuple_get_types<In>>>;
};

} // namespace detail

template <typename Out, typename In>
concept OutputTuple = TupleLike<Out>&& mp_apply<mp_all, is_output_tuple<Out, In>>::value;

//
// Check that we canconstruct something from a range needed for output ranges and
// container
//
namespace detail
{
template <typename, typename>
struct constructible_from_range_impl {
    using type = std::false_type;
};
template <typename T, NonTupleRange Arg>
struct constructible_from_range_impl<T, Arg> {
    using C = decltype(vs::common(std::declval<Arg>()));
    using type = std::is_constructible<T, rs::iterator_t<C>, rs::sentinel_t<C>>;
};

template <typename T, NonTupleRange Arg>
requires rs::common_range<Arg> struct constructible_from_range_impl<T, Arg> {
    using type = std::is_constructible<T, rs::iterator_t<Arg>, rs::sentinel_t<Arg>>;
};
} // namespace detail

template <typename T, typename Arg>
using is_constructible_from_range = detail::constructible_from_range_impl<T, Arg>::type;

template <typename T, typename Arg>
concept ConstructibleFromRange = is_constructible_from_range<T, Arg>::value;

//
// Combine all supported construction methods into a single trait
//
template <typename T, typename Arg>
using is_constructible_from =
    mp_or<is_constructible_from_range<T, Arg>, std::is_constructible<T, Arg>>;

//
// Traits for nested construction
//
namespace detail
{
template <typename, typename>
struct is_tuple_constructible_from_impl;
}
template <typename T, typename Arg>
using is_tuple_constructible_from =
    detail::is_tuple_constructible_from_impl<T, Arg>::type;

namespace detail
{
template <typename T, typename Arg>
struct is_tuple_constructible_from_impl {
    using type = is_constructible_from<T, Arg>;
};

// Note that we can't use tuple_get_types here for T since that adds references via get
// which change the constructibility of the components of T
template <TupleLike T, TupleLike Arg>
requires(mp_size<std::remove_cvref_t<T>>::value ==
         mp_size<std::remove_cvref_t<Arg>>::
             value) struct is_tuple_constructible_from_impl<T, Arg> {
    using type = mp_flatten<mp_transform<is_tuple_constructible_from,
                                         // tuple_get_types<T>,
                                         mp_rename<std::remove_reference_t<T>, mp_list>,
                                         tuple_get_types<Arg>>>;
};

// template <typename, typename>
// constexpr bool direct_construct_v = false;

// template <template <typename...> typename To,
//           template <typename...>
//           typename From,
//           typename... Ts,
//           typename... Fs>
// requires(sizeof...(Ts) ==
//          sizeof...(Fs)) constexpr bool direct_construct_v<From<Fs...>, To<Ts...>> =
//     (std::constructible_from<Ts, Fs> && ...);

// template <typename, typename>
// constexpr bool from_range_v = false;

// template <template <typename...> typename To,
//           template <typename...>
//           typename From,
//           typename... Ts,
//           typename... Fs>
// requires(sizeof...(Ts) ==
//          sizeof...(Fs)) constexpr bool from_range_v<From<Fs...>, To<Ts...>> =
//     (ConstructibleFromRange<Fs, Ts> && ...);

} // namespace detail

template <typename T, typename Arg>
concept TupleFromTuple =
    SimilarTuples<T, Arg>&& mp_apply<mp_all, is_tuple_constructible_from<T, Arg>>::value;

template <typename Arg, typename T>
concept TupleToTuple = TupleFromTuple<T, Arg>;

// template <typename From, typename To>
// concept FromTupleDirect = TupleLike<To>&& TupleLike<From>&&
//     detail::direct_construct_v<std::remove_cvref_t<From>, To>;

// template <typename From, typename To>
// concept FromTupleRange =
//     TupleLike<To>&& TupleLike<From>&& detail::from_range_v<std::remove_cvref_t<From>,
//     To>;

// // Needed to simplify construction of ContainerTuples
// template <typename From, typename To>
// concept FromTuple = FromTupleDirect<From, To> || FromTupleRange<From, To>;

// traits for functions on Tuples

namespace detail
{

template <typename>
struct tup_index_sequence;

template <template <typename...> typename T, typename... Args>
struct tup_index_sequence<T<Args...>> {
    using type = std::make_index_sequence<sizeof...(Args)>;
};
} // namespace detail

} // namespace traits

template <traits::TupleLike T>
using IndexSeq =
    typename traits::detail::tup_index_sequence<std::remove_cvref_t<T>>::type;

namespace traits
{
//
// traits for invoking functions over tuples
//
namespace detail
{
template <typename, typename...>
struct is_nested_invocable_impl;
}
template <typename F, typename... T>
using is_nested_invocable = detail::is_nested_invocable_impl<F, T...>::type;

namespace detail
{
template <typename F, typename... Args>
struct is_nested_invocable_impl {
    using type = std::is_invocable<F, Args...>;
};

template <typename F, TupleLike... Args>
struct is_nested_invocable_impl<F, Args...> {
    using type = mp_flatten<
        mp_transform_q<mp_bind_front<is_nested_invocable, F>, tuple_get_types<Args>...>>;
};
} // namespace detail

//
// NestedInvocableOver tests to see if we can recursively drill down to the non-tuple
// elements of a nested tuple and apply a given function
//
template <typename F, typename... T>
concept NestedInvocableOver =
    (NestedTupleLike<T> && ...) && mp_apply<mp_all, is_nested_invocable<F, T...>>::value;

//
// Invokable Over tries to apply the Function to each element of the Tuples.
// preference is given to NestedInvokable
//
template <typename F, typename... T>
concept InvocableOver =
    (!NestedInvocableOver<F, T...>)&&(!TupleLike<F>)&&(TupleLike<T>&&...) &&
    mp_same<IndexSeq<T>...>::value&& mp_apply<
        mp_all,
        mp_transform_q<mp_bind_front<std::is_invocable, F>,
                       tuple_get_types<T>...>>::value;

//
// It can be convenient to invoke a Function whose first argument is an mp_size_t<I>
// Currently we don't have a use for a nested version
//
template <typename F, typename... T>
concept IndexedInvocableOver =
    (!TupleLike<F>)&&(!InvocableOver<F, T...>)&&(TupleLike<T>&&...) &&
    mp_same<IndexSeq<T>...>::value&& mp_apply<
        mp_all,
        mp_transform_q<mp_bind_front<std::is_invocable, F>,
                       mp_iota<mp_size<mp_front<mp_list<std::remove_cvref_t<T>...>>>>,
                       tuple_get_types<T>...>>::value;

//
// Invoke a Tuple-of-functions over a tuple
//
template <typename F, typename... T>
concept TupleInvocableOver = TupleLike<F> && (TupleLike<T> && ...) &&
                             mp_same<IndexSeq<F>, IndexSeq<T>...>::value&& mp_apply<
                                 mp_all,
                                 mp_transform_q<mp_bind_front<std::is_invocable>,
                                                tuple_get_types<F>,
                                                tuple_get_types<T>...>>::value;

namespace detail
{
template <typename, typename>
struct is_pipeable_impl : std::false_type {
};

template <typename F, typename T>
requires requires(F f, T t)
{
    t | f;
}
struct is_pipeable_impl<F, T> : std::true_type {
};
} // namespace detail

template <typename F, typename T>
using is_pipeable = detail::is_pipeable_impl<F, T>::type;

//
// traits for nested pipeables
//
namespace detail
{
template <typename, typename>
struct is_nested_pipeable_impl;
}

template <typename F, typename T>
using is_nested_pipeable = detail::is_nested_pipeable_impl<F, T>::type;

namespace detail
{
template <typename F, typename T>
struct is_nested_pipeable_impl {
    using type = is_pipeable<F, T>;
};

template <typename F, TupleLike T>
struct is_nested_pipeable_impl<F, T> {
    using type = mp_flatten<
        mp_transform_q<mp_bind_front<is_nested_pipeable, F>, tuple_get_types<T>>>;
};
} // namespace detail

//
// PipeableOver tests for applying operator| to each element of a potentially nested
// Tuple. Since the constrained algorithms simply call transform there is no need to make
// a distinction here
//
template <typename F, typename T>
concept PipeableOver = TupleLike<T>&& mp_apply<mp_all, is_nested_pipeable<F, T>>::value;

template <typename F, typename T>
concept TuplePipeableOver = TupleLike<T>&& TupleLike<F>&& mp_same<Seq<F>, Seq<T>>::value&&
    mp_apply<mp_all,
             mp_transform_q<mp_bind_front<is_pipeable>,
                            tuple_get_types<F>,
                            tuple_get_types<T>>>::value;
} // namespace traits

template <typename...>
struct from_view {
};

// ViewTuple is mainly used for testing
template <traits::ViewTupleType U>
struct from_view<U> {
    static constexpr auto create = [](auto&&, auto&&... args) {
        return ViewTuple{FWD(args)...};
    };
};

// ViewTuple is mainly used for testing
template <traits::ViewTupleType U, traits::ViewTupleType V>
struct from_view<U, V> {
    static constexpr auto create = [](auto&&, auto&&, auto&&... args) {
        return ViewTuple{FWD(args)...};
    };
};

// single argument views
template <traits::TupleType U>
struct from_view<U> {
    static constexpr auto create = [](auto&&, auto&&... args) {
        return Tuple{tag{}, FWD(args)...};
    };
};

// Combination views
template <traits::TupleType U, traits::TupleType V>
struct from_view<U, V> {
    static constexpr auto create = [](auto&&, auto&&, auto&&... args) {
        return Tuple{tag{}, FWD(args)...};
    };
};

template <typename... T>
static constexpr auto create_from_view = from_view<std::remove_cvref_t<T>...>::create;

template <typename... T>
concept From_View = requires
{
    from_view<std::remove_cvref_t<T>...>::create;
};

} // namespace ccs::field::tuple

namespace ccs::field
{
using tuple::Tuple;
}

// need to specialize this bool inorder for r_tuples to have the correct behavior.
// This is somewhat tricky and is hopefully tested by all the "concepts" tests in
// r_tuple.t.cpp
namespace ranges
{
template <typename... Args>
inline constexpr bool enable_view<ccs::field::Tuple<Args...>> = false;

template <ccs::field::tuple::All T>
inline constexpr bool enable_view<ccs::field::Tuple<T>> = true;
} // namespace ranges
