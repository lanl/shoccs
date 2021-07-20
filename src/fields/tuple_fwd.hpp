#pragma once

#include "types.hpp"

#include <range/v3/range/concepts.hpp>
#include <range/v3/view/common.hpp>
#include <range/v3/view/view.hpp>

#include <boost/mp11.hpp>
#include <boost/type_traits/copy_cv_ref.hpp>

namespace ccs //::field::tuple
{

using std::get;
using namespace boost::mp11;

// Concept for being able to apply vs::all.  Need to exclude int3 as
// an "Allable" range to trigger proper tuple construction calls
template <typename T>
concept All = rs::range<T&> && rs::viewable_range<T> &&
    (!std::same_as<int3, std::remove_cvref_t<T>>);

// Forward decls for main types
template <typename...>
struct tuple;

// forward decls for component types
template <typename...>
struct container_tuple;

template <typename...>
struct view_tuple_base;

template <typename...>
struct view_tuple;

//
// container_tuple concepts
//
template <typename>
struct is_container_tuple : std::false_type {
};

template <typename... Args>
struct is_container_tuple<container_tuple<Args...>> : std::true_type {
};

template <typename T>
concept ContainerTuple = is_container_tuple<std::remove_cvref_t<T>>::value;

//
// ViewBase/View Tuples
//
template <typename>
struct is_view_tuple_base : std::false_type {
};

template <typename... Args>
struct is_view_tuple_base<view_tuple_base<Args...>> : std::true_type {
};

template <typename T>
concept ViewTupleBase = is_view_tuple_base<std::remove_cvref_t<T>>::value;

template <typename>
struct is_view_tuple : std::false_type {
};

template <typename... Args>
struct is_view_tuple<view_tuple<Args...>> : std::true_type {
};

template <typename T>
concept ViewTuple = is_view_tuple<std::remove_cvref_t<T>>::value;

//
// Tuple concepts
//
template <typename>
struct is_tuple : std::false_type {
};

template <typename... Args>
struct is_tuple<tuple<Args...>> : std::true_type {
};

template <typename T>
concept Tuple = is_tuple<std::remove_cvref_t<T>>::value;

//
// "Owning" Tuples are those that have the datastructure associated with the view
// In the current implementation, Owning Tuples always inherit from container_tuple
//
namespace detail
{
    template <typename>
    struct has_container_tuple : std::false_type {
    };

    template <template <typename...> typename T, typename... Args>
    struct has_container_tuple<T<Args...>>
        : std::is_base_of<container_tuple<Args...>, T<Args...>> {
    };
} // namespace detail

template <typename T>
using is_owning_tuple = detail::has_container_tuple<std::remove_cvref_t<T>>::type;

template <typename T>
concept OwningTuple = detail::has_container_tuple<std::remove_cvref_t<T>>::value;

//
// The concepts are supposed to work with all manner of the tuples in the code (i.e.
// ViewTuples, ContainerTuples, Tuples, std::tuple, ...).  Some of the range-v3 ranges
// result in custom range tuples so we have to be strict about what we call tuple_like
//
namespace detail
{
    template <typename T>
    struct is_tuple_like_impl : std::false_type {
    };

    template <typename... Args>
    struct is_tuple_like_impl<std::tuple<Args...>> : std::true_type {
    };

    template <typename... Args>
    struct is_tuple_like_impl<rs::common_tuple<Args...>> : std::true_type {
    };

    template <template <typename...> typename T, typename... Args>
    struct is_tuple_like_impl<T<Args...>>
        : mp_or<std::is_base_of<container_tuple<Args...>, T<Args...>>,
                std::is_base_of<view_tuple_base<Args...>, T<Args...>>> {
    };
} // namespace detail
template <typename T>
using is_tuple_like = detail::is_tuple_like_impl<std::remove_cvref_t<T>>::type;

template <typename T>
concept TupleLike = is_tuple_like<T>::value;

namespace detail
{
    template <typename>
    struct is_stdarray_impl : std::false_type {
    };
    template <typename T, auto N>
    struct is_stdarray_impl<std::array<T, N>> : std::true_type {
    };
} // namespace detail
template <typename T>
using is_stdarray = detail::is_stdarray_impl<std::remove_cvref_t<T>>::type;

template <typename T>
concept NumericTuple = is_stdarray<T>::value ||
    (TupleLike<T>&&
         mp_apply<mp_all,
                  mp_transform_q<mp_compose<std::remove_cvref_t, std::is_arithmetic>,
                                 mp_rename<std::remove_reference_t<T>, mp_list>>>::value);

template <typename T>
concept Non_Owning_Tuple = TupleLike<T> &&(!OwningTuple<T>);

template <typename T>
concept NonTupleRange = rs::input_range<T> &&(!TupleLike<T>);

//
// Determine the nesting level of the tuples.  Will be needed for smart selections
//
namespace detail
{
template <typename T, typename V>
struct tuple_levels_impl;
}
template <typename T, typename V = mp_size_t<0>>
using tuple_levels = detail::tuple_levels_impl<T, V>::type;

namespace detail
{
template <typename T, typename V>
struct tuple_levels_impl {
    using type = V;
};

template <TupleLike T, auto I>
struct tuple_levels_impl<T, mp_size_t<I>> {
    using type = tuple_levels<mp_first<std::remove_cvref_t<T>>, mp_size_t<I + 1>>;
};
} // namespace detail
template <typename T>
constexpr auto tuple_levels_v = tuple_levels<T>::value;

//
// Will need index_sequences for the tuple indices
//
template <typename T>
using tuple_seq = std::make_index_sequence<mp_size<std::remove_cvref_t<T>>::value>;

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
    using type = mp_list<decltype(get<Is>(std::declval<T>()))...>;
};
} // namespace detail

template <TupleLike T>
using tuple_get_types = detail::tuple_get_types_impl<T, tuple_seq<T>>::type;

//
// concept for nested tuples.  Will allow for more intuitive mapping and for_each
//
template <typename T>
concept NestedTuple = TupleLike<T> &&
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

template <NestedTuple T>
struct tuple_shape_impl<T> {
    using type = mp_transform<tuple_shape, mp_apply<mp_list, T>>;
};
} // namespace detail

template <typename T, typename U>
concept SimilarTuples =
    TupleLike<T> && TupleLike<U> && std::same_as<tuple_shape<T>, tuple_shape<U>>;

//
// Range concepts
//
template <typename T>
concept Range = rs::input_range<T> &&(!std::same_as<int3, std::remove_cvref_t<T>>);

// template <typename T>
// concept AnyOutputRange = rs::range<T>&& rs::output_range<T, rs::range_value_t<T>>;

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
        ((rs::range<Rin> && rs::output_range<Rout, rs::range_value_t<Rin>>) ||
         (rs::output_range<Rout, Rin>)) struct is_output_range_impl<Rout, Rin>
        : std::true_type {
    };
} // namespace detail

template <typename Rout, typename Rin = Rout>
using is_output_range = detail::is_output_range_impl<Rout, Rin>::type;

template <typename Rout, typename Rin = Rout>
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

template <TupleLike Out, SimilarTuples<Out> In>
struct is_output_tuple_impl<Out, In> {
    using type = mp_flatten<
        mp_transform<is_output_tuple, tuple_get_types<Out>, tuple_get_types<In>>>;
};

} // namespace detail

template <typename Out, typename In>
concept OutputTuple = TupleLike<Out> && mp_apply<mp_all, is_output_tuple<Out, In>>::value;

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
        requires rs::common_range<Arg>
    struct constructible_from_range_impl<T, Arg> {
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
             mp_size<std::remove_cvref_t<Arg>>::value)
struct is_tuple_constructible_from_impl<T, Arg> {
    using type = mp_flatten<mp_transform<is_tuple_constructible_from,
                                         // tuple_get_types<T>,
                                         mp_rename<std::remove_reference_t<T>, mp_list>,
                                         tuple_get_types<Arg>>>;
};

} // namespace detail

template <typename T, typename Arg>
concept TupleFromTuple =
    SimilarTuples<T, Arg> && mp_apply<mp_all, is_tuple_constructible_from<T, Arg>>::value;

template <typename Arg, typename T>
concept TupleToTuple = TupleFromTuple<T, Arg>;

template <typename T, typename Arg>
concept ArrayFromTuple = is_stdarray<T>::value && TupleLike<Arg> &&
    mp_apply<mp_all,
             mp_transform_q<mp_bind_front<std::is_constructible, typename T::value_type>,
                            tuple_get_types<Arg>>>::value;

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
    (NestedTuple<T> && ...) && mp_apply<mp_all, is_nested_invocable<F, T...>>::value;

//
// Invokable Over tries to apply the Function to each element of the Tuples.
// preference is given to NestedInvokable
//
template <typename F, typename... T>
concept InvocableOver =
    (!NestedInvocableOver<F, T...>)&&(!TupleLike<F>)&&(TupleLike<T>&&...) &&
    mp_same<tuple_seq<T>...>::value&& mp_apply<
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
    mp_same<tuple_seq<T>...>::value&& mp_apply<
        mp_all,
        mp_transform_q<mp_bind_front<std::is_invocable, F>,
                       mp_iota<mp_size<mp_front<mp_list<std::remove_cvref_t<T>...>>>>,
                       tuple_get_types<T>...>>::value;

//
// Invoke a Tuple-of-functions over a tuple
//
template <typename F, typename... T>
concept TupleInvocableOver = TupleLike<F> &&(TupleLike<T>&&...) &&
                             mp_same<tuple_seq<F>, tuple_seq<T>...>::value&& mp_apply<
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
    requires requires(F f, T t) { t | f; }
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
concept PipeableOver = TupleLike<T> && mp_apply<mp_all, is_nested_pipeable<F, T>>::value;

template <typename F, typename T>
concept TuplePipeableOver =
    TupleLike<T> && TupleLike<F> && mp_same<tuple_seq<F>, tuple_seq<T>>::value &&
    mp_apply<mp_all,
             mp_transform_q<mp_bind_front<is_pipeable>,
                            tuple_get_types<F>,
                            tuple_get_types<T>>>::value;

//
// Traits for view_closures and Tuples of view_closures
//
namespace detail
{
    template <typename T>
    struct is_view_closure_impl : std::false_type {
    };

    template <typename Fn>
    struct is_view_closure_impl<vs::view_closure<Fn>> : std::true_type {
    };
} // namespace detail

template <typename T>
using is_view_closure = detail::is_view_closure_impl<std::remove_cvref_t<T>>::type;

namespace detail
{
template <typename>
struct is_nested_view_closure_impl;
}
template <typename T>
using is_nested_view_closure =
    detail::is_nested_view_closure_impl<std::remove_cvref_t<T>>::type;

namespace detail
{
template <typename T>
struct is_nested_view_closure_impl {
    using type = mp_list<is_view_closure<T>>;
};

template <TupleLike T>
struct is_nested_view_closure_impl<T> {
    using type = mp_flatten<mp_transform<is_nested_view_closure,
                                         mp_rename<std::remove_reference_t<T>, mp_list>>>;
};

} // namespace detail

template <typename T>
concept ViewClosure = is_view_closure<T>::value;

template <typename T>
concept ViewClosures = mp_apply<mp_all, is_nested_view_closure<T>>::value;

//
// Traits for ref_views - needed to implement intuitive assignment for ViewTuples
//
namespace detail
{
    template <typename>
    struct is_ref_view_impl : std::false_type {
    };

    template <typename Rng>
    struct is_ref_view_impl<rs::ref_view<Rng>> : std::true_type {
    };
} // namespace detail

template <typename R>
using is_ref_view = detail::is_ref_view_impl<std::remove_cvref_t<R>>::type;

template <typename R>
concept RefView = is_ref_view<R>::value;

//
// traits for being able to assign the components of a tuple
//
template <typename T, typename U>
concept AssignableRefView = TupleLike<T> &&
    mp_apply<mp_all, mp_transform<is_ref_view, tuple_get_types<T>>>::value &&
    (!TupleLike<U>)&&mp_apply<
        mp_all,
        mp_transform_q<mp_bind_back<std::is_assignable, U>,
                       mp_rename<std::remove_reference_t<T>, mp_list>>>::value;

template <typename T, typename U>
concept AssignableDirect = TupleLike<T> &&(!TupleLike<U>)&&mp_apply<
    mp_all,
    mp_transform_q<mp_bind_back<std::is_assignable, U>, tuple_get_types<T>>>::value;

template <typename T, typename U>
concept AssignableDirectTuple = TupleLike<T> && SimilarTuples<T, U> && mp_apply<
    mp_all,
    mp_transform<std::is_assignable, tuple_get_types<T>, tuple_get_types<U>>>::value;

//
// type functions for construct list of types for get based access.  Facilitates
// selection logic
//
template <auto... Is>
using list_index = mp_list<mp_size_t<Is>...>;

namespace detail
{
template <typename>
struct is_list_index_impl : std::false_type {
};

template <auto... Is>
struct is_list_index_impl<list_index<Is...>> : std::true_type {
};
} // namespace detail

template <typename T>
using is_list_index = detail::is_list_index_impl<T>::type;

template <typename T>
concept ListIndex = is_list_index<T>::value;

namespace detail
{
    template <typename>
    struct is_list_indices : std::false_type {
    };

    template <typename... T>
    struct is_list_indices<mp_list<T...>> {
        using type = mp_apply<mp_all, mp_transform<is_list_index, mp_list<T...>>>;
    };

} // namespace detail
template <typename T>
using is_list_indices = detail::is_list_indices<T>::type;

template <typename T>
concept ListIndices = is_list_indices<T>::value;

template <ListIndex L, std::size_t I>
constexpr auto index_v = mp_at_c<L, I>::value;

namespace detail
{
template <typename T>
struct viewable_range_by_value_impl {
    using type = T;
};

template <All T>
struct viewable_range_by_value_impl<T&> {
    using type = T;
};
} // namespace detail

template <typename T>
using viewable_range_by_value = detail::viewable_range_by_value_impl<T>::type;

namespace detail
{
template <typename>
struct underlying_range_impl;
}

template <typename T>
using underlying_range_t = detail::underlying_range_impl<T>::type;

namespace detail
{
template <typename>
struct underlying_range_impl {
    static_assert("No underlying range type for tuple");
};

template <Range R>
struct underlying_range_impl<R> {
    using type = std::remove_cvref_t<R>;
};

template <TupleLike T>
    requires(!Range<T>)
struct underlying_range_impl<T> {
    using type = underlying_range_t<std::remove_cvref_t<mp_first<tuple_get_types<T>>>>;
};
} // namespace detail

} // namespace ccs

// need to specialize this bool inorder for r_tuples to have the correct behavior.
// This is somewhat tricky and is hopefully tested by all the "concepts" tests in
// r_tuple.t.cpp
namespace ranges
{
template <typename... Args>
inline constexpr bool enable_view<ccs::tuple<Args...>> = false;

template <ccs::All T>
inline constexpr bool enable_view<ccs::tuple<T>> = true;

template <ccs::All T>
inline constexpr bool enable_view<ccs::view_tuple<T>> = true;

} // namespace ranges
