#pragma once

#include "types.hpp"
#include <concepts>
#include <range/v3/range/concepts.hpp>
#include <range/v3/view/view.hpp>

#include <boost/mp11.hpp>

// Can we leverage something like MP11 to rewrite
// most of these traits?

// Can we use mp11's mapping capabilities

namespace ccs::field::tuple
{

using std::get;

template <int I, typename R>
constexpr auto view(R&&);

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

template <typename T>
concept Range = rs::input_range<T> && (!std::same_as<int3, std::remove_cvref_t<T>>);

template <typename T>
concept AnyOutputRange = rs::range<T>&& rs::output_range<T, rs::range_value_t<T>>;

template <typename Rout, typename Rin>
concept OutputRange = rs::range<Rout> &&
                      ((rs::input_range<Rin> &&
                        rs::output_range<Rout, rs::range_value_t<Rin>>) ||
                       (rs::output_range<Rout, Rin>));
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

template <typename T>
concept NonOwningTuple = TupleLike<T> && (!OwningTuple<T>);

template <typename T>
concept NonTupleRange = rs::input_range<T> && (!TupleLike<T>);

//
// Check that we canconstruct something from a range needed for output ranges and
// container
//
template <typename From, typename To>
concept FromRange = ranges::range<From>&& requires(From& r)
{
    To{rs::begin(r), rs::end(r)};
};

namespace detail
{

template <typename, typename>
constexpr bool OutputTuple_v = false;

template <template <typename...> typename Out, typename In, typename... Os>
requires(!TupleLike<In>) constexpr bool OutputTuple_v<Out<Os...>, In> =
    (OutputRange<Os, In> && ...);

template <template <typename...> typename Out,
          template <typename...>
          typename In,
          typename... Os,
          typename... Is>
requires(TupleLike<In<Is...>> &&
         sizeof...(Is) ==
             sizeof...(Os)) constexpr bool OutputTuple_v<Out<Os...>, In<Is...>> =
    (OutputRange<Os, Is> && ...);

} // namespace detail

template <typename Out, typename In>
concept OutputTuple = TupleLike<Out>&&
    detail::OutputTuple_v<std::remove_reference_t<Out>, std::remove_cvref_t<In>>;

namespace detail
{

template <typename, typename>
constexpr bool direct_construct_v = false;

template <template <typename...> typename To,
          template <typename...>
          typename From,
          typename... Ts,
          typename... Fs>
requires(sizeof...(Ts) ==
         sizeof...(Fs)) constexpr bool direct_construct_v<From<Fs...>, To<Ts...>> =
    (std::constructible_from<Ts, Fs> && ...);

template <typename, typename>
constexpr bool from_range_v = false;

template <template <typename...> typename To,
          template <typename...>
          typename From,
          typename... Ts,
          typename... Fs>
requires(sizeof...(Ts) ==
         sizeof...(Fs)) constexpr bool from_range_v<From<Fs...>, To<Ts...>> =
    (FromRange<Fs, Ts> && ...);

} // namespace detail

template <typename From, typename To>
concept FromTupleDirect = TupleLike<To>&& TupleLike<From>&&
    detail::direct_construct_v<std::remove_cvref_t<From>, To>;

template <typename From, typename To>
concept FromTupleRange =
    TupleLike<To>&& TupleLike<From>&& detail::from_range_v<std::remove_cvref_t<From>, To>;

// Needed to simplify construction of ContainerTuples
template <typename From, typename To>
concept FromTuple = FromTupleDirect<From, To> || FromTupleRange<From, To>;

// traits for functions on Tuples

template <auto I, typename F, typename... Args>
concept TemplateInvocable = requires(F f, Args... args)
{
    f.template operator()<I>(args...);
};

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

template <typename F, typename... T>
concept InvocableOver = (!TupleLike<F>)&&(TupleLike<T>&&...) &&
                        mp_same<IndexSeq<T>...>::value&& mp_apply<
                            mp_all,
                            mp_transform_q<mp_bind_front<std::is_invocable, F>,
                                           std::remove_cvref_t<T>...>>::value;

template <typename F, typename... T>
concept IndexedInvocableOver =
    (!TupleLike<F>)&&(!InvocableOver<F, T...>)&&(TupleLike<T>&&...) &&
    mp_same<IndexSeq<T>...>::value&& mp_apply<
        mp_all,
        mp_transform_q<mp_bind_front<std::is_invocable, F>,
                       mp_iota<mp_size<mp_front<mp_list<std::remove_cvref_t<T>...>>>>,
                       std::remove_cvref_t<T>...>>::value;

template <typename F, typename... T>
concept TupleInvocableOver = TupleLike<F> && (TupleLike<T> && ...) &&
                             mp_same<IndexSeq<F>, IndexSeq<T>...>::value&& mp_apply<
                                 mp_all,
                                 mp_transform_q<mp_bind_front<std::is_invocable>,
                                                std::remove_cvref_t<F>,
                                                std::remove_cvref_t<T>...>>::value;

namespace detail
{
// concepts for view closures
template <auto I, typename F, typename T>
concept pipeable_element = requires(F f, T t)
{
    get<I>(t) | f;
};

// get based
template <typename...>
constexpr bool pipeable_over_v = false;

template <typename ViewFn, typename T, auto... Is>
constexpr bool pipeable_over_v<vs::view_closure<ViewFn>, T, std::index_sequence<Is...>> =
    (pipeable_element<Is, vs::view_closure<ViewFn>, T> && ...);

template <typename...>
constexpr bool tuple_pipeable_over_v = false;

template <template <typename...> typename F, typename... Fs, typename T, auto... Is>
requires(sizeof...(Fs) ==
         sizeof...(Is)) constexpr bool tuple_pipeable_over_v<F<vs::view_closure<Fs>...>,
                                                             T,
                                                             std::index_sequence<Is...>> =
    (pipeable_element<Is, vs::view_closure<Fs>, T> && ...);
} // namespace detail

template <typename F, typename T>
concept PipeableOver = TupleLike<T>&& detail::pipeable_over_v<F, T, IndexSeq<T>>;

template <typename F, typename T>
concept TuplePipeableOver = TupleLike<T>&& TupleLike<F>&&
    detail::tuple_pipeable_over_v<std::remove_cvref_t<F>, T, IndexSeq<T>>;

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
