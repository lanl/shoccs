#pragma once

#include <cstddef>
#include <iterator>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// ===========================================================================
// basic_common_reference specialization for std::tuple (C++23 P2321R2 backport)
//
// In C++20, std::common_reference<tuple<const T&,...>&&, tuple<T,...>&> has no
// type because tuple lacks common_reference specializations. This means
// iterators with reference=tuple<const T&,...> and value_type=tuple<T,...>
// don't satisfy std::indirectly_readable → std::input_iterator. This backport
// enables cartesian_product_view::iterator (and similar) to model standard
// iterator concepts, so std::views::transform etc. can pipe through them.
// ===========================================================================
#if !defined(__cpp_lib_ranges_zip)
namespace std
{
template <typename... Ts,
          typename... Us,
          template <class>
          class TQual,
          template <class>
          class UQual>
    requires(sizeof...(Ts) == sizeof...(Us)) &&
            requires { typename tuple<common_reference_t<TQual<Ts>, UQual<Us>>...>; }
struct basic_common_reference<tuple<Ts...>, tuple<Us...>, TQual, UQual> {
    using type = tuple<common_reference_t<TQual<Ts>, UQual<Us>>...>;
};
} // namespace std
#endif

namespace ccs
{

// ===========================================================================
// repeat_n_view<T>
//
// Lazy view of n copies of value v.
// Models random_access_range and sized_range.
// ===========================================================================
template <typename T>
class repeat_n_view : public std::ranges::view_interface<repeat_n_view<T>>
{
    T value_;
    std::ptrdiff_t count_;

public:
    class iterator
    {
        T value_;
        std::ptrdiff_t pos_;

    public:
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using reference = const T&;
        using iterator_concept = std::random_access_iterator_tag;
        using iterator_category = std::random_access_iterator_tag;

        iterator() = default;
        constexpr iterator(T v, std::ptrdiff_t p) : value_{std::move(v)}, pos_{p} {}

        constexpr reference operator*() const { return value_; }
        constexpr reference operator[](difference_type) const { return value_; }

        constexpr iterator& operator++()
        {
            ++pos_;
            return *this;
        }
        constexpr iterator operator++(int)
        {
            auto tmp = *this;
            ++pos_;
            return tmp;
        }
        constexpr iterator& operator--()
        {
            --pos_;
            return *this;
        }
        constexpr iterator operator--(int)
        {
            auto tmp = *this;
            --pos_;
            return tmp;
        }
        constexpr iterator& operator+=(difference_type n)
        {
            pos_ += n;
            return *this;
        }
        constexpr iterator& operator-=(difference_type n)
        {
            pos_ -= n;
            return *this;
        }

        friend constexpr iterator operator+(iterator it, difference_type n)
        {
            return {it.value_, it.pos_ + n};
        }
        friend constexpr iterator operator+(difference_type n, iterator it)
        {
            return {it.value_, it.pos_ + n};
        }
        friend constexpr iterator operator-(iterator it, difference_type n)
        {
            return {it.value_, it.pos_ - n};
        }
        friend constexpr difference_type operator-(const iterator& a, const iterator& b)
        {
            return a.pos_ - b.pos_;
        }

        friend constexpr bool operator==(const iterator& a, const iterator& b)
        {
            return a.pos_ == b.pos_;
        }
        friend constexpr auto operator<=>(const iterator& a, const iterator& b)
        {
            return a.pos_ <=> b.pos_;
        }
    };

    repeat_n_view() = default;
    constexpr repeat_n_view(T value, std::ptrdiff_t count)
        : value_{std::move(value)}, count_{count}
    {
    }

    constexpr iterator begin() const { return {value_, 0}; }
    constexpr iterator end() const { return {value_, count_}; }
    constexpr std::ptrdiff_t size() const { return count_; }
};

// Factory function
struct repeat_n_fn {
    template <typename T>
    constexpr auto operator()(T value, std::ptrdiff_t n) const
    {
        return repeat_n_view<std::decay_t<T>>(std::move(value), n);
    }
};
inline constexpr repeat_n_fn repeat_n{};

// ===========================================================================
// stride_view<Rng>
//
// Lazy view that yields every n-th element of Rng.
// Models view_interface. Supports random-access if base range does.
// ===========================================================================
template <std::ranges::input_range Rng>
    requires std::ranges::view<Rng>
class stride_view : public std::ranges::view_interface<stride_view<Rng>>
{
    Rng base_;
    std::ranges::range_difference_t<Rng> stride_;

public:
    class iterator
    {
        using base_iter = std::ranges::iterator_t<Rng>;
        using base_sent = std::ranges::sentinel_t<Rng>;
        using diff_t = std::ranges::range_difference_t<Rng>;

        base_iter current_{};
        base_sent end_{};
        diff_t stride_{};
        diff_t missing_{}; // how many steps short of stride we were at end

    public:
        using difference_type = diff_t;
        using value_type = std::ranges::range_value_t<Rng>;
        using reference = std::ranges::range_reference_t<Rng>;
        using iterator_concept = std::conditional_t<
            std::ranges::random_access_range<Rng>,
            std::random_access_iterator_tag,
            std::conditional_t<std::ranges::bidirectional_range<Rng>,
                               std::bidirectional_iterator_tag,
                               std::conditional_t<std::ranges::forward_range<Rng>,
                                                  std::forward_iterator_tag,
                                                  std::input_iterator_tag>>>;
        using iterator_category = iterator_concept;

        iterator() = default;

        constexpr iterator(base_iter current, base_sent end, diff_t stride,
                           diff_t missing = 0)
            : current_{std::move(current)}
            , end_{std::move(end)}
            , stride_{stride}
            , missing_{missing}
        {
        }

        constexpr reference operator*() const { return *current_; }

        constexpr iterator& operator++()
        {
            missing_ = std::ranges::advance(current_, stride_, end_);
            return *this;
        }

        constexpr iterator operator++(int)
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        constexpr iterator& operator--()
            requires std::ranges::bidirectional_range<Rng>
        {
            std::ranges::advance(current_, missing_ - stride_);
            missing_ = 0;
            return *this;
        }

        constexpr iterator operator--(int)
            requires std::ranges::bidirectional_range<Rng>
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        constexpr iterator& operator+=(difference_type n)
            requires std::ranges::random_access_range<Rng>
        {
            if (n > 0) {
                missing_ = std::ranges::advance(current_, stride_ * n, end_);
            } else if (n < 0) {
                std::ranges::advance(current_, stride_ * n + missing_);
                missing_ = 0;
            }
            return *this;
        }

        constexpr iterator& operator-=(difference_type n)
            requires std::ranges::random_access_range<Rng>
        {
            return *this += -n;
        }

        constexpr reference operator[](difference_type n) const
            requires std::ranges::random_access_range<Rng>
        {
            return *(*this + n);
        }

        friend constexpr iterator operator+(iterator it, difference_type n)
            requires std::ranges::random_access_range<Rng>
        {
            it += n;
            return it;
        }

        friend constexpr iterator operator+(difference_type n, iterator it)
            requires std::ranges::random_access_range<Rng>
        {
            it += n;
            return it;
        }

        friend constexpr iterator operator-(iterator it, difference_type n)
            requires std::ranges::random_access_range<Rng>
        {
            it -= n;
            return it;
        }

        friend constexpr difference_type operator-(const iterator& a, const iterator& b)
            requires std::ranges::random_access_range<Rng>
        {
            auto dist = a.current_ - b.current_;
            if (dist > 0)
                return (dist + a.missing_) / a.stride_;
            else if (dist < 0)
                return -((-dist + b.missing_) / b.stride_);
            else
                return 0;
        }

        friend constexpr bool operator==(const iterator& a, const iterator& b)
        {
            return a.current_ == b.current_;
        }

        friend constexpr auto operator<=>(const iterator& a, const iterator& b)
            requires std::ranges::random_access_range<Rng>
        {
            return a.current_ <=> b.current_;
        }
    };

    stride_view() = default;

    constexpr stride_view(Rng base, std::ranges::range_difference_t<Rng> stride)
        : base_{std::move(base)}, stride_{stride}
    {
    }

    constexpr auto begin()
    {
        return iterator{std::ranges::begin(base_), std::ranges::end(base_), stride_};
    }

    constexpr auto begin() const
        requires std::ranges::range<const Rng>
    {
        return iterator{std::ranges::begin(base_), std::ranges::end(base_), stride_};
    }

    constexpr auto end()
    {
        if constexpr (std::ranges::sized_range<Rng>) {
            auto sz = std::ranges::size(base_);
            auto d = static_cast<std::ranges::range_difference_t<Rng>>(sz);
            auto missing = (stride_ - d % stride_) % stride_;
            auto end_it = std::ranges::end(base_);
            return iterator{std::move(end_it),
                            std::ranges::end(base_),
                            stride_,
                            missing};
        } else {
            auto end_it = std::ranges::end(base_);
            return iterator{std::move(end_it), std::ranges::end(base_), stride_};
        }
    }

    constexpr auto end() const
        requires std::ranges::range<const Rng>
    {
        if constexpr (std::ranges::sized_range<const Rng>) {
            auto sz = std::ranges::size(base_);
            auto d = static_cast<std::ranges::range_difference_t<const Rng>>(sz);
            auto missing = (stride_ - d % stride_) % stride_;
            auto end_it = std::ranges::end(base_);
            return iterator{std::move(end_it),
                            std::ranges::end(base_),
                            stride_,
                            missing};
        } else {
            auto end_it = std::ranges::end(base_);
            return iterator{std::move(end_it), std::ranges::end(base_), stride_};
        }
    }

    constexpr auto size()
        requires std::ranges::sized_range<Rng>
    {
        auto sz = std::ranges::size(base_);
        auto d = static_cast<std::ranges::range_difference_t<Rng>>(sz);
        return (d + stride_ - 1) / stride_;
    }

    constexpr auto size() const
        requires std::ranges::sized_range<const Rng>
    {
        auto sz = std::ranges::size(base_);
        auto d = static_cast<std::ranges::range_difference_t<const Rng>>(sz);
        return (d + stride_ - 1) / stride_;
    }
};

template <typename Rng>
stride_view(Rng&&, std::ranges::range_difference_t<Rng>) -> stride_view<std::views::all_t<Rng>>;

// Factory function (takes range + stride)
struct stride_fn {
    template <std::ranges::viewable_range Rng>
    constexpr auto operator()(Rng&& rng,
                              std::ranges::range_difference_t<std::remove_cvref_t<Rng>> n) const
    {
        return stride_view(std::views::all(std::forward<Rng>(rng)), n);
    }
};
inline constexpr stride_fn stride{};

// ===========================================================================
// cartesian_product_view<R1, R2, R3>
//
// Lazy view over three ranges yielding
// std::tuple<range_reference_t<R1>, range_reference_t<R2>, range_reference_t<R3>>
// in triple-nested-loop order (R1 slowest, R3 fastest).
// Models forward_range and sized_range (if all bases are sized).
// ===========================================================================
template <std::ranges::forward_range R1,
          std::ranges::forward_range R2,
          std::ranges::forward_range R3>
    requires(std::ranges::view<R1> && std::ranges::view<R2> && std::ranges::view<R3>)
class cartesian_product_view
    : public std::ranges::view_interface<cartesian_product_view<R1, R2, R3>>
{
    R1 r1_;
    R2 r2_;
    R3 r3_;

public:
    class iterator
    {
        std::ranges::iterator_t<R1> it1_{};
        std::ranges::iterator_t<R2> it2_{};
        std::ranges::iterator_t<R3> it3_{};
        std::ranges::iterator_t<R2> begin2_{};
        std::ranges::sentinel_t<R2> end2_{};
        std::ranges::iterator_t<R3> begin3_{};
        std::ranges::sentinel_t<R3> end3_{};

    public:
        using value_type = std::tuple<std::ranges::range_value_t<R1>,
                                      std::ranges::range_value_t<R2>,
                                      std::ranges::range_value_t<R3>>;
        using reference = std::tuple<std::ranges::range_reference_t<R1>,
                                     std::ranges::range_reference_t<R2>,
                                     std::ranges::range_reference_t<R3>>;
        using difference_type = std::ptrdiff_t;
        using iterator_concept = std::forward_iterator_tag;

        iterator() = default;

        constexpr iterator(std::ranges::iterator_t<R1> it1,
                           std::ranges::iterator_t<R2> it2,
                           std::ranges::iterator_t<R3> it3,
                           std::ranges::iterator_t<R2> begin2,
                           std::ranges::sentinel_t<R2> end2,
                           std::ranges::iterator_t<R3> begin3,
                           std::ranges::sentinel_t<R3> end3)
            : it1_{std::move(it1)}
            , it2_{std::move(it2)}
            , it3_{std::move(it3)}
            , begin2_{std::move(begin2)}
            , end2_{std::move(end2)}
            , begin3_{std::move(begin3)}
            , end3_{std::move(end3)}
        {
        }

        constexpr reference operator*() const { return reference{*it1_, *it2_, *it3_}; }

        constexpr iterator& operator++()
        {
            ++it3_;
            if (it3_ == end3_) {
                it3_ = begin3_;
                ++it2_;
                if (it2_ == end2_) {
                    it2_ = begin2_;
                    ++it1_;
                }
            }
            return *this;
        }

        constexpr iterator operator++(int)
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        friend constexpr bool operator==(const iterator& a, const iterator& b)
        {
            return a.it1_ == b.it1_ && a.it2_ == b.it2_ && a.it3_ == b.it3_;
        }
    };

    cartesian_product_view() = default;

    constexpr cartesian_product_view(R1 r1, R2 r2, R3 r3)
        : r1_{std::move(r1)}, r2_{std::move(r2)}, r3_{std::move(r3)}
    {
    }

    constexpr iterator begin()
    {
        if (std::ranges::empty(r1_) || std::ranges::empty(r2_) ||
            std::ranges::empty(r3_))
            return end();
        return iterator{std::ranges::begin(r1_), std::ranges::begin(r2_),
                        std::ranges::begin(r3_), std::ranges::begin(r2_),
                        std::ranges::end(r2_), std::ranges::begin(r3_),
                        std::ranges::end(r3_)};
    }

    constexpr iterator begin() const
    {
        if (std::ranges::empty(r1_) || std::ranges::empty(r2_) ||
            std::ranges::empty(r3_))
            return end();
        return iterator{std::ranges::begin(r1_), std::ranges::begin(r2_),
                        std::ranges::begin(r3_), std::ranges::begin(r2_),
                        std::ranges::end(r2_), std::ranges::begin(r3_),
                        std::ranges::end(r3_)};
    }

    constexpr iterator end()
    {
        return iterator{std::ranges::end(r1_), std::ranges::begin(r2_),
                        std::ranges::begin(r3_), std::ranges::begin(r2_),
                        std::ranges::end(r2_), std::ranges::begin(r3_),
                        std::ranges::end(r3_)};
    }

    constexpr iterator end() const
    {
        return iterator{std::ranges::end(r1_), std::ranges::begin(r2_),
                        std::ranges::begin(r3_), std::ranges::begin(r2_),
                        std::ranges::end(r2_), std::ranges::begin(r3_),
                        std::ranges::end(r3_)};
    }

    constexpr auto size() const
        requires(std::ranges::sized_range<const R1> && std::ranges::sized_range<const R2> &&
                 std::ranges::sized_range<const R3>)
    {
        return std::ranges::size(r1_) * std::ranges::size(r2_) *
               std::ranges::size(r3_);
    }

    constexpr auto size()
        requires(std::ranges::sized_range<R1> && std::ranges::sized_range<R2> &&
                 std::ranges::sized_range<R3>)
    {
        return std::ranges::size(r1_) * std::ranges::size(r2_) *
               std::ranges::size(r3_);
    }
};

template <typename R1, typename R2, typename R3>
cartesian_product_view(R1&&, R2&&, R3&&)
    -> cartesian_product_view<std::views::all_t<R1>,
                              std::views::all_t<R2>,
                              std::views::all_t<R3>>;

// Factory function
struct cartesian_product_fn {
    template <std::ranges::viewable_range R1,
              std::ranges::viewable_range R2,
              std::ranges::viewable_range R3>
    constexpr auto operator()(R1&& r1, R2&& r2, R3&& r3) const
    {
        return cartesian_product_view(std::views::all(std::forward<R1>(r1)),
                                      std::views::all(std::forward<R2>(r2)),
                                      std::views::all(std::forward<R3>(r3)));
    }
};
inline constexpr cartesian_product_fn cartesian_product{};

// ===========================================================================
// linear_distribute(mn, mx, n)
//
// Returns std::vector<T> of n linearly-spaced values from mn to mx.
// ===========================================================================
template <typename T>
std::vector<T> linear_distribute(T mn, T mx, int n)
{
    std::vector<T> v(n);
    for (int i = 0; i < n; ++i)
        v[i] = n > 1 ? mn + i * (mx - mn) / (n - 1) : mn;
    return v;
}

} // namespace ccs
