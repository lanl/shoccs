#pragma once

#include "field_registry.hpp"
#include "kokkos_types.hpp"
#include "shoccs_config.hpp"

#include <type_traits>

namespace ccs
{

// ---------------------------------------------------------------------------
// Expression node types for expression templates.
//
// Each node type carries pre-extracted data (pointers or values) and provides
// operator()(int i) to evaluate at index i. All types are trivially copyable
// to ensure safe capture in Kokkos lambdas (D-ET2).
// ---------------------------------------------------------------------------

struct handle_expr {
    real* ptr;
    constexpr real operator()(int i) const { return ptr[i]; }
};

struct scalar_literal_expr {
    real value;
    constexpr real operator()(int i) const { (void)i; return value; }
};

template <typename Op, typename Lhs, typename Rhs>
struct binary_expr {
    static_assert(std::is_trivially_copyable_v<Op>);
    static_assert(std::is_trivially_copyable_v<Lhs>);
    static_assert(std::is_trivially_copyable_v<Rhs>);
    Op op;
    Lhs lhs;
    Rhs rhs;
    constexpr real operator()(int i) const { return op(lhs(i), rhs(i)); }
};

template <typename Op, typename Arg>
struct unary_expr {
    static_assert(std::is_trivially_copyable_v<Op>);
    static_assert(std::is_trivially_copyable_v<Arg>);
    Op op;
    Arg arg;
    constexpr real operator()(int i) const { return op(arg(i)); }
};

// Trivially-copyable assertions at namespace scope.
static_assert(std::is_trivially_copyable_v<handle_expr>);
static_assert(std::is_trivially_copyable_v<scalar_literal_expr>);

// ---------------------------------------------------------------------------
// Aliasing detection: contains_ptr checks if a destination pointer appears
// anywhere in an expression tree (D-ET3).
// ---------------------------------------------------------------------------

inline bool contains_ptr(const handle_expr& e, const real* target)
{
    return e.ptr == target;
}

inline bool contains_ptr(const scalar_literal_expr&, const real*)
{
    return false;
}

template <typename Op, typename Lhs, typename Rhs>
inline bool contains_ptr(const binary_expr<Op, Lhs, Rhs>& e, const real* target)
{
    return contains_ptr(e.lhs, target) || contains_ptr(e.rhs, target);
}

template <typename Op, typename Arg>
inline bool contains_ptr(const unary_expr<Op, Arg>& e, const real* target)
{
    return contains_ptr(e.arg, target);
}

// ---------------------------------------------------------------------------
// assign: evaluate expression into destination buffer via parallel_for.
// If the destination pointer appears in the expression tree (aliasing),
// stage through a temporary Kokkos::View to avoid data races (D-ET3).
//
// IMPORTANT: Requires synchronous execution_space (DefaultHostExecutionSpace).
// Expr captures raw real* pointers whose lifetime is only guaranteed for the
// duration of this call. For async/device execution spaces, Expr must capture
// Kokkos::View instead of raw pointers.
// ---------------------------------------------------------------------------

template <typename Expr>
void assign(real* dst, int n, Expr expr)
{
    if (contains_ptr(expr, dst)) {
        // Alias detected: evaluate into temporary, then copy back.
        Kokkos::View<real*, memory_space> tmp("expr_tmp", n);
        real* tmp_ptr = tmp.data();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, n),
            KOKKOS_LAMBDA(int i) { tmp_ptr[i] = expr(i); });
        Kokkos::View<real*, memory_space, Kokkos::MemoryUnmanaged> dst_um(dst, n);
        Kokkos::deep_copy(dst_um, tmp);
    } else {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, n),
            KOKKOS_LAMBDA(int i) { dst[i] = expr(i); });
    }
}

// ---------------------------------------------------------------------------
// Mutating operators (+=, -=, *=, /=): no aliasing check needed (D-ET3).
// Element-wise compound-assign is always safe since each thread accesses
// only dst[i].
//
// Same synchronous execution_space requirement as assign() — Expr captures
// raw real* pointers valid only for the duration of the call.
// ---------------------------------------------------------------------------

template <typename Expr>
void plus_assign(real* dst, int n, Expr expr)
{
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i) { dst[i] += expr(i); });
}

template <typename Expr>
void minus_assign(real* dst, int n, Expr expr)
{
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i) { dst[i] -= expr(i); });
}

template <typename Expr>
void times_assign(real* dst, int n, Expr expr)
{
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i) { dst[i] *= expr(i); });
}

template <typename Expr>
void divide_assign(real* dst, int n, Expr expr)
{
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i) { dst[i] /= expr(i); });
}

// ---------------------------------------------------------------------------
// Scalar-level mutating operators.
// ---------------------------------------------------------------------------

template <int MS, int MaxS, int MaxV>
void times_assign_scalar(field_registry<MS, MaxS, MaxV>& reg,
                         field_ref ref,
                         scalar_handle sh,
                         real value)
{
    auto bufs = sh.all();
    for (int i = 0; i < 4; ++i) {
        times_assign(
            reg.data(ref, bufs[i]),
            reg.size(ref, bufs[i]),
            scalar_literal_expr{value});
    }
}

} // namespace ccs
