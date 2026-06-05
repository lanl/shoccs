#pragma once

//
// Compile-time handle arithmetic for flat field storage.
//
// Layout (max-capacity):
//   Index 0..3:                         scalar[0] buffers (D, Rx, Ry, Rz)
//   Index 4..7:                         scalar[1] buffers (D, Rx, Ry, Rz)
//   ...
//   Index 4*MaxS .. 4*MaxS+11:          vector[0] buffers (x.D, x.Rx, x.Ry, x.Rz,
//                                                          y.D, y.Rx, y.Ry, y.Rz,
//                                                          z.D, z.Rx, z.Ry, z.Rz)
//   ...
//   Total capacity: 4*MaxS + 12*MaxV buffers
//
// Design decisions:
//   - MaxScalars/MaxVectors are template parameters of the layout, NOT globals.
//     Different systems (heat: {1,0}, scalar_wave: {1,1}) instantiate different
//     layouts. The layout is a value type, usable as an NTTP.
//   - Handles are pure-value, trivially-copyable structs with defaulted operator==,
//     satisfying C++20 structural type requirements for use as NTTPs.
//   - Factory functions are consteval: index violations are compile-time errors.
//   - The handle stores only a buffer index, not a length. Lengths are queried
//     from the registry at kernel-launch time (see proposal section 10.7).
//

#include <array>
#include <type_traits>

namespace ccs
{

// ---------------------------------------------------------------------------
// Layout descriptor: template parameters fix the maximum capacity.
// ---------------------------------------------------------------------------

template <int MaxS, int MaxV>
struct field_layout {
    static_assert(MaxS >= 0 && MaxV >= 0);

    static constexpr int max_scalars      = MaxS;
    static constexpr int max_vectors      = MaxV;
    static constexpr int scalar_stride    = 4;   // D, Rx, Ry, Rz
    static constexpr int vector_stride    = 12;  // 3 components x 4 buffers
    static constexpr int vector_base      = MaxS * scalar_stride;
    static constexpr int total_buffers    = vector_base + MaxV * vector_stride;

    // Active counts (runtime state embedded in the layout value).
    // These are set once at construction and then frozen.
    int n_scalars = 0;
    int n_vectors = 0;

    constexpr bool operator==(const field_layout&) const = default;
};

// ---------------------------------------------------------------------------
// Buffer handle: a single buffer index into flat storage.
// Structural type => usable as NTTP.
// ---------------------------------------------------------------------------

struct buf_handle {
    int id = -1;

    constexpr bool operator==(const buf_handle&) const = default;
    constexpr explicit operator bool() const { return id >= 0; }
};

// ---------------------------------------------------------------------------
// Scalar handle: named access to the 4 buffers of one scalar field.
//
//   scalar[i] occupies indices [i*4 .. i*4+3]:
//     +0 = D,  +1 = Rx,  +2 = Ry,  +3 = Rz
// ---------------------------------------------------------------------------

struct scalar_handle {
    int base = -1;  // = scalar_index * 4

    constexpr buf_handle D()  const { return {base + 0}; }
    constexpr buf_handle Rx() const { return {base + 1}; }
    constexpr buf_handle Ry() const { return {base + 2}; }
    constexpr buf_handle Rz() const { return {base + 3}; }

    // All 4 buffer indices, for iteration.
    constexpr std::array<buf_handle, 4> all() const
    {
        return {D(), Rx(), Ry(), Rz()};
    }

    // R-component handles (Rx, Ry, Rz) as a group.
    constexpr std::array<buf_handle, 3> R() const
    {
        return {Rx(), Ry(), Rz()};
    }

    constexpr bool operator==(const scalar_handle&) const = default;
    constexpr explicit operator bool() const { return base >= 0; }
};

// ---------------------------------------------------------------------------
// Vector handle: named access to the 12 buffers of one vector field.
//
//   vector[i] occupies indices [vector_base + i*12 .. vector_base + i*12 + 11]:
//     x: +0..+3   (D, Rx, Ry, Rz)
//     y: +4..+7   (D, Rx, Ry, Rz)
//     z: +8..+11  (D, Rx, Ry, Rz)
// ---------------------------------------------------------------------------

struct vector_handle {
    int base = -1;  // = vector_base + vector_index * 12

    // Component access as scalar_handles.
    constexpr scalar_handle x() const { return {base + 0}; }
    constexpr scalar_handle y() const { return {base + 4}; }
    constexpr scalar_handle z() const { return {base + 8}; }

    // Direct buffer access with full names matching current codebase selectors.
    // Domain components.
    constexpr buf_handle Dx()  const { return x().D(); }
    constexpr buf_handle Dy()  const { return y().D(); }
    constexpr buf_handle Dz()  const { return z().D(); }

    // x-component boundary.
    constexpr buf_handle xRx() const { return x().Rx(); }
    constexpr buf_handle xRy() const { return x().Ry(); }
    constexpr buf_handle xRz() const { return x().Rz(); }

    // y-component boundary.
    constexpr buf_handle yRx() const { return y().Rx(); }
    constexpr buf_handle yRy() const { return y().Ry(); }
    constexpr buf_handle yRz() const { return y().Rz(); }

    // z-component boundary.
    constexpr buf_handle zRx() const { return z().Rx(); }
    constexpr buf_handle zRy() const { return z().Ry(); }
    constexpr buf_handle zRz() const { return z().Rz(); }

    // All 12 buffer indices.
    constexpr std::array<buf_handle, 12> all() const
    {
        auto xa = x().all();
        auto ya = y().all();
        auto za = z().all();
        return {xa[0], xa[1], xa[2], xa[3],
                ya[0], ya[1], ya[2], ya[3],
                za[0], za[1], za[2], za[3]};
    }

    // All 3 component scalar_handles.
    constexpr std::array<scalar_handle, 3> components() const
    {
        return {x(), y(), z()};
    }

    constexpr bool operator==(const vector_handle&) const = default;
    constexpr explicit operator bool() const { return base >= 0; }
};

// ---------------------------------------------------------------------------
// Consteval factory functions.
//
// These guarantee compile-time bounds checking. If you write
//   constexpr auto h = make_scalar_handle<layout>(2);
// and the layout only has 1 active scalar, the program is ill-formed.
//
// Use consteval so that even in non-constexpr contexts the call is
// evaluated at compile time and the assertion fires as a compile error.
// ---------------------------------------------------------------------------

template <int MaxS, int MaxV>
consteval scalar_handle make_scalar_handle(field_layout<MaxS, MaxV> layout, int i)
{
    // Bounds check against active count.
    if (i < 0 || i >= layout.n_scalars) {
        // Trigger a compile-time error: calling a non-constexpr function
        // inside consteval makes the program ill-formed.
        throw "scalar index out of bounds";
    }
    return {i * field_layout<MaxS, MaxV>::scalar_stride};
}

template <int MaxS, int MaxV>
consteval vector_handle make_vector_handle(field_layout<MaxS, MaxV> layout, int i)
{
    if (i < 0 || i >= layout.n_vectors) {
        throw "vector index out of bounds";
    }
    return {field_layout<MaxS, MaxV>::vector_base +
            i * field_layout<MaxS, MaxV>::vector_stride};
}

// Compile-time proof that handle types satisfy structural type requirements.
static_assert(std::is_trivially_copyable_v<buf_handle>);
static_assert(std::is_trivially_copyable_v<scalar_handle>);
static_assert(std::is_trivially_copyable_v<vector_handle>);

static_assert(std::is_aggregate_v<buf_handle>);
static_assert(std::is_aggregate_v<scalar_handle>);
static_assert(std::is_aggregate_v<vector_handle>);

} // namespace ccs
