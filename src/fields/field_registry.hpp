#pragma once

//
// Centralized field storage registry.
//
// Owns a flat array of Kokkos::View<real*> organized by slot and buffer index.
// Each slot holds up to MaxS scalar fields (4 buffers each) and MaxV vector
// fields (12 buffers each), following the layout defined by field_layout<MaxS,MaxV>.
//
// Handles (scalar_handle, vector_handle, buf_handle) index into this storage.
// field_ref is a lightweight 12-byte token that identifies a slot and its
// current allocation state.
//

#include "handle.hpp"
#include "kokkos_types.hpp"
#include "scalar.hpp"
#include "shoccs_config.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <span>
#include <string>
#include <utility>

namespace ccs
{

// ---------------------------------------------------------------------------
// system_size: plain sizing token for field allocation.
// ---------------------------------------------------------------------------

struct system_size {
    integer nscalars{};
    integer nvectors{};
    integer d_size{}, rx_size{}, ry_size{}, rz_size{};

    constexpr friend auto operator<=>(const system_size&, const system_size&) = default;
};

// ---------------------------------------------------------------------------
// field_ref: lightweight slot token (trivially copyable, fits in SBO).
// ---------------------------------------------------------------------------

struct field_ref {
    int slot      = -1;
    int n_scalars = 0;
    int n_vectors = 0;

    constexpr bool operator==(const field_ref&) const = default;
};

static_assert(std::is_trivially_copyable_v<field_ref>);
static_assert(sizeof(field_ref) == 12);

// ---------------------------------------------------------------------------
// field_registry: flat Kokkos::View storage indexed by handles.
// ---------------------------------------------------------------------------

template <int MaxSlots, int MaxS, int MaxV>
class field_registry
{
public:
    using layout_type = field_layout<MaxS, MaxV>;
    static constexpr int buffers_per_slot = layout_type::total_buffers;

    field_registry() = default;

    // -- Allocation ----------------------------------------------------------

    field_ref allocate_scalar(int slot, int scalar_index,
                              int d_sz, int rx_sz, int ry_sz, int rz_sz)
    {
        assert(slot >= 0 && slot < MaxSlots);
        assert(scalar_index >= 0 && scalar_index < MaxS);
        assert(scalar_index == metadata_[slot].n_scalars);

        // Construct handle via direct arithmetic (cannot use consteval factory
        // with runtime index).
        scalar_handle sh{scalar_index * layout_type::scalar_stride};

        int base = slot * buffers_per_slot;
        auto label = [&](const char* suffix) {
            return "s" + std::to_string(scalar_index) + "_" + suffix;
        };

        buffers_[base + sh.D().id]  = Kokkos::View<real*>(label("D"),  d_sz);
        buffers_[base + sh.Rx().id] = Kokkos::View<real*>(label("Rx"), rx_sz);
        buffers_[base + sh.Ry().id] = Kokkos::View<real*>(label("Ry"), ry_sz);
        buffers_[base + sh.Rz().id] = Kokkos::View<real*>(label("Rz"), rz_sz);

        metadata_[slot].slot = slot;
        metadata_[slot].n_scalars++;

        return metadata_[slot];
    }

    field_ref allocate_vector(int slot, int vector_index,
                              int d_sz, int rx_sz, int ry_sz, int rz_sz)
    {
        assert(slot >= 0 && slot < MaxSlots);
        assert(vector_index >= 0 && vector_index < MaxV);
        assert(vector_index == metadata_[slot].n_vectors);

        vector_handle vh{layout_type::vector_base +
                         vector_index * layout_type::vector_stride};

        int base = slot * buffers_per_slot;

        // All 3 components share the same sizes.
        const char* comp_names[] = {"x", "y", "z"};
        const char* buf_names[]  = {"D", "Rx", "Ry", "Rz"};
        int sizes[] = {d_sz, rx_sz, ry_sz, rz_sz};

        auto comps = vh.components();
        for (int c = 0; c < 3; ++c) {
            auto bufs = comps[c].all();
            for (int b = 0; b < 4; ++b) {
                std::string lbl = "v" + std::to_string(vector_index) +
                                  "_" + comp_names[c] + buf_names[b];
                buffers_[base + bufs[b].id] =
                    Kokkos::View<real*>(lbl, sizes[b]);
            }
        }

        metadata_[slot].slot = slot;
        metadata_[slot].n_vectors++;

        return metadata_[slot];
    }

    // -- Access --------------------------------------------------------------

    Kokkos::View<real*>& view(field_ref ref, buf_handle h)
    {
        assert(ref.slot >= 0 && ref.slot < MaxSlots);
        assert(h.id >= 0 && h.id < buffers_per_slot);
        return buffers_[ref.slot * buffers_per_slot + h.id];
    }

    const Kokkos::View<real*>& view(field_ref ref, buf_handle h) const
    {
        assert(ref.slot >= 0 && ref.slot < MaxSlots);
        assert(h.id >= 0 && h.id < buffers_per_slot);
        return buffers_[ref.slot * buffers_per_slot + h.id];
    }

    real* data(field_ref ref, buf_handle h)
    {
        assert(ref.slot >= 0 && ref.slot < MaxSlots);
        assert(h.id >= 0 && h.id < buffers_per_slot);
        return view(ref, h).data();
    }

    const real* data(field_ref ref, buf_handle h) const
    {
        assert(ref.slot >= 0 && ref.slot < MaxSlots);
        assert(h.id >= 0 && h.id < buffers_per_slot);
        return view(ref, h).data();
    }

    int size(field_ref ref, buf_handle h) const
    {
        assert(ref.slot >= 0 && ref.slot < MaxSlots);
        assert(h.id >= 0 && h.id < buffers_per_slot);
        return static_cast<int>(view(ref, h).extent(0));
    }

    // -- Bulk operations -----------------------------------------------------

    void deep_copy_slot(int dst, int src)
    {
        assert(dst >= 0 && dst < MaxSlots);
        assert(src >= 0 && src < MaxSlots);

        int dst_base = dst * buffers_per_slot;
        int src_base = src * buffers_per_slot;

        for (int i = 0; i < buffers_per_slot; ++i) {
            if (buffers_[src_base + i].extent(0) > 0) {
                assert(buffers_[dst_base + i].extent(0) == buffers_[src_base + i].extent(0));
                Kokkos::deep_copy(buffers_[dst_base + i],
                                  buffers_[src_base + i]);
            }
        }
    }

    void swap_slots(int a, int b)
    {
        assert(a >= 0 && a < MaxSlots);
        assert(b >= 0 && b < MaxSlots);
        assert(metadata_[a].n_scalars == metadata_[b].n_scalars &&
               metadata_[a].n_vectors == metadata_[b].n_vectors);

        int a_base = a * buffers_per_slot;
        int b_base = b * buffers_per_slot;

        for (int i = 0; i < buffers_per_slot; ++i) {
            std::swap(buffers_[a_base + i], buffers_[b_base + i]);
        }

        // Swap metadata and fix slot indices.
        std::swap(metadata_[a], metadata_[b]);
        metadata_[a].slot = a;
        metadata_[b].slot = b;
    }

private:
    static constexpr int total_views_ = MaxSlots * buffers_per_slot;
    std::array<Kokkos::View<real*>, total_views_> buffers_{};
    std::array<field_ref, MaxSlots> metadata_{};
};

// ---------------------------------------------------------------------------
// Span bridge: extract scalar_span / scalar_view from registry storage.
// ---------------------------------------------------------------------------

template <int MaxSlots, int MaxS, int MaxV>
scalar_span extract_scalar_span(field_registry<MaxSlots, MaxS, MaxV>& reg,
                                field_ref ref, scalar_handle h)
{
    auto sp = [&](buf_handle bh) -> std::span<real> {
        return {reg.data(ref, bh),
                static_cast<std::size_t>(reg.size(ref, bh))};
    };
    return scalar_span{sp(h.D()), sp(h.Rx()), sp(h.Ry()), sp(h.Rz())};
}

template <int MaxSlots, int MaxS, int MaxV>
scalar_view extract_scalar_view(const field_registry<MaxSlots, MaxS, MaxV>& reg,
                                field_ref ref, scalar_handle h)
{
    auto sp = [&](buf_handle bh) -> std::span<const real> {
        return {reg.data(ref, bh),
                static_cast<std::size_t>(reg.size(ref, bh))};
    };
    return scalar_view{sp(h.D()), sp(h.Rx()), sp(h.Ry()), sp(h.Rz())};
}

// ---------------------------------------------------------------------------
// Simulation-chain registry: the single concrete type used by systems,
// integrators, and simulation_cycle.
// ---------------------------------------------------------------------------

using sim_registry = field_registry<8, 8, 4>;

} // namespace ccs
