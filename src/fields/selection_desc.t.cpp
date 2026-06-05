#include "fields/selection_desc.hpp"

#include "fields/expr.hpp"
#include "operators/boundaries.hpp"

#include <set>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace ccs;

// ---------------------------------------------------------------------------
// Custom main: Kokkos must be initialized before any test allocates Views.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

// ---------------------------------------------------------------------------
// 11.1a — contiguous_selection
// ---------------------------------------------------------------------------

TEST_CASE("contiguous_selection element and count")
{
    contiguous_selection sel{10, 5};
    REQUIRE(sel.count() == 5);
    REQUIRE(sel.element(0) == 10);
    REQUIRE(sel.element(1) == 11);
    REQUIRE(sel.element(2) == 12);
    REQUIRE(sel.element(3) == 13);
    REQUIRE(sel.element(4) == 14);
}

TEST_CASE("contiguous_selection is trivially copyable")
{
    STATIC_REQUIRE(std::is_trivially_copyable_v<contiguous_selection>);
}

// ---------------------------------------------------------------------------
// 11.1a — strided_selection
// ---------------------------------------------------------------------------

TEST_CASE("strided_selection element and count")
{
    // Example: y-plane pattern for 8x6x4 mesh at j=1
    // offset = j*nz = 1*4 = 4, inner_count = nz = 4, outer_count = nx = 8,
    // outer_stride = ny*nz = 24
    strided_selection sel{4, 4, 8, 24};
    REQUIRE(sel.count() == 32); // 4 * 8

    // First block: elements 4, 5, 6, 7
    REQUIRE(sel.element(0) == 4);
    REQUIRE(sel.element(1) == 5);
    REQUIRE(sel.element(2) == 6);
    REQUIRE(sel.element(3) == 7);

    // Second block: elements 4 + 24 = 28, 29, 30, 31
    REQUIRE(sel.element(4) == 28);
    REQUIRE(sel.element(5) == 29);
    REQUIRE(sel.element(6) == 30);
    REQUIRE(sel.element(7) == 31);
}

TEST_CASE("strided_selection z-plane degenerate case")
{
    // z-plane pattern for 8x6x4 mesh at k=2
    // offset = 2, inner_count = 1, outer_count = nx*ny = 48, outer_stride = nz = 4
    strided_selection sel{2, 1, 48, 4};
    REQUIRE(sel.count() == 48);

    // element(i) = 2 + i*4
    REQUIRE(sel.element(0) == 2);
    REQUIRE(sel.element(1) == 6);
    REQUIRE(sel.element(2) == 10);
    REQUIRE(sel.element(47) == 2 + 47 * 4);
}

TEST_CASE("strided_selection is trivially copyable")
{
    STATIC_REQUIRE(std::is_trivially_copyable_v<strided_selection>);
}

// ---------------------------------------------------------------------------
// 11.1a — gather_selection
// ---------------------------------------------------------------------------

TEST_CASE("gather_selection element and count")
{
    Kokkos::View<int*, memory_space> idx("idx", 4);
    auto h = Kokkos::create_mirror_view(idx);
    h(0) = 3;
    h(1) = 7;
    h(2) = 1;
    h(3) = 42;
    Kokkos::deep_copy(idx, h);

    gather_selection sel{idx};
    REQUIRE(sel.count() == 4);
    // Verify element() returns the correct index for each position
    REQUIRE(sel.element(0) == 3);
    REQUIRE(sel.element(1) == 7);
    REQUIRE(sel.element(2) == 1);
    REQUIRE(sel.element(3) == 42);
}

// ---------------------------------------------------------------------------
// 11.1a — Cross-check: plane descriptors match expected flat indices
// ---------------------------------------------------------------------------

TEST_CASE("x-plane descriptor cross-check against flat indices")
{
    // 8x6x4 mesh, x-plane at i=3
    int nx = 8, ny = 6, nz = 4;
    index_extents ext{{nx, ny, nz}};
    int plane_i = 3;

    auto desc = make_x_plane_desc(ext, plane_i);

    // Build expected set of flat indices: {i*ny*nz + j*nz + k} for fixed i
    std::set<int> expected;
    for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k)
            expected.insert(plane_i * ny * nz + j * nz + k);

    REQUIRE(desc.count() == static_cast<int>(expected.size()));

    std::set<int> actual;
    for (int q = 0; q < desc.count(); ++q)
        actual.insert(desc.element(q));

    REQUIRE(actual == expected);
}

TEST_CASE("y-plane descriptor cross-check against flat indices")
{
    // 8x6x4 mesh, y-plane at j=3
    int nx = 8, ny = 6, nz = 4;
    index_extents ext{{nx, ny, nz}};
    int plane_j = 3;

    auto desc = make_y_plane_desc(ext, plane_j);

    std::set<int> expected;
    for (int i = 0; i < nx; ++i)
        for (int k = 0; k < nz; ++k)
            expected.insert(i * ny * nz + plane_j * nz + k);

    REQUIRE(desc.count() == static_cast<int>(expected.size()));

    std::set<int> actual;
    for (int q = 0; q < desc.count(); ++q)
        actual.insert(desc.element(q));

    REQUIRE(actual == expected);
}

TEST_CASE("z-plane descriptor cross-check against flat indices")
{
    // 8x6x4 mesh, z-plane at k=2
    int nx = 8, ny = 6, nz = 4;
    index_extents ext{{nx, ny, nz}};
    int plane_k = 2;

    auto desc = make_z_plane_desc(ext, plane_k);

    std::set<int> expected;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            expected.insert(i * ny * nz + j * nz + plane_k);

    REQUIRE(desc.count() == static_cast<int>(expected.size()));

    std::set<int> actual;
    for (int q = 0; q < desc.count(); ++q)
        actual.insert(desc.element(q));

    REQUIRE(actual == expected);
}

// ---------------------------------------------------------------------------
// 11.2a — assign_selected
// ---------------------------------------------------------------------------

TEST_CASE("assign_selected with contiguous_selection")
{
    constexpr int N = 20;
    Kokkos::View<real*, memory_space> dst("dst", N);
    Kokkos::View<real*, memory_space> src("src", N);

    // Fill dst with sentinel, src with known values
    Kokkos::deep_copy(dst, -1.0);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, N),
        KOKKOS_LAMBDA(int i) { src(i) = 100.0 + i; });

    contiguous_selection sel{5, 4}; // elements 5,6,7,8
    assign_selected(dst.data(), sel, handle_expr{src.data()});

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst);
    // Untouched elements remain -1
    for (int i = 0; i < 5; ++i) REQUIRE(h(i) == -1.0);
    // Selected elements get src values
    REQUIRE(h(5) == 105.0);
    REQUIRE(h(6) == 106.0);
    REQUIRE(h(7) == 107.0);
    REQUIRE(h(8) == 108.0);
    // Untouched after selection
    for (int i = 9; i < N; ++i) REQUIRE(h(i) == -1.0);
}

TEST_CASE("assign_selected with strided_selection")
{
    // y-plane-like pattern: 3 outer blocks of 2 inner, stride 6
    // Elements: 1,2, 7,8, 13,14
    constexpr int N = 20;
    Kokkos::View<real*, memory_space> dst("dst", N);
    Kokkos::View<real*, memory_space> src("src", N);

    Kokkos::deep_copy(dst, -1.0);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, N),
        KOKKOS_LAMBDA(int i) { src(i) = 200.0 + i; });

    strided_selection sel{1, 2, 3, 6};
    assign_selected(dst.data(), sel, handle_expr{src.data()});

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst);
    // Check selected elements
    REQUIRE(h(1) == 201.0);
    REQUIRE(h(2) == 202.0);
    REQUIRE(h(7) == 207.0);
    REQUIRE(h(8) == 208.0);
    REQUIRE(h(13) == 213.0);
    REQUIRE(h(14) == 214.0);
    // Check some untouched elements
    REQUIRE(h(0) == -1.0);
    REQUIRE(h(3) == -1.0);
    REQUIRE(h(6) == -1.0);
    REQUIRE(h(9) == -1.0);
    REQUIRE(h(15) == -1.0);
}

TEST_CASE("assign_selected with gather_selection")
{
    constexpr int N = 20;
    Kokkos::View<real*, memory_space> dst("dst", N);
    Kokkos::View<real*, memory_space> src("src", N);

    Kokkos::deep_copy(dst, -1.0);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, N),
        KOKKOS_LAMBDA(int i) { src(i) = 300.0 + i; });

    Kokkos::View<int*, memory_space> idx("idx", 3);
    auto hidx = Kokkos::create_mirror_view(idx);
    hidx(0) = 2;
    hidx(1) = 10;
    hidx(2) = 17;
    Kokkos::deep_copy(idx, hidx);

    gather_selection sel{idx};
    assign_selected(dst.data(), sel, handle_expr{src.data()});

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst);
    REQUIRE(h(2) == 302.0);
    REQUIRE(h(10) == 310.0);
    REQUIRE(h(17) == 317.0);
    // Untouched elements
    REQUIRE(h(0) == -1.0);
    REQUIRE(h(1) == -1.0);
    REQUIRE(h(5) == -1.0);
    REQUIRE(h(19) == -1.0);
}

// ---------------------------------------------------------------------------
// 11.2a — fill_selected
// ---------------------------------------------------------------------------

TEST_CASE("fill_selected with contiguous_selection")
{
    constexpr int N = 15;
    Kokkos::View<real*, memory_space> dst("dst", N);
    Kokkos::deep_copy(dst, -1.0);

    contiguous_selection sel{3, 5}; // elements 3..7
    fill_selected(dst.data(), sel, 42.0);

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst);
    for (int i = 0; i < 3; ++i) REQUIRE(h(i) == -1.0);
    for (int i = 3; i < 8; ++i) REQUIRE(h(i) == 42.0);
    for (int i = 8; i < N; ++i) REQUIRE(h(i) == -1.0);
}

TEST_CASE("fill_selected with strided_selection")
{
    constexpr int N = 20;
    Kokkos::View<real*, memory_space> dst("dst", N);
    Kokkos::deep_copy(dst, -1.0);

    // z-plane-like: offset=2, inner=1, outer=4, stride=5
    // Elements: 2, 7, 12, 17
    strided_selection sel{2, 1, 4, 5};
    fill_selected(dst.data(), sel, 99.0);

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst);
    REQUIRE(h(2) == 99.0);
    REQUIRE(h(7) == 99.0);
    REQUIRE(h(12) == 99.0);
    REQUIRE(h(17) == 99.0);
    // Untouched
    REQUIRE(h(0) == -1.0);
    REQUIRE(h(1) == -1.0);
    REQUIRE(h(3) == -1.0);
    REQUIRE(h(6) == -1.0);
}

TEST_CASE("fill_selected with gather_selection")
{
    constexpr int N = 10;
    Kokkos::View<real*, memory_space> dst("dst", N);
    Kokkos::deep_copy(dst, -1.0);

    Kokkos::View<int*, memory_space> idx("idx", 3);
    auto hidx = Kokkos::create_mirror_view(idx);
    hidx(0) = 0;
    hidx(1) = 4;
    hidx(2) = 9;
    Kokkos::deep_copy(idx, hidx);

    gather_selection sel{idx};
    fill_selected(dst.data(), sel, 7.5);

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst);
    REQUIRE(h(0) == 7.5);
    REQUIRE(h(4) == 7.5);
    REQUIRE(h(9) == 7.5);
    // Untouched
    REQUIRE(h(1) == -1.0);
    REQUIRE(h(3) == -1.0);
    REQUIRE(h(8) == -1.0);
}

// ---------------------------------------------------------------------------
// 11.2a — plus_assign_selected
// ---------------------------------------------------------------------------

TEST_CASE("plus_assign_selected with contiguous_selection")
{
    constexpr int N = 15;
    Kokkos::View<real*, memory_space> dst("dst", N);
    Kokkos::View<real*, memory_space> src("src", N);

    // Fill dst with 10.0, src with index values
    Kokkos::deep_copy(dst, 10.0);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, N),
        KOKKOS_LAMBDA(int i) { src(i) = static_cast<real>(i); });

    contiguous_selection sel{2, 4}; // elements 2,3,4,5
    plus_assign_selected(dst.data(), sel, handle_expr{src.data()});

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst);
    // Untouched elements remain 10.0
    REQUIRE(h(0) == 10.0);
    REQUIRE(h(1) == 10.0);
    // Selected elements: 10.0 + i
    REQUIRE(h(2) == 12.0);
    REQUIRE(h(3) == 13.0);
    REQUIRE(h(4) == 14.0);
    REQUIRE(h(5) == 15.0);
    // Untouched after
    REQUIRE(h(6) == 10.0);
    REQUIRE(h(14) == 10.0);
}

TEST_CASE("plus_assign_selected with strided_selection")
{
    constexpr int N = 20;
    Kokkos::View<real*, memory_space> dst("dst", N);
    Kokkos::View<real*, memory_space> src("src", N);

    Kokkos::deep_copy(dst, 5.0);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, N),
        KOKKOS_LAMBDA(int i) { src(i) = 1.0 + i; });

    // offset=1, inner=2, outer=3, stride=6 -> elements 1,2, 7,8, 13,14
    strided_selection sel{1, 2, 3, 6};
    plus_assign_selected(dst.data(), sel, handle_expr{src.data()});

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst);
    REQUIRE(h(1) == 5.0 + 2.0);  // 5 + (1+1)
    REQUIRE(h(2) == 5.0 + 3.0);  // 5 + (1+2)
    REQUIRE(h(7) == 5.0 + 8.0);  // 5 + (1+7)
    REQUIRE(h(8) == 5.0 + 9.0);  // 5 + (1+8)
    REQUIRE(h(13) == 5.0 + 14.0); // 5 + (1+13)
    REQUIRE(h(14) == 5.0 + 15.0); // 5 + (1+14)
    // Untouched
    REQUIRE(h(0) == 5.0);
    REQUIRE(h(3) == 5.0);
    REQUIRE(h(6) == 5.0);
}

TEST_CASE("plus_assign_selected with gather_selection")
{
    constexpr int N = 10;
    Kokkos::View<real*, memory_space> dst("dst", N);
    Kokkos::View<real*, memory_space> src("src", N);

    Kokkos::deep_copy(dst, 100.0);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, N),
        KOKKOS_LAMBDA(int i) { src(i) = static_cast<real>(i * 10); });

    Kokkos::View<int*, memory_space> idx("idx", 3);
    auto hidx = Kokkos::create_mirror_view(idx);
    hidx(0) = 1;
    hidx(1) = 5;
    hidx(2) = 8;
    Kokkos::deep_copy(idx, hidx);

    gather_selection sel{idx};
    plus_assign_selected(dst.data(), sel, handle_expr{src.data()});

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst);
    REQUIRE(h(1) == 110.0);  // 100 + 10
    REQUIRE(h(5) == 150.0);  // 100 + 50
    REQUIRE(h(8) == 180.0);  // 100 + 80
    // Untouched
    REQUIRE(h(0) == 100.0);
    REQUIRE(h(2) == 100.0);
    REQUIRE(h(9) == 100.0);
}

// ---------------------------------------------------------------------------
// 11.2a — scalar_literal_expr as expression argument
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 11.3a — make_gather_from_slices
// ---------------------------------------------------------------------------

TEST_CASE("make_gather_from_slices builds correct index array")
{
    std::vector<index_slice> slices = {{0, 5}, {10, 15}};
    auto sel = make_gather_from_slices(slices);

    REQUIRE(sel.count() == 10);
    REQUIRE(sel.element(0) == 0);
    REQUIRE(sel.element(1) == 1);
    REQUIRE(sel.element(2) == 2);
    REQUIRE(sel.element(3) == 3);
    REQUIRE(sel.element(4) == 4);
    REQUIRE(sel.element(5) == 10);
    REQUIRE(sel.element(6) == 11);
    REQUIRE(sel.element(7) == 12);
    REQUIRE(sel.element(8) == 13);
    REQUIRE(sel.element(9) == 14);
}

TEST_CASE("make_gather_from_slices with single slice")
{
    std::vector<index_slice> slices = {{3, 7}};
    auto sel = make_gather_from_slices(slices);

    REQUIRE(sel.count() == 4);
    REQUIRE(sel.element(0) == 3);
    REQUIRE(sel.element(1) == 4);
    REQUIRE(sel.element(2) == 5);
    REQUIRE(sel.element(3) == 6);
}

TEST_CASE("make_gather_from_slices with empty slices")
{
    std::vector<index_slice> slices = {};
    auto sel = make_gather_from_slices(slices);
    REQUIRE(sel.count() == 0);
}

TEST_CASE("make_gather_from_slices with multiple disjoint slices")
{
    // Three slices: [0,2), [5,8), [20,22)
    std::vector<index_slice> slices = {{0, 2}, {5, 8}, {20, 22}};
    auto sel = make_gather_from_slices(slices);

    REQUIRE(sel.count() == 7);
    // [0,2)
    REQUIRE(sel.element(0) == 0);
    REQUIRE(sel.element(1) == 1);
    // [5,8)
    REQUIRE(sel.element(2) == 5);
    REQUIRE(sel.element(3) == 6);
    REQUIRE(sel.element(4) == 7);
    // [20,22)
    REQUIRE(sel.element(5) == 20);
    REQUIRE(sel.element(6) == 21);
}

// ---------------------------------------------------------------------------
// 11.3b — make_gather_from_predicate
// ---------------------------------------------------------------------------

TEST_CASE("make_gather_from_predicate selects matching indices")
{
    // Build a small mock mesh_object_info array with known shape_ids.
    // shape_id 0 = "Dirichlet", shape_id 1 = "not Dirichlet"
    std::vector<mesh_object_info> infos(6);
    infos[0].shape_id = 1; // not Dirichlet
    infos[1].shape_id = 0; // Dirichlet
    infos[2].shape_id = 0; // Dirichlet
    infos[3].shape_id = 1; // not Dirichlet
    infos[4].shape_id = 0; // Dirichlet
    infos[5].shape_id = 1; // not Dirichlet

    // Select elements where shape_id == 0 (simulating Dirichlet predicate)
    auto sel = make_gather_from_predicate(
        std::span<const mesh_object_info>(infos),
        [](const mesh_object_info& info) { return info.shape_id == 0; });

    REQUIRE(sel.count() == 3);
    REQUIRE(sel.element(0) == 1);
    REQUIRE(sel.element(1) == 2);
    REQUIRE(sel.element(2) == 4);
}

TEST_CASE("make_gather_from_predicate complement predicate")
{
    // Same array, select complement (non-Dirichlet: shape_id != 0)
    std::vector<mesh_object_info> infos(6);
    infos[0].shape_id = 1;
    infos[1].shape_id = 0;
    infos[2].shape_id = 0;
    infos[3].shape_id = 1;
    infos[4].shape_id = 0;
    infos[5].shape_id = 1;

    auto sel = make_gather_from_predicate(
        std::span<const mesh_object_info>(infos),
        [](const mesh_object_info& info) { return info.shape_id != 0; });

    REQUIRE(sel.count() == 3);
    REQUIRE(sel.element(0) == 0);
    REQUIRE(sel.element(1) == 3);
    REQUIRE(sel.element(2) == 5);
}

TEST_CASE("make_gather_from_predicate no matches")
{
    std::vector<mesh_object_info> infos(4);
    for (auto& info : infos)
        info.shape_id = 1;

    auto sel = make_gather_from_predicate(
        std::span<const mesh_object_info>(infos),
        [](const mesh_object_info& info) { return info.shape_id == 99; });

    REQUIRE(sel.count() == 0);
}

TEST_CASE("make_gather_from_predicate all match")
{
    std::vector<mesh_object_info> infos(3);
    for (auto& info : infos)
        info.shape_id = 2;

    auto sel = make_gather_from_predicate(
        std::span<const mesh_object_info>(infos),
        [](const mesh_object_info& info) { return info.shape_id == 2; });

    REQUIRE(sel.count() == 3);
    REQUIRE(sel.element(0) == 0);
    REQUIRE(sel.element(1) == 1);
    REQUIRE(sel.element(2) == 2);
}

TEST_CASE("make_gather_from_predicate empty input")
{
    std::span<const mesh_object_info> infos;
    auto sel = make_gather_from_predicate(
        infos, [](const mesh_object_info&) { return true; });

    REQUIRE(sel.count() == 0);
}

// ---------------------------------------------------------------------------
// 11.2a — scalar_literal_expr as expression argument
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 11.3d — for_each_grid_bc_desc
// ---------------------------------------------------------------------------

TEST_CASE("for_each_grid_bc_desc selects correct Dirichlet faces")
{
    int nx = 8, ny = 6, nz = 4;
    index_extents ext{{nx, ny, nz}};

    // x: D/D, y: D/N, z: F/F
    bcs::Grid grid = {{bcs::dd, bcs::dn, bcs::ff}};

    // Collect all indices from Dirichlet face descriptors
    std::set<int> actual;
    for_each_grid_bc_desc<bcs::Dirichlet>(grid, ext, [&](auto desc) {
        for (int i = 0; i < desc.count(); ++i)
            actual.insert(desc.element(i));
    });

    // Build expected: xmin (i=0), xmax (i=7), ymin (j=0)
    std::set<int> expected;
    for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k)
            expected.insert(0 * ny * nz + j * nz + k);
    for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k)
            expected.insert(7 * ny * nz + j * nz + k);
    for (int i = 0; i < nx; ++i)
        for (int k = 0; k < nz; ++k)
            expected.insert(i * ny * nz + 0 * nz + k);

    REQUIRE(actual == expected);
}

TEST_CASE("for_each_grid_bc_desc with no matching faces")
{
    index_extents ext{{8, 6, 4}};
    bcs::Grid grid = {{bcs::nn, bcs::nn, bcs::nn}};

    int count = 0;
    for_each_grid_bc_desc<bcs::Dirichlet>(grid, ext, [&](auto) { ++count; });

    REQUIRE(count == 0);
}

TEST_CASE("for_each_grid_bc_desc all faces Dirichlet produces 6 descriptors")
{
    index_extents ext{{8, 6, 4}};
    bcs::Grid grid = {{bcs::dd, bcs::dd, bcs::dd}};

    int desc_count = 0;
    for_each_grid_bc_desc<bcs::Dirichlet>(grid, ext, [&](auto) { ++desc_count; });

    REQUIRE(desc_count == 6);
}

TEST_CASE("for_each_grid_bc_desc Neumann faces only")
{
    int nx = 8, ny = 6, nz = 4;
    index_extents ext{{nx, ny, nz}};

    // x: N/D, y: N/N, z: D/N
    bcs::Grid grid = {{bcs::nd, bcs::nn, bcs::dn}};

    // Collect Neumann face indices
    std::set<int> actual;
    for_each_grid_bc_desc<bcs::Neumann>(grid, ext, [&](auto desc) {
        for (int i = 0; i < desc.count(); ++i)
            actual.insert(desc.element(i));
    });

    // Expected Neumann faces: xmin (i=0), ymin (j=0), ymax (j=5), zmax (k=3)
    std::set<int> expected;
    // xmin: i=0
    for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k)
            expected.insert(0 * ny * nz + j * nz + k);
    // ymin: j=0
    for (int i = 0; i < nx; ++i)
        for (int k = 0; k < nz; ++k)
            expected.insert(i * ny * nz + 0 * nz + k);
    // ymax: j=5
    for (int i = 0; i < nx; ++i)
        for (int k = 0; k < nz; ++k)
            expected.insert(i * ny * nz + 5 * nz + k);
    // zmax: k=3
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            expected.insert(i * ny * nz + j * nz + 3);

    REQUIRE(actual == expected);
}

// ---------------------------------------------------------------------------
// 11.2a — scalar_literal_expr as expression argument
// ---------------------------------------------------------------------------

TEST_CASE("assign_selected with scalar_literal_expr")
{
    constexpr int N = 10;
    Kokkos::View<real*, memory_space> dst("dst", N);
    Kokkos::deep_copy(dst, -1.0);

    contiguous_selection sel{2, 3};
    assign_selected(dst.data(), sel, scalar_literal_expr{77.0});

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst);
    REQUIRE(h(0) == -1.0);
    REQUIRE(h(1) == -1.0);
    REQUIRE(h(2) == 77.0);
    REQUIRE(h(3) == 77.0);
    REQUIRE(h(4) == 77.0);
    REQUIRE(h(5) == -1.0);
}
