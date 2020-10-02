#include "gradient.hpp"
#include "identity_stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "random/random.hpp"

#include <range/v3/algorithm/count.hpp>
#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/filter.hpp>

#include <iostream>

TEST_CASE("domain")
{
    using namespace ccs;

    stencil st{identity_stencil{}};
    const int3 extents{12, 10, 13};
    auto m = mesh{real3{}, real3{1, 1, 1}, extents};

    auto dd = domain_boundaries{boundary::dirichlet, boundary::dirichlet};
    auto nn = domain_boundaries{boundary::neumann, boundary::neumann};
    grid_boundaries grid_b{dd, nn, dd};
    object_boundaries obj_b{};

    auto g = geometry{};

    auto grad = op::gradient{st, m, g, grid_b, obj_b};

    randomize();

    auto solution =
        vs::generate_n([]() { return pick(); }, m.size()) | rs::to<std::vector<real>>();

    x_field f{solution, extents};
    x_field df{solution, extents};

    // no objects in domain
    vector_range<std::vector<real>> f_bvals{};
    vector_range<std::vector<real>> df_bvals{};

    vector_field<real> dxyz{extents};

    grad(f, df, f_bvals, df_bvals, dxyz);
    // check x-direction results
    REQUIRE(rs::equal(f, dxyz.x));
    // check y-direction results
    REQUIRE(rs::equal(y_field{f}, dxyz.y));
    // check z-direction results
    REQUIRE(rs::equal(z_field{f}, dxyz.z));
}

TEST_CASE("objects")
{
    using namespace ccs;

    stencil st{identity_stencil{}};
    const int3 extents{31, 33, 32};
    auto m = mesh{real3{-1, -1, -1}, real3{1, 1, 1}, extents};

    auto shapes = std::vector{make_sphere(0, real3{-0.1, -0.2, -0.3}, 0.3),
                              make_sphere(1, real3{0.2, 0.3, 0.4}, 0.4)};

    auto dd = domain_boundaries{boundary::dirichlet, boundary::dirichlet};
    auto nn = domain_boundaries{boundary::neumann, boundary::neumann};
    grid_boundaries grid_b{dd, nn, dd};
    object_boundaries obj_b{boundary::dirichlet, boundary::neumann};

    auto geom = geometry{shapes, m};

    auto grad = op::gradient{st, m, geom, grid_b, obj_b};

    randomize();

    auto solution =
        vs::generate_n([]() { return pick(); }, m.size()) | rs::to<std::vector<real>>();

    x_field f{solution, extents};
    x_field df{solution, extents};
    vector_field<real> dxyz{extents};

    auto coords = vs::transform([](auto&& info) { return info.solid_coord; });
    auto is_neumann = vs::filter(
        [&obj_b](auto&& info) { return obj_b[info.shape_id] == boundary::neumann; });

    // copy field values to boundary range.
    vector_range<std::vector<real>> b_values_f{};
    b_values_f <<= f >> select(geom.Rxyz() >> coords);
    const auto b_f = b_values_f;
    const auto d_sz = b_f.size();

    // copy field values to use as neumann bcs
    // recall that the filter does not produce a sized range so we need to count and
    // resize
    int3 df_size = [&geom, &obj_b]() {
        auto s = geom.Rxyz() >> [&obj_b](auto&& r) {
            return rs::count(
                r, boundary::neumann, [&obj_b](auto&& i) { return obj_b[i.shape_id]; });
        };
        return int3{(int)s.x, (int)s.y, (int)s.z};
    }();
    vector_range<std::vector<real>> b_values_df{};
    b_values_df.resize(df_size);
    b_values_df <<= f >> select(geom.Rxyz() >> is_neumann >> coords);
    const auto n_sz = b_values_df.size();

    grad(f, df, b_values_f, b_values_df, dxyz);

    // size should not have changed
    REQUIRE(b_f.size() == b_values_f.size());
    // compare boundary values
    //std::cout << "# of dirichlet " << vs::all(d_sz) << '\n';
    //std::cout << "# of neumann " << vs::all(n_sz) << '\n';
    for (auto&& [c, e] : vs::zip(b_values_f.x, b_f.x)) REQUIRE(c == Catch::Approx(e));
    for (auto&& [c, e] : vs::zip(b_values_f.y, b_f.y)) REQUIRE(c == Catch::Approx(e));
    for (auto&& [c, e] : vs::zip(b_values_f.z, b_f.z)) REQUIRE(c == Catch::Approx(e));

    // zero out the solid points so we can easily compare fluid values
    vector_field vf{f};
    vf >> select(geom.Sxyz()) <<= 1000;
    dxyz >> select(geom.Sxyz()) <<= 1000;

    std::cout << (vf.x | vs::take(10)) << '\n';
    std::cout << (dxyz.x | vs::take(10)) << '\n';

    for (int i = 0; auto&& [c, e] : vs::zip(dxyz.x, vf.x)) {
        std::cout << "[" << (i++) << "]\t" << c << '\t' << e << '\n';
        REQUIRE(c == Catch::Approx(e));
    }
    for (auto&& [c, e] : vs::zip(dxyz.y, vf.y)) REQUIRE(c == Catch::Approx(e));
    for (auto&& [c, e] : vs::zip(dxyz.z, vf.z)) REQUIRE(c == Catch::Approx(e));
}
