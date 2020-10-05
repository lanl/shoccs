#include "divergence.hpp"
#include "identity_stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "random/random.hpp"

#include <range/v3/algorithm/count.hpp>
#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/filter.hpp>

#include "fields/matchers.hpp"


TEST_CASE("domain")
{
    using namespace ccs;
    using Catch::Matchers::Approx;

    stencil st{identity_stencil{}};
    const int3 extents{5, 4, 1};
    auto m = mesh{real3{-1, -1, -1}, real3{1, 1, 1}, extents};

    auto dd = domain_boundaries{boundary::dirichlet, boundary::dirichlet};
    auto nn = domain_boundaries{boundary::neumann, boundary::neumann};
    grid_boundaries grid_b{dd, dd, nn};
    object_boundaries obj_b{};

    auto g = geometry{};

    auto div = op::divergence{st, m, g, grid_b, obj_b};

    randomize();

    auto solution =
        vs::generate_n([]() { return pick(); }, m.size()) | rs::to<std::vector<real>>();

    y_field s{solution, extents};
    const v_field f{s};
    const v_field df{f};
    y_field dxyz {extents};

    // no objects in domain
    vector_range<std::vector<real>> f_bvals{};
    vector_range<std::vector<real>> df_bvals{};

    s *= m.dims();
    div(f, df, f_bvals, df_bvals, dxyz);

    REQUIRE_THAT(s, Approx(dxyz));
}

TEST_CASE("objects")
{
    using namespace ccs;
    using Catch::Matchers::Approx;

    stencil st{identity_stencil{}};
    const int3 extents{31, 33, 32};
    auto m = mesh{real3{-1, -1, -1}, real3{1, 1, 1}, extents};

    auto shapes = std::vector{make_sphere(0, real3{-0.1, -0.3, -0.3}, 0.3),
                              make_sphere(1, real3{0.2, 0.3, 0.4}, 0.4)};

    auto dd = domain_boundaries{boundary::dirichlet, boundary::dirichlet};
    auto nn = domain_boundaries{boundary::neumann, boundary::neumann};
    grid_boundaries grid_b{dd, nn, dd};
    object_boundaries obj_b{boundary::dirichlet, boundary::neumann};

    auto geom = geometry{shapes, m};
    auto div = op::divergence{st, m, geom, grid_b, obj_b};

    randomize();

    auto solution =
        vs::generate_n([]() { return pick(); }, m.size()) | rs::to<std::vector<real>>();

    z_field s{solution, extents};
    const v_field f{s};
    const v_field df{f};
    z_field dxyz{extents};

    auto coords = vs::transform([](auto&& info) { return info.solid_coord; });
    auto is_neumann = vs::filter(
        [&obj_b](auto&& info) { return obj_b[info.shape_id] == boundary::neumann; });

    // copy field values to boundary range.
    vector_range<std::vector<real>> b_values_f{};
    b_values_f <<= f >> select(geom.Rxyz() >> coords);
    const auto b_f = b_values_f;

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

    div(f, df, b_values_f, b_values_df, dxyz);

    // size should not have changed
    REQUIRE(b_f.size() == b_values_f.size());
    // compare boundary values
    REQUIRE_THAT(b_values_f, Approx(b_f));
    
    // zero out the solid points so we can easily compare fluid values
    dxyz >> select(geom.Sy()) <<= 0;
    s >> select(geom.Sy()) <<= 0;
    s *= 3;

    REQUIRE_THAT(s, Approx(dxyz));
}