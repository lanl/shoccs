#include "field_io.hpp"
#include <numeric>
#include <sstream>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("field_data")
{
    using namespace ccs;
    int3 bounds = {4, 5, 6};
    auto fd = ccs::field_data{bounds};
    auto ss = std::stringstream{};
    auto v = std::vector<real>(bounds[0] * bounds[1] * bounds[2]);
    auto vv = std::vector<real>(v.size());
    std::iota(v.begin(), v.end(), -10.0);

    fd.write(ss, &v[0]);
    ss.read(reinterpret_cast<char*>(&vv[0]), sizeof(vv[0]) * vv.size());

    REQUIRE(v == vv);
}

// For this test, one needs to load the output in paraview
TEST_CASE("field_io_2D")
{
    using namespace ccs;

    int3 bounds = {4, 5, 1};
    real3 length = {1.0, 2.0, 3.0};

    auto io = field_io(xdmf(bounds, length),
                       field_data(bounds),
                       interval<int>{},
                       interval<real>{1.0},
                       "iotest");

    auto x = std::vector<real>(bounds[0] * bounds[1] * bounds[2]);
    auto y = std::vector<real>(x.size());
    auto z = std::vector<real>(x.size());

    io.add("X", &x[0]);
    io.add("Y", &y[0]);
    io.add("Z", &z[0]);

    std::iota(x.begin(), x.end(), -10.0);
    std::iota(y.begin(), y.end(), -5.0);
    std::iota(z.begin(), z.end(), 0.0);

    io.write(0, 0.0, 0.0);

    // update field and write again
    std::iota(x.begin(), x.end(), -5.0);
    std::iota(y.begin(), y.end(), 0.0);
    std::iota(z.begin(), z.end(), 5.0);

    io.write(32, 0.5, 0.1); // this call shouldn't write anything
    io.write(32, 0.9, 0.5); // this call should
}
