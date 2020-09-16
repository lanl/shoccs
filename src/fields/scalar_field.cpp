#include "scalar_field.hpp"

#include <range/v3/algorithm/equal.hpp>

#include <cassert>
#include <iostream>

int main()
{
    using namespace ccs;

    const auto x = scalar_field{std::vector{1.0, 2.0, 3.0}};
    auto y = scalar_field{std::vector{4.0, 5.0, 6.0}};

    auto sum = 1 + x + y - x + x * x - 3 * y;

    std::cout << sum.range() << '\n';
    // auto ans = std::vector<real>{
    //    1 * 1 + 1 + 2 * 4 + 1, 2 * 2 + 2 + 2 * 5 + 1, 3 * 3 + 3 + 2 * 6 + 1};
    auto ans = std::vector<real>{-6, -5, -2};

    assert(ranges::equal(sum, ans));

    auto z = scalar_field{};
    z = sum;

    assert(ranges::equal(z, ans));
    assert(ranges::equal(z, sum));

}
