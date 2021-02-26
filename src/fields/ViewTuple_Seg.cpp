#include "ViewTuple.hpp"

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/zip_with.hpp>

#include <vector>

#include <iostream>

int main()
{

    using namespace ccs;
    using namespace ccs::field::tuple;
    using T = std::vector<real>;

    auto v = T{1, 2, 3};
    ViewTuple<T&> x{v};

    std::cout << "x == v" << rs::equal(x, v);

    // why is && needed here?
    std::tuple_element<0, ViewTuple<T&>> t;
    auto [yy] = x;
    std::cout << "type(yy): " << debug::type(yy) << '\n';
}