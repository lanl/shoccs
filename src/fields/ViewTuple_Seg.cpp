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

    ViewBaseTuple<T&> x{v};

    auto u = T{4, 5, 6};
    ViewBaseTuple<T&> y{u};

    //    x = y;
    resize_and_copy(x, y);
    // std::cout << rs::equal(v, T{4, 5, 6}) << '\n';
}