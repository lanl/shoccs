#include "r_tuple.hpp"

#include "types.hpp"

#include <range/v3/algorithm/equal.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/zip_with.hpp>

#include <iostream>
#include <vector>

using namespace ccs;

using T = std::vector<real>;

template <int I>
using D = directional_field<T, I>;

int main()
{
    {
        std::cout << "Move with container_tuple\n";
        auto x = detail::container_tuple<T>{vs::iota(0, 10)};
        std::cout << "x.size()" << std::get<0>(x.c).size() << '\n';
        std::cout << "x: " << vs::all(std::get<0>(x.c)) << "\n\n";

        detail::container_tuple<T> y{MOVE(x)};
        std::cout << "y.size()" << std::get<0>(y.c).size() << '\n';
        std::cout << "y: " << vs::all(std::get<0>(y.c)) << "\n\n";
    }

    {
        std::cout << "Move with r_tuple<T>\n";
        auto x = r_tuple<T>{vs::iota(0, 10)};
        std::cout << "x.size()" << x.size() << '\n';
        std::cout << "x: " << x << "\n\n";

        r_tuple<T> y{MOVE(x)};
        std::cout << "y.size()" << y.size() << '\n';
        std::cout << "y: " << y << "\n\n";
    }

    auto d = D<0>{vs::iota(0, 16), int3{2, 4, 2}};
    d[0] = -1;
    {
        std::cout << "moving r_tuple<C&>\n";
        auto x = r_tuple{d};
        std::cout << std::same_as<decltype(x), r_tuple<D<0>&>> << '\n';
        std::cout << "x.size(): " << x.size() << '\n';
        auto v = view<0>(x);
        std::cout << v << '\n';

        r_tuple<D<0>&> y{MOVE(x)};
        std::cout << "y.size(): " << y.size() << '\n';
        std::cout << view<0>(y) << "\n\n";
    }
    {
        std::cout << "moveing r_tuple<C>\n";
        auto y = r_tuple{MOVE(d)};
        std::cout << std::same_as<decltype(y), r_tuple<D<0>>> << '\n';
        std::cout << "y.size(): " << y.size() << '\n';
        auto v = view<0>(y);
        std::cout << v << '\n';

        const auto& c = y.container().c;
        std::cout
            << std::same_as<std::remove_cvref_t<decltype(c)>, std::tuple<D<0>>> << '\n';
    }


    {
        auto y = r_tuple{D<0>{vs::iota(0, 16), int3{2, 4, 2}}};
        std::cout << std::same_as<decltype(y), r_tuple<D<0>>> << '\n';
        std::cout << "y.size(): " << y.size() << '\n';
        auto v = view<0>(y);
        std::cout << v << '\n';
    }

    // REQUIRE(v[0] == 0);
    // REQUIRE(v[15] == 15);

    // auto y = x + 1;

    // REQUIRE(x[0] == 1);
    // REQUIRE(x[15] == 16);
}