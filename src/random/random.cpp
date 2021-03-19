#include "random.hpp"

namespace ccs
{

std::default_random_engine& global_urng()
{
    static std::default_random_engine u{};
    return u;
}

void randomize()
{
    static std::random_device rd{};
    global_urng().seed(rd());
}

int pick(int from, int thru)
{
    static std::uniform_int_distribution<> d{};
    using param_t = decltype(d)::param_type;
    return d(global_urng(), param_t{from, thru});
}

real pick(real from, real upto)
{
    static std::uniform_real_distribution<real> d{};
    using param_t = decltype(d)::param_type;
    return d(global_urng(), param_t{from, upto});
}

real pick_r(real from, real upto) {
    return pick(from, upto);
}

} // namespace ccs