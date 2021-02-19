#include "fields/SystemField.hpp"

namespace ccs::operators
{

struct Divergence {

    template <typename U>
    constexpr auto operator()(U&&)
    {
        return SystemField{};
    }
};
} // namespace ccs::operators