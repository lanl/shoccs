#include "fields/field.hpp"

namespace ccs
{

struct divergence {

    template <typename U>
    constexpr auto operator()(U&&)
    {
        return field{};
    }
};
} // namespace ccs
