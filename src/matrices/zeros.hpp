#include "types.hpp"

#include "fields/r_tuple.hpp"
#include <range/v3/view/repeat_n.hpp>

namespace ccs::matrix
{
class zeros
{
    int rows_;

public:
    zeros() = default;

    zeros(int rows) : rows_{rows} {}

    constexpr int rows() const { return rows_; }

private:
    template <ranges::input_range R>
    friend constexpr auto operator*(const zeros& mat, [[maybe_unused]] R&& rng)
    {        
        return r_tuple{vs::repeat_n(real{}, mat.rows())};
    }
};
} // namespace ccs::matrix