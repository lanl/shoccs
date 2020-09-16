#include "types.hpp"
#include "range_operators.hpp"

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
        return ranges::views::repeat_n(real{}, mat.rows());
    }
};
} // namespace ccs::matrix