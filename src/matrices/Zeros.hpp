#include "types.hpp"

#include "fields/Tuple.hpp"
#include <range/v3/view/repeat_n.hpp>

namespace ccs::matrix
{
class Zeros
{
    int rows_;

public:
    Zeros() = default;

    constexpr Zeros(int rows) : rows_{rows} {}

    constexpr int rows() const { return rows_; }

private:
    template <ranges::input_range R>
    friend constexpr auto operator*(const Zeros& mat, R&&)
    {
        return field::Tuple{vs::repeat_n(real{}, mat.rows())};
    }
};
} // namespace ccs::matrix