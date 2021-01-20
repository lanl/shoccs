#pragma once

#include "circulant.hpp"
#include "dense.hpp"
#include "inner_block.hpp"
#include "types.hpp"
#include "zeros.hpp"

#include "fields/r_tuple.hpp"

#include <concepts>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/zip.hpp>

namespace ccs::matrix
{
// Block matrix arising from method-of-lines discretization over whole domain.
// Due to the requirements of a cut-cell mesh, we are required to have the possiblity of
// zeros between the inner_block portions of the block matrix.  To simplify construction,
// a builder class is exposed which computes all the zero locations at the end of the
// construction process
class block
{
    std::vector<inner_block> blocks;
    std::vector<zeros> z;

public:
    block() = default;

    block(std::vector<inner_block>&& blocks, std::vector<zeros>&& z)
        : blocks{std::move(blocks)}, z{std::move(z)}
    {
    }

    int rows() const
    {
        if (blocks.empty()) return 0;
        const auto& b = blocks.back();
        return b.first_row() + b.rows() + z.back().rows();
    }

    struct builder_ {
        std::vector<inner_block> b;
        std::vector<zeros> z;

        builder_() = default;

        builder_(int n) { b.reserve(n); }

        template <typename... Args>
        requires std::constructible_from<inner_block, Args...> void
        add_inner_block(Args&&... args)
        {
            b.emplace_back(std::forward<Args>(args)...);
        }

        block to_block(int nrows) &&
        {
            z.reserve(b.size() + 1);

            int first = 0;
            for (auto&& blk : b) {
                // gap between [first, last) should be filled up with zeros
                z.emplace_back(std::max(0, blk.first_row() - first));
                first = blk.first_row() + blk.rows();
            }
            // handle final zero matrix
            z.emplace_back(std::max(0, nrows - first));

            return block{std::move(b), std::move(z)};
        }
    };

    static builder_ builder(int n = 0) { return n ? builder_{n} : builder_{}; }

private:
    template <ranges::random_access_range R>
    friend constexpr auto operator*(const block& mat, R&& rng)
    {
        return r_tuple{vs::concat(
            mat.z[0] * rng,
            vs::for_each(vs::zip(mat.blocks, mat.z | vs::drop(1)), [rng](auto&& t) {
                auto&& [m, z] = t;
                return rs::yield_from(vs::concat(m * rng, z * rng));
            }))};
    }
};
} // namespace ccs::matrix