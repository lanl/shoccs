#pragma once

#include "InnerBlock.hpp"

#include <concepts>

namespace ccs::matrix
{
// Block matrix arising from method-of-lines discretization over whole domain.
// Due to the requirements of a cut-cell mesh, the InnerBlocks may not be adjacent to
// eachother.  To simplify construction, a builder class is exposed which computes all
// the zero locations at the end of the construction process
class Block
{
    std::vector<InnerBlock> blocks;

public:
    Block() = default;

    Block(std::vector<InnerBlock>&& blocks) : blocks{std::move(blocks)} {}

    integer rows() const
    {
        if (blocks.empty()) return 0;
        const auto& b = blocks.back();
        // doesn't properly handle case when last point in domain is inside an object
        return b.row_offset() + b.rows() * b.stride();
    }

    void operator()(std::span<const real> x, std::span<real> b) const
    {
        for (auto&& block : blocks) { block(x, b); }
    }

    struct Builder;
};

struct Block::Builder {
    std::vector<InnerBlock> b;

    Builder() = default;

    Builder(integer n) { b.reserve(n); }

    template <typename... Args>
    requires std::constructible_from<InnerBlock, Args...> void
    add_InnerBlock(Args&&... args)
    {
        b.emplace_back(std::forward<Args>(args)...);
    }

    Block to_Block() && { return Block{MOVE(b)}; }
};
} // namespace ccs::matrix
