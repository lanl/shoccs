#pragma once

#include "matrices/coefficient_visitor.hpp"
#include "operator_visitor.hpp"

namespace ccs
{
class eigenvalue_visitor : public operator_visitor
{
    matrix::unit_stride_visitor u;
    matrix::coefficient_visitor v;
    std::vector<real> eigs_real, eigs_imag;

public:
    eigenvalue_visitor() = default;

    template <Range Rx, Range Ry, Range Rz>
    eigenvalue_visitor(int3 nxyz, Rx&& rx, Ry&& ry, Rz&& rz)
        : u{nxyz, FWD(rx), FWD(ry), FWD(rz)}
    {
        assert(nxyz[2] == 1);
        assert(nxyz[1] == 1);
    }

    void visit(const derivative&) override;

    std::span<const real> eigenvalues_real() const;
    std::span<const real> eigenvalues_imag() const;
};
} // namespace ccs
