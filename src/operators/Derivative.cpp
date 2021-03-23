#include "Derivative.hpp"
#include "fields/Selector.hpp"

namespace ccs::operators
{

Derivative::Derivative(int dir,
                       real h,
                       std::span<const mesh::Line> lines,
                       const stencils::Stencil& stencil,
                       const bcs::Grid& grid_bcs,
                       const bcs::Object& object_bcs)
    : dir{dir}
{
    // query the stencil and allocate memory
    auto [p, rmax, tmax, ex_max] = stencil.query_max();
    // set up the interior stencil
    interior_c.resize(2 * p + 1);
    stencil.interior(h, interior_c);

    // allocate maximum amount of memory required by any boundary conditions
    std::vector<real> left(rmax * tmax);
    std::vector<real> right(rmax * tmax);
    std::vector<real> extra(ex_max);

    auto B_builder = matrix::CSR::Builder();
    auto O_builder = matrix::Block::Builder();

    for (auto&& line : lines) {}

    O = MOVE(O_builder).to_Block();
    B = MOVE(B_builder.to_CSR(0));
}

void Derivative::operator()(field::ScalarView_Const u, field::ScalarView_Mutable du) const
{
    using namespace selector::scalar;
    du = 0;
    O(get<D>(u), get<D>(du));
    // This is ugly
    switch (dir) {
    case 0:
        B(get<Rx>(u), get<Rx>(du));
        break;
    case 1:
        B(get<Ry>(u), get<Ry>(du));
        break;
    default:
        B(get<Rz>(u), get<Rz>(du));
    }
}

} // namespace ccs::operators