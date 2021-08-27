#pragma once

#include "types.hpp"

namespace ccs::matrix
{
class dense;
class circulant;
class inner_block;

struct visitor {

    virtual void visit(const dense&) = 0;
    virtual void visit(const circulant&) = 0;
    //  virtual void visit(const inner_block&) = 0;
};
} // namespace ccs::matrix
