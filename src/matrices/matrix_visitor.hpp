#pragma once

#include "types.hpp"

namespace ccs::matrix
{
class dense;
class circulant;
class csr;

struct visitor {

    virtual void visit(const dense&) = 0;
    virtual void visit(const circulant&) = 0;
    virtual void visit(const csr&) = 0;
};
} // namespace ccs::matrix
