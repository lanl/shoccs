#pragma once

#include "types.hpp"

namespace ccs
{
class derivative;

struct operator_visitor {
    virtual void visit(const derivative&) = 0;
};
} // namespace ccs
