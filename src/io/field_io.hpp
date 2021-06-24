#pragma once
#include <array>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "field_data.hpp"
#include "fields/field.hpp"
#include "interval.hpp"
#include "types.hpp"
#include "xdmf.hpp"

#include <sol/forward.hpp>

namespace ccs
{
// Forward decls
class step_controller;

class field_io
{

    xdmf xdmf_w;
    field_data field_data_w;

    d_interval dump_interval;
    std::string io_dir;
    int suffix_length;

public:
    field_io() = default;

    field_io(xdmf&& xdmf_w,
             field_data&& fied_data_w,
             d_interval&& dump_interval,
             std::string&& io_dir,
             int suffix_length);

    bool write(std::span<const std::string>,
               const field& field,
               const step_controller& controller,
               real dt);

    static std::optional<field_io> from_lua(const sol::table&);
};

} // namespace ccs
